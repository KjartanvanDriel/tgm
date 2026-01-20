"""Benchmark comparing original vs GPU-optimized eval procedures."""

import argparse
import time
from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
from tgb.linkproppred.evaluate import Evaluator
from tgm import DGBatch, DGraph
from tgm.constants import (
    METRIC_TGB_LINKPROPPRED,
    PADDED_NODE_ID,
    RECIPE_TGB_LINK_PRED,
)
from tgm.data import DGData, DGDataLoader
from tgm.hooks import RecencyNeighborHook, RecipeRegistry, StatelessHook
from tgm.nn import LinkPredictor, MLPMixer, Time2Vec
from tgm.util.seed import seed_everything
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Benchmark Eval Procedures',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--dataset', type=str, default='tgbl-wiki')
parser.add_argument('--bsize', type=int, default=200)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--n-nbrs', type=int, default=20)
parser.add_argument('--time-dim', type=int, default=100)
parser.add_argument('--embed-dim', type=int, default=128)
parser.add_argument('--node-dim', type=int, default=100)
parser.add_argument('--time-gap', type=int, default=2000)
parser.add_argument('--num-batches', type=int, default=10, help='Number of batches to benchmark')

args = parser.parse_args()


# ============================================================================
# Memory tracking utilities
# ============================================================================

class MemoryTracker:
    def __init__(self, device: str):
        self.device = device
        self.is_cuda = device.startswith('cuda')
        self.peak_memory = 0
        self.allocations = []

    def reset(self):
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self.peak_memory = 0
        self.allocations = []

    def snapshot(self, label: str = ""):
        if self.is_cuda:
            torch.cuda.synchronize()
            current = torch.cuda.memory_allocated() / 1024**2  # MB
            peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
            self.allocations.append((label, current, peak))
            self.peak_memory = max(self.peak_memory, peak)
            return current, peak
        return 0, 0

    def report(self):
        if not self.is_cuda:
            print("Memory tracking only available on CUDA devices")
            return
        print(f"\nPeak memory: {self.peak_memory:.2f} MB")
        for label, current, peak in self.allocations:
            print(f"  {label}: current={current:.2f} MB, peak={peak:.2f} MB")


# ============================================================================
# Hooks (need both versions for comparison)
# ============================================================================

class OriginalGraphMixerHook(StatelessHook):
    """Original CPU-based hook using Python dicts."""

    requires = {'neg'}
    produces = {'time_gap_nbrs'}

    def __init__(self, time_gap: int) -> None:
        self._time_gap = time_gap

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        time_gap_slice = replace(dg._slice)
        time_gap_slice.start_idx = max(dg._slice.end_idx - self._time_gap, 0)
        time_gap_slice.end_time = int(batch.edge_time.min()) - 1
        time_gap_src, time_gap_dst, _ = dg._storage.get_edges(time_gap_slice)

        nbr_index = defaultdict(list)
        for u, v in zip(time_gap_src.tolist(), time_gap_dst.tolist()):
            nbr_index[u].append(v)
            nbr_index[v].append(u)

        seed_nodes = torch.cat(
            [batch.edge_src, batch.edge_dst, batch.neg.to(dg.device)]
        )
        batch.time_gap_nbrs = [nbr_index.get(nid, []) for nid in seed_nodes.tolist()]
        return batch


def _isin_sorted_unique(x: torch.Tensor, su: torch.Tensor) -> torch.Tensor:
    idx = torch.searchsorted(su, x)
    idx = idx.clamp_max(su.numel() - 1)
    return su[idx] == x


class GPUTimeGapHook(StatelessHook):
    """GPU-optimized hook using tensor operations."""

    requires = {'neg'}
    produces = {'time_gap_nbrs'}

    def __init__(self, time_gap: int):
        self._time_gap = int(time_gap)

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        device = dg.device

        time_gap_slice = replace(dg._slice)
        time_gap_slice.start_idx = max(dg._slice.end_idx - self._time_gap, 0)
        time_gap_slice.end_time = int(batch.edge_time.min()) - 1

        src, dst, _ = dg._storage.get_edges(time_gap_slice)
        src = src.to(device, non_blocking=True)
        dst = dst.to(device, non_blocking=True)

        seed_nodes = torch.cat(
            [batch.edge_src, batch.edge_dst, batch.neg.to(device)], dim=0
        )
        seed_su = torch.sort(torch.unique(seed_nodes)).values

        src_mask = _isin_sorted_unique(src, seed_su)
        dst_mask = _isin_sorted_unique(dst, seed_su)

        src2 = torch.cat([src[src_mask], dst[dst_mask]], dim=0)
        dst2 = torch.cat([dst[src_mask], src[dst_mask]], dim=0)

        row = torch.searchsorted(seed_su, src2)
        row_occ = torch.searchsorted(seed_su, seed_nodes)

        batch.time_gap_nbrs = (seed_nodes, seed_su, row, row_occ, dst2)
        return batch


# ============================================================================
# Encoder variants
# ============================================================================

class OriginalGraphMixerEncoder(nn.Module):
    """Original encoder with Python loop for time_gap_feat."""

    def __init__(self, time_dim, embed_dim, num_tokens, node_dim, edge_dim,
                 num_layers=2, token_dim_expansion=0.5, channel_dim_expansion=4.0, dropout=0.1):
        super().__init__()
        self.time_encoder = Time2Vec(time_dim=time_dim)
        for param in self.time_encoder.parameters():
            param.requires_grad = False

        self.projection_layer = nn.Linear(edge_dim + time_dim, edge_dim)
        self.mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=num_tokens, num_channels=edge_dim,
                     token_dim_expansion_factor=token_dim_expansion,
                     channel_dim_expansion_factor=channel_dim_expansion, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(edge_dim + node_dim, embed_dim)

    def forward(self, batch: DGBatch, node_feat: torch.Tensor) -> torch.Tensor:
        edge_feat = batch.nbr_edge_x[0]
        nbr_time_feat = self.time_encoder(batch.seed_times[0][:, None] - batch.nbr_edge_time[0])
        z_link = self.projection_layer(torch.cat([edge_feat, nbr_time_feat], dim=-1))
        for mixer in self.mlp_mixers:
            z_link = mixer(z_link)

        valid_nbrs_mask = batch.nbr_nids[0] != PADDED_NODE_ID
        z_link = z_link * valid_nbrs_mask.unsqueeze(-1)
        z_link = z_link.sum(dim=1) / valid_nbrs_mask.sum(dim=1, keepdim=True).clamp(min=1)

        # Original Python loop
        num_nodes, feat_dim = len(batch.time_gap_nbrs), node_feat.shape[1]
        time_gap_feat = torch.zeros((num_nodes, feat_dim), device=node_feat.device)
        for i in range(num_nodes):
            if batch.time_gap_nbrs[i]:
                time_gap_feat[i] = node_feat[batch.time_gap_nbrs[i]].mean(dim=0)

        z_node = time_gap_feat + node_feat[torch.cat([batch.edge_src, batch.edge_dst, batch.neg])]
        return self.output_layer(torch.cat([z_link, z_node], dim=1))


class GPUGraphMixerEncoder(nn.Module):
    """GPU encoder using scatter_reduce for time_gap_feat."""

    def __init__(self, time_dim, embed_dim, num_tokens, node_dim, edge_dim,
                 num_layers=2, token_dim_expansion=0.5, channel_dim_expansion=4.0, dropout=0.1):
        super().__init__()
        self.time_encoder = Time2Vec(time_dim=time_dim)
        for param in self.time_encoder.parameters():
            param.requires_grad = False

        self.projection_layer = nn.Linear(edge_dim + time_dim, edge_dim)
        self.mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=num_tokens, num_channels=edge_dim,
                     token_dim_expansion_factor=token_dim_expansion,
                     channel_dim_expansion_factor=channel_dim_expansion, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(edge_dim + node_dim, embed_dim)

    def forward(self, batch: DGBatch, node_feat: torch.Tensor) -> torch.Tensor:
        edge_feat = batch.nbr_edge_x[0]
        nbr_time_feat = self.time_encoder(batch.seed_times[0][:, None] - batch.nbr_edge_time[0])
        z_link = self.projection_layer(torch.cat([edge_feat, nbr_time_feat], dim=-1))
        for mixer in self.mlp_mixers:
            z_link = mixer(z_link)

        valid_nbrs_mask = batch.nbr_nids[0] != PADDED_NODE_ID
        z_link = z_link * valid_nbrs_mask.unsqueeze(-1)
        z_link = z_link.sum(dim=1) / valid_nbrs_mask.sum(dim=1, keepdim=True).clamp(min=1)

        # GPU scatter_reduce with reduce='mean'
        seed_nodes, seed_su, row, row_occ, col = batch.time_gap_nbrs
        num_rows = seed_su.numel()
        D = node_feat.shape[1]

        mean_u = torch.zeros(num_rows, D, device=node_feat.device, dtype=node_feat.dtype)
        mean_u.scatter_reduce_(0, row.unsqueeze(1).expand(-1, D), node_feat[col], reduce='mean')
        time_gap_feat = mean_u[row_occ]

        z_node = time_gap_feat + node_feat[torch.cat([batch.edge_src, batch.edge_dst, batch.neg])]
        return self.output_layer(torch.cat([z_link, z_node], dim=1))


# ============================================================================
# Eval functions
# ============================================================================

@torch.no_grad()
def eval_original(
    loader: DGDataLoader, encoder: nn.Module, decoder: nn.Module,
    evaluator: Evaluator, num_batches: int = None
) -> tuple[float, float]:
    """Original eval with Python dict and per-edge loop."""
    encoder.eval()
    decoder.eval()
    perf_list = []
    static_node_x = loader.dgraph.static_node_x
    total_time = 0

    for i, batch in enumerate(tqdm(loader, desc="Original eval")):
        if num_batches and i >= num_batches:
            break

        start = time.perf_counter()
        z = encoder(batch, static_node_x)
        id_map = {nid.item(): i for i, nid in enumerate(batch.seed_nids[0])}

        for idx, neg_batch in enumerate(batch.neg_batch_list):
            dst_ids = torch.cat([batch.edge_dst[idx].unsqueeze(0), neg_batch])
            src_ids = batch.edge_src[idx].repeat(len(dst_ids))

            src_idx = torch.tensor([id_map[n.item()] for n in src_ids], device=z.device)
            dst_idx = torch.tensor([id_map[n.item()] for n in dst_ids], device=z.device)
            z_src = z[src_idx]
            z_dst = z[dst_idx]
            y_pred = decoder(z_src, z_dst).sigmoid()

            input_dict = {
                'y_pred_pos': y_pred[0],
                'y_pred_neg': y_pred[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])

        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start

    return float(np.mean(perf_list)) if perf_list else 0.0, total_time


@torch.no_grad()
def eval_gpu(
    loader: DGDataLoader, encoder: nn.Module, decoder: nn.Module,
    evaluator: Evaluator, num_batches: int = None
) -> tuple[float, float]:
    """GPU eval with batched tensor operations."""
    encoder.eval()
    decoder.eval()
    perf_list = []
    static_node_x = loader.dgraph.static_node_x
    total_time = 0

    for i, batch in enumerate(tqdm(loader, desc="GPU eval")):
        if num_batches and i >= num_batches:
            break

        start = time.perf_counter()
        z = encoder(batch, static_node_x)
        device = z.device

        seed = batch.seed_nids[0].to(device)
        seed_sorted, perm = seed.sort()

        src_e = batch.edge_src.to(device)
        pos_dst_e = batch.edge_dst.to(device)
        E = src_e.numel()

        neg_list = [t.to(device) for t in batch.neg_batch_list]
        lengths = torch.tensor([1 + t.numel() for t in neg_list], device=device)

        src_flat = torch.repeat_interleave(src_e, lengths)

        dst_chunks = [
            torch.cat([pos_dst_e[i : i + 1], neg_list[i]], dim=0) for i in range(E)
        ]
        dst_flat = torch.cat(dst_chunks, dim=0)

        src_pos = torch.searchsorted(seed_sorted, src_flat)
        dst_pos = torch.searchsorted(seed_sorted, dst_flat)

        src_idx = perm[src_pos]
        dst_idx = perm[dst_pos]

        y_flat = decoder(z[src_idx], z[dst_idx]).sigmoid()

        y_chunks = torch.split(y_flat, lengths.tolist())
        for yi in y_chunks:
            input_dict = {
                'y_pred_pos': yi[0],
                'y_pred_neg': yi[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])

        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start

    return float(np.mean(perf_list)) if perf_list else 0.0, total_time


# ============================================================================
# Main benchmark
# ============================================================================

def main():
    print(f"Device: {args.device}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.bsize}")
    print(f"Num batches to benchmark: {args.num_batches}")
    print("-" * 60)

    seed_everything(args.seed)
    evaluator = Evaluator(name=args.dataset)

    full_data = DGData.from_tgb(args.dataset)
    if full_data.static_node_x is None:
        full_data.static_node_x = torch.randn(
            (full_data.num_nodes, args.node_dim), device=args.device
        )

    _, val_data, _ = full_data.split()
    val_dg = DGraph(val_data, device=args.device)

    # Setup for original implementation
    hm_original = RecipeRegistry.build(
        RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=val_dg
    )
    _, val_key_orig, _ = hm_original.keys
    hm_original.register_shared(OriginalGraphMixerHook(args.time_gap))
    hm_original.register_shared(
        RecencyNeighborHook(
            num_nbrs=[args.n_nbrs],
            num_nodes=full_data.num_nodes,
            seed_nodes_keys=['edge_src', 'edge_dst', 'neg'],
            seed_times_keys=['edge_time', 'edge_time', 'neg_time'],
        )
    )
    val_loader_original = DGDataLoader(val_dg, args.bsize, hook_manager=hm_original)

    # Setup for GPU implementation
    hm_gpu = RecipeRegistry.build(
        RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=val_dg
    )
    _, val_key_gpu, _ = hm_gpu.keys
    hm_gpu.register_shared(GPUTimeGapHook(args.time_gap))
    hm_gpu.register_shared(
        RecencyNeighborHook(
            num_nbrs=[args.n_nbrs],
            num_nodes=full_data.num_nodes,
            seed_nodes_keys=['edge_src', 'edge_dst', 'neg'],
            seed_times_keys=['edge_time', 'edge_time', 'neg_time'],
        )
    )
    val_loader_gpu = DGDataLoader(val_dg, args.bsize, hook_manager=hm_gpu)

    # Create encoders and decoder
    encoder_original = OriginalGraphMixerEncoder(
        node_dim=val_dg.static_node_x_dim,
        edge_dim=val_dg.edge_x_dim,
        time_dim=args.time_dim,
        embed_dim=args.embed_dim,
        num_tokens=args.n_nbrs,
    ).to(args.device)

    encoder_gpu = GPUGraphMixerEncoder(
        node_dim=val_dg.static_node_x_dim,
        edge_dim=val_dg.edge_x_dim,
        time_dim=args.time_dim,
        embed_dim=args.embed_dim,
        num_tokens=args.n_nbrs,
    ).to(args.device)

    # Copy weights from original to GPU encoder for fair comparison
    encoder_gpu.load_state_dict(encoder_original.state_dict())

    decoder = LinkPredictor(node_dim=args.embed_dim, hidden_dim=args.embed_dim).to(args.device)

    memory_tracker = MemoryTracker(args.device)

    # Warmup
    print("\nWarming up...")
    with hm_original.activate(val_key_orig):
        for i, batch in enumerate(val_loader_original):
            if i >= 2:
                break
            _ = encoder_original(batch, val_loader_original.dgraph.static_node_x)

    # Benchmark original
    print("\n" + "=" * 60)
    print("ORIGINAL IMPLEMENTATION (Python dict + per-edge loop)")
    print("=" * 60)
    memory_tracker.reset()
    with hm_original.activate(val_key_orig):
        mrr_orig, time_orig = eval_original(
            val_loader_original, encoder_original, decoder, evaluator, args.num_batches
        )
    memory_tracker.snapshot("After original eval")
    print(f"MRR: {mrr_orig:.4f}")
    print(f"Total time: {time_orig*1000:.2f} ms")
    print(f"Avg per batch: {time_orig*1000/args.num_batches:.2f} ms")

    # Reset hook state
    hm_original.reset_state()
    hm_gpu.reset_state()

    # Benchmark GPU
    print("\n" + "=" * 60)
    print("GPU IMPLEMENTATION (tensor ops + batched decoder)")
    print("=" * 60)
    memory_tracker.reset()
    with hm_gpu.activate(val_key_gpu):
        mrr_gpu, time_gpu = eval_gpu(
            val_loader_gpu, encoder_gpu, decoder, evaluator, args.num_batches
        )
    memory_tracker.snapshot("After GPU eval")
    print(f"MRR: {mrr_gpu:.4f}")
    print(f"Total time: {time_gpu*1000:.2f} ms")
    print(f"Avg per batch: {time_gpu*1000/args.num_batches:.2f} ms")
    print(f"Speedup vs original: {time_orig/time_gpu:.2f}x")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Implementation':<40} {'Time (ms)':<15} {'Speedup':<10} {'MRR':<10}")
    print("-" * 75)
    print(f"{'Original (Python dict + per-edge loop)':<40} {time_orig*1000:<15.2f} {'1.00x':<10} {mrr_orig:<10.4f}")
    print(f"{'GPU (tensor ops + batched decoder)':<40} {time_gpu*1000:<15.2f} {f'{time_orig/time_gpu:.2f}x':<10} {mrr_gpu:<10.4f}")

    # Verify correctness
    print("\n" + "=" * 60)
    print("CORRECTNESS CHECK")
    print("=" * 60)
    mrr_diff = abs(mrr_orig - mrr_gpu)
    print(f"MRR diff (original vs GPU): {mrr_diff:.6f} {'✓' if mrr_diff < 0.001 else '✗'}")

    memory_tracker.report()


if __name__ == '__main__':
    main()
