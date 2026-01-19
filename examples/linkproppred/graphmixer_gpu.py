import argparse
from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from tgm.util.logging import enable_logging, log_gpu, log_latency, log_metric
from tgm.util.seed import seed_everything
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='GraphMixer LinkPropPred Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed to use')
parser.add_argument('--dataset', type=str, default='tgbl-wiki', help='Dataset name')
parser.add_argument('--bsize', type=int, default=200, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='torch device')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout rate')
parser.add_argument('--n-nbrs', type=int, default=20, help='num sampled nbrs')
parser.add_argument(
    '--time-dim',
    type=int,
    default=100,
    help='time encoding dimension',
)
parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
parser.add_argument(
    '--node-dim', type=int, default=100, help='node feat dimension if not provided'
)
parser.add_argument(
    '--time-gap',
    type=int,
    default=2000,
    help='graphmixer time slot size',
)
parser.add_argument(
    '--token-dim-expansion',
    type=float,
    default=0.5,
    help='token dimension expansion factor in MLP sub-blocks',
)
parser.add_argument(
    '--channel-dim-expansion',
    type=float,
    default=4.0,
    help='channel dimension expansion factor in MLP sub-blocks',
)
parser.add_argument(
    '--log-file-path', type=str, default=None, help='Optional path to write logs'
)

args = parser.parse_args()
enable_logging(log_file_path=args.log_file_path)


def _isin_via_searchsorted(
    x: torch.Tensor, values_sorted_unique: torch.Tensor
) -> torch.Tensor:
    """
    GPU-friendly membership test:
    values_sorted_unique must be sorted & unique.
    Returns mask same shape as x.
    """
    idx = torch.searchsorted(values_sorted_unique, x)
    idx = idx.clamp_max(values_sorted_unique.numel() - 1)
    return values_sorted_unique[idx] == x


def build_csr_for_nodes(
    src: torch.Tensor,
    dst: torch.Tensor,
    nodes_sorted_unique: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build CSR adjacency for *rows = nodes_sorted_unique*.
    Returns (row_ptr, col) where:
      row_ptr: (R+1,)
      col: (nnz,)
    Rows are in the same order as nodes_sorted_unique.
    """
    # Map src node ids -> row ids in [0, R)
    row = torch.searchsorted(nodes_sorted_unique, src)
    # src is guaranteed in nodes_sorted_unique for kept edges
    # Sort edges by row to build CSR
    perm = torch.argsort(row)
    row = row[perm]
    col = dst[perm]

    R = nodes_sorted_unique.numel()
    counts = torch.bincount(row, minlength=R)
    row_ptr = torch.empty(R + 1, device=src.device, dtype=torch.long)
    row_ptr[0] = 0
    row_ptr[1:] = torch.cumsum(counts, dim=0)
    return row_ptr, col


from dataclasses import replace

import torch
from torch_scatter import scatter_mean


def _isin_sorted_unique(x: torch.Tensor, su: torch.Tensor) -> torch.Tensor:
    """
    x: (...,)
    su: (R,) sorted unique
    returns: mask (...,) where mask[i]=True iff x[i] in su
    """
    idx = torch.searchsorted(su, x)
    # clamp to valid range (handles idx==R)
    idx = idx.clamp_max(su.numel() - 1)
    return su[idx] == x


class TimeGapSeedNbrAggHook(StatelessHook):
    """
    Produces a scatter-friendly representation of time-gap "neighbors of seed nodes"
    without building Python lists or CSR.

    Outputs attached to batch:
      - batch.time_gap_seed_nodes      : (B,) seed nodes in original order (with duplicates)
      - batch.time_gap_seed_su         : (R,) sorted unique seed nodes
      - batch.time_gap_row_occ         : (B,) maps each seed occurrence -> unique seed row id in [0..R-1]
      - batch.time_gap_row             : (nnz2,) row ids for scatter (unique seed row per edge endpoint)
      - batch.time_gap_col             : (nnz2,) neighbor node ids corresponding to row entries
    """

    requires = {'neg'}
    produces = {'time_gap_nbrs'}

    def __init__(self, time_gap: int):
        self._time_gap = int(time_gap)

    def __call__(self, dg: 'DGraph', batch: 'DGBatch') -> 'DGBatch':
        device = dg.device

        # 1) Build time-gap slice
        time_gap_slice = replace(dg._slice)
        time_gap_slice.start_idx = max(dg._slice.end_idx - self._time_gap, 0)
        time_gap_slice.end_time = int(batch.edge_time.min()) - 1

        # Get edges in that window (src, dst are assumed 1D tensors)
        src, dst, _ = dg._storage.get_edges(time_gap_slice)
        src = src.to(device, non_blocking=True)
        dst = dst.to(device, non_blocking=True)

        # 2) Seeds (order preserved, duplicates preserved)
        seed_nodes = torch.cat(
            [batch.edge_src, batch.edge_dst, batch.neg.to(device)], dim=0
        )  # (B,)

        # Unique seeds (sorted for searchsorted)
        seed_su = torch.sort(torch.unique(seed_nodes)).values  # (R,)

        # 3) Ensure every element of src2 is in seeds by construction:
        #    - if src is seed, keep directed edge (src -> dst)
        #    - if dst is seed, keep directed edge (dst -> src)
        src_mask = _isin_sorted_unique(src, seed_su)
        dst_mask = _isin_sorted_unique(dst, seed_su)

        src2 = torch.cat([src[src_mask], dst[dst_mask]], dim=0)
        dst2 = torch.cat([dst[src_mask], src[dst_mask]], dim=0)

        # 4) Map src2 (seed endpoint) to [0..R-1] for scatter
        row = torch.searchsorted(seed_su, src2)  # (nnz2,)

        # 5) Map each seed occurrence to [0..R-1] to preserve duplicates + order later
        row_occ = torch.searchsorted(seed_su, seed_nodes)  # (B,)

        batch.time_gap_nbrs = (
            seed_nodes,
            seed_su,
            row,
            row_occ,
            dst2,
        )
        return batch


def compute_time_gap_mean_features(
    node_feat: torch.Tensor, batch: 'DGBatch'
) -> torch.Tensor:
    """
    Returns:
      time_gap_feat: (B, D) features aligned to batch.time_gap_seed_nodes (original order, duplicates preserved)

    Definition:
      For each seed occurrence i (in original seed_nodes order),
      time_gap_feat[i] = mean_{nbr in time-gap neighbors of that seed node} node_feat[nbr]

    If a seed has zero time-gap neighbors, its mean row will be zero.
    """
    seed_nodes, seed_su, row, row_occ, col = batch.time_gap_nbrs
    num_rows = seed_su.numel()
    mean_u = scatter_mean(
        node_feat[col], row, dim=0, dim_size=num_rows
    )  # (num_rows, D)
    return mean_u[row_occ]  # (B, D)


class GraphMixerEncoder(nn.Module):
    def __init__(
        self,
        time_dim: int,
        embed_dim: int,
        num_tokens: int,
        node_dim: int,
        edge_dim: int,
        num_layers: int = 2,
        token_dim_expansion: float = 0.5,
        channel_dim_expansion: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # GraphMixer edge_time encoding function is not trainable
        self.time_encoder = Time2Vec(time_dim=time_dim)
        for param in self.time_encoder.parameters():
            param.requires_grad = False

        self.projection_layer = nn.Linear(edge_dim + time_dim, edge_dim)
        self.mlp_mixers = nn.ModuleList(
            [
                MLPMixer(
                    num_tokens=num_tokens,
                    num_channels=edge_dim,
                    token_dim_expansion_factor=token_dim_expansion,
                    channel_dim_expansion_factor=channel_dim_expansion,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Linear(
            in_features=edge_dim + node_dim, out_features=embed_dim
        )

    def forward(self, batch: DGBatch, node_feat: torch.Tensor) -> torch.Tensor:
        # Link Encoder
        edge_feat = batch.nbr_edge_x[0]
        nbr_time_feat = self.time_encoder(
            batch.seed_times[0][:, None] - batch.nbr_edge_time[0]
        )
        z_link = self.projection_layer(torch.cat([edge_feat, nbr_time_feat], dim=-1))
        for mixer in self.mlp_mixers:
            z_link = mixer(z_link)

        valid_nbrs_mask = batch.nbr_nids[0] != PADDED_NODE_ID
        z_link = z_link * valid_nbrs_mask.unsqueeze(-1)
        z_link = z_link.sum(dim=1) / valid_nbrs_mask.sum(dim=1, keepdim=True).clamp(
            min=1
        )

        time_gap_feat = compute_time_gap_mean_features(node_feat, batch)

        z_node = (
            time_gap_feat
            + node_feat[torch.cat([batch.edge_src, batch.edge_dst, batch.neg])]
        )
        z = self.output_layer(torch.cat([z_link, z_node], dim=1))
        return z


@log_gpu
@log_latency
def train(
    loader: DGDataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    opt: torch.optim.Optimizer,
) -> float:
    encoder.train()
    decoder.train()
    total_loss = 0
    static_node_x = loader.dgraph.static_node_x

    for batch in tqdm(loader):
        opt.zero_grad()

        z = encoder(batch, static_node_x)
        z_src, z_dst, z_neg = torch.chunk(z, 3)

        pos_out = decoder(z_src, z_dst)
        neg_out = decoder(z_src, z_neg)

        loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        loss += F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        loss.backward()
        opt.step()
        total_loss += float(loss)
    return total_loss


@log_gpu
@log_latency
@torch.no_grad()
def eval(
    loader: DGDataLoader, encoder: nn.Module, decoder: nn.Module, evaluator: Evaluator
) -> float:
    encoder.eval()
    decoder.eval()
    perf_list = []
    static_node_x = loader.dgraph.static_node_x

    for batch in tqdm(loader):
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

    return float(np.mean(perf_list))


seed_everything(args.seed)
evaluator = Evaluator(name=args.dataset)

full_data = DGData.from_tgb(args.dataset)
print(f'full_data: {full_data}')
print('Attributes of full_data:')
for attr in dir(full_data):
    if not attr.startswith('_'):
        print(f'{attr}')
if full_data.static_node_x is None:
    full_data.static_node_x = torch.randn(
        (full_data.num_nodes, args.node_dim), device=args.device
    )

train_data, val_data, test_data = full_data.split()
train_dg = DGraph(train_data, device=args.device)
val_dg = DGraph(val_data, device=args.device)
test_dg = DGraph(test_data, device=args.device)

hm = RecipeRegistry.build(
    RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
)
train_key, val_key, test_key = hm.keys
hm.register_shared(TimeGapSeedNbrAggHook(args.time_gap))
hm.register_shared(
    RecencyNeighborHook(
        num_nbrs=[args.n_nbrs],
        num_nodes=full_data.num_nodes,
        seed_nodes_keys=['edge_src', 'edge_dst', 'neg'],
        seed_times_keys=['edge_time', 'edge_time', 'neg_time'],
    )
)

train_loader = DGDataLoader(train_dg, args.bsize, hook_manager=hm)
val_loader = DGDataLoader(val_dg, args.bsize, hook_manager=hm)
test_loader = DGDataLoader(test_dg, args.bsize, hook_manager=hm)

encoder = GraphMixerEncoder(
    node_dim=train_dg.static_node_x_dim,
    edge_dim=train_dg.edge_x_dim,
    time_dim=args.time_dim,
    embed_dim=args.embed_dim,
    num_tokens=args.n_nbrs,
    token_dim_expansion=float(args.token_dim_expansion),
    channel_dim_expansion=float(args.channel_dim_expansion),
    dropout=float(args.dropout),
).to(args.device)
decoder = LinkPredictor(node_dim=args.embed_dim, hidden_dim=args.embed_dim).to(
    args.device
)
opt = torch.optim.Adam(
    set(encoder.parameters()) | set(decoder.parameters()), lr=float(args.lr)
)

for epoch in range(1, args.epochs + 1):
    with hm.activate(train_key):
        loss = train(train_loader, encoder, decoder, opt)

    with hm.activate(val_key):
        val_mrr = eval(val_loader, encoder, decoder, evaluator)

    log_metric('Loss', loss, epoch=epoch)
    log_metric(f'Validation {METRIC_TGB_LINKPROPPRED}', val_mrr, epoch=epoch)

    if epoch < args.epochs:  # Reset hooks after each epoch, except last epoch
        hm.reset_state()

with hm.activate('test'):
    test_mrr = eval(test_loader, encoder, decoder, evaluator)
log_metric(f'Test {METRIC_TGB_LINKPROPPRED}', test_mrr, epoch=args.epochs)
