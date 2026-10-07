import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class MultiSourceAttentionFusion(nn.Module):
    def __init__(self, hidden_size, dropout=0.5):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.a = nn.Linear(hidden_size, 4, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        self.test_attention_weights = None

    def _init_weights(self):
        nn.init.xavier_normal_(self.W.weight)
        nn.init.xavier_normal_(self.a.weight)

    def forward(self, h_intra_same, h_intra_diff, h_cross_same, h_cross_diff):
        w_intra_same = self.W(h_intra_same)
        w_intra_diff = self.W(h_intra_diff)
        w_cross_same = self.W(h_cross_same)
        w_cross_diff = self.W(h_cross_diff)

        e_intra_same = self.a(torch.tanh(w_intra_same))
        e_intra_diff = self.a(torch.tanh(w_intra_diff))
        e_cross_same = self.a(torch.tanh(w_cross_same))
        e_cross_diff = self.a(torch.tanh(w_cross_diff))

        combined_scores = (e_intra_same + e_intra_diff + e_cross_same + e_cross_diff) / 4.0
        attention_weights = F.softmax(combined_scores, dim=1)

        alpha_intra_same = attention_weights[:, 0].unsqueeze(1)
        alpha_intra_diff = attention_weights[:, 1].unsqueeze(1)
        alpha_cross_same = attention_weights[:, 2].unsqueeze(1)
        alpha_cross_diff = attention_weights[:, 3].unsqueeze(1)

        fused = (alpha_intra_same * w_intra_same
                 + alpha_intra_diff * w_intra_diff
                 + alpha_cross_same * w_cross_same
                 + alpha_cross_diff * w_cross_diff)

        if not self.training:
            self.test_attention_weights = attention_weights.detach().cpu()
        return self.dropout(self.layer_norm(fused))


class HeteroGraphAttention(nn.Module):
    def __init__(self, in_features, hidden_size, num_heads=2, dropout=0.5):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.gat_intra_same = GATConv(in_features, hidden_size, heads=num_heads, dropout=dropout)
        self.gat_intra_diff = GATConv(in_features, hidden_size, heads=num_heads, dropout=dropout)
        self.gat_cross_same = GATConv(in_features, hidden_size, heads=num_heads, dropout=dropout)
        self.gat_cross_diff = GATConv(in_features, hidden_size, heads=num_heads, dropout=dropout)

        self.attention_fusion = MultiSourceAttentionFusion(hidden_size * num_heads, dropout=dropout)

        if in_features != hidden_size:
            self.residual_fc = nn.Linear(in_features, hidden_size)
        else:
            self.residual_fc = nn.Identity()

        self.fc = nn.Linear(hidden_size * num_heads, hidden_size)
        self._init_weights()

    def _init_weights(self):
        if isinstance(self.residual_fc, nn.Linear):
            nn.init.xavier_normal_(self.residual_fc.weight)
            nn.init.constant_(self.residual_fc.bias, 0)

    def forward(self, x, edge_indices):
        edge_intra_same, edge_intra_diff, edge_cross_same, edge_cross_diff = edge_indices

        h_intra_same, (ei_is, att_is) = self.gat_intra_same(
            x, edge_intra_same, return_attention_weights=True)
        h_intra_diff, (ei_id, att_id) = self.gat_intra_diff(
            x, edge_intra_diff, return_attention_weights=True)
        h_cross_same, (ei_cs, att_cs) = self.gat_cross_same(
            x, edge_cross_same, return_attention_weights=True)
        h_cross_diff, (ei_cd, att_cd) = self.gat_cross_diff(
            x, edge_cross_diff, return_attention_weights=True)

        if not self.training:
            self.edge_attention = {
                'intra_same': (ei_is.detach().cpu(), att_is.detach().cpu()),
                'intra_diff': (ei_id.detach().cpu(), att_id.detach().cpu()),
                'cross_same': (ei_cs.detach().cpu(), att_cs.detach().cpu()),
                'cross_diff': (ei_cd.detach().cpu(), att_cd.detach().cpu()),
            }

        h_intra_same = F.elu(h_intra_same)
        h_intra_diff = F.elu(h_intra_diff)
        h_cross_same = F.elu(h_cross_same)
        h_cross_diff = F.elu(h_cross_diff)

        fused = self.attention_fusion(h_intra_same, h_intra_diff, h_cross_same, h_cross_diff)
        out = self.fc(fused)

        residual = self.residual_fc(x)
        return F.elu(out + residual)
