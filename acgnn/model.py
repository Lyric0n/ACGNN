import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import build_heterogeneous_graph, extract_main_features
from .graphnet import HeteroGraphAttention
from .losses import MultimodalContrastiveLearning


class DualModalMaskReconstruction(nn.Module):
    def __init__(self, visual_dim, audio_dim, feature_size, d=1):
        super().__init__()
        self.d = d
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.feature_size = feature_size

        self.v2a_encoder = nn.Linear(feature_size, feature_size)
        self.a2v_encoder = nn.Linear(feature_size, feature_size)

        self.visual_decoder = nn.Linear(feature_size, visual_dim)
        self.audio_decoder = nn.Linear(feature_size, audio_dim)

    def create_mask(self, dia_len, max_len, device):
        batch_size = len(dia_len)
        visual_mask = torch.zeros(max_len, batch_size, dtype=torch.bool, device=device)
        audio_mask = torch.zeros(max_len, batch_size, dtype=torch.bool, device=device)

        for b in range(batch_size):
            seq_len = dia_len[b]
            positions = torch.arange(seq_len, device=device)

            visual_mask[positions[::self.d * 2], b] = True
            audio_mask[positions[self.d::self.d * 2], b] = True

            visual_mask[seq_len:, b] = False
            audio_mask[seq_len:, b] = False

        return visual_mask, audio_mask

    def apply_mask(self, features, mask):
        mask_expanded = mask.unsqueeze(-1).expand_as(features)
        masked_features = features.clone()
        masked_features[mask_expanded] = 0
        return masked_features

    def forward(self, visual_feat_encoded, audio_feat_encoded, visual_feat, audio_feat,
                dia_len, return_loss=True):
        max_len, batch_size, _ = visual_feat_encoded.shape
        device = visual_feat_encoded.device

        visual_mask, audio_mask = self.create_mask(dia_len, max_len, device)

        visual_feat_masked = self.apply_mask(visual_feat_encoded, visual_mask)
        audio_feat_masked = self.apply_mask(audio_feat_encoded, audio_mask)

        audio_for_visual_recon = self.a2v_encoder(audio_feat_masked)
        visual_for_audio_recon = self.v2a_encoder(visual_feat_masked)

        visual_mask_expanded = visual_mask.unsqueeze(-1).expand_as(visual_feat_encoded)
        audio_mask_expanded = audio_mask.unsqueeze(-1).expand_as(audio_feat_encoded)

        visual_feat_recon_encoded = torch.where(visual_mask_expanded,
                                                audio_for_visual_recon,
                                                visual_feat_encoded)
        audio_feat_recon_encoded = torch.where(audio_mask_expanded,
                                               visual_for_audio_recon,
                                               audio_feat_encoded)

        visual_feat_recon = self.visual_decoder(visual_feat_recon_encoded)
        audio_feat_recon = self.audio_decoder(audio_feat_recon_encoded)

        if not return_loss:
            return visual_feat_recon_encoded, audio_feat_recon_encoded

        visual_mask_expanded_orig = visual_mask.unsqueeze(-1).expand_as(visual_feat)
        audio_mask_expanded_orig = audio_mask.unsqueeze(-1).expand_as(audio_feat)

        visual_recon_loss = F.mse_loss(
            visual_feat_recon[visual_mask_expanded_orig].view(-1, self.visual_dim),
            visual_feat[visual_mask_expanded_orig].view(-1, self.visual_dim))
        audio_recon_loss = F.mse_loss(
            audio_feat_recon[audio_mask_expanded_orig].view(-1, self.audio_dim),
            audio_feat[audio_mask_expanded_orig].view(-1, self.audio_dim))

        return visual_feat_recon_encoded, audio_feat_recon_encoded, \
            visual_recon_loss + audio_recon_loss


class ACGNN(nn.Module):
    def __init__(self, args, DataDim, n_classes):
        super().__init__()
        self.d_l, self.d_v, self.d_a = DataDim
        self.no_cuda = args.no_cuda
        self.unimodal_hs = args.unimodal_hs
        self.graph_hs = args.graph_hs
        self.dropout = args.dropout
        self.temp = args.temp
        self.win_aux = args.win_aux
        self.num_heads = args.num_heads
        self.batch_size = args.batch_size
        self.alpha = args.alpha
        self.d_masked = args.d_masked

        self.contrastive_loss = MultimodalContrastiveLearning(
            self.unimodal_hs, n_classes, self.temp)
        self.AVE = DualModalMaskReconstruction(
            self.d_v, self.d_a, self.unimodal_hs, d=self.d_masked)

        self.linear_a = nn.Linear(self.d_a, self.unimodal_hs)
        self.linear_v = nn.Linear(self.d_v, self.unimodal_hs)

        self.linear_av = nn.Sequential(
            nn.Linear(self.unimodal_hs * 2, self.unimodal_hs),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.lstm_l = nn.LSTM(input_size=self.d_l, hidden_size=self.unimodal_hs // 2,
                              num_layers=2, bidirectional=True, dropout=self.dropout)

        self.graphnet_l_av = HeteroGraphAttention(
            in_features=self.unimodal_hs, hidden_size=self.graph_hs,
            num_heads=self.num_heads, dropout=self.dropout)

        self.log_var_task = nn.Parameter(torch.zeros(1))
        self.log_var_cl = nn.Parameter(torch.zeros(1))

        self.smax_fc = nn.Linear(self.graph_hs, n_classes)

    def uncertainty_weighted_loss(self, task_loss, contrastive_loss):
        """Combine the task and contrastive objectives with learned uncertainties."""
        precision_task = torch.exp(-self.log_var_task)
        precision_cl = torch.exp(-self.log_var_cl)
        return (task_loss * precision_task + 0.5 * self.log_var_task
                + contrastive_loss * precision_cl + 0.5 * self.log_var_cl)

    def forward(self, U_l, qmask, seq_lengths, emo_labels, U_a=None, U_v=None):
        audio_proj = self.linear_a(U_a)
        visual_proj = self.linear_v(U_v)

        contrastive_loss = self.contrastive_loss(visual_proj, audio_proj,
                                                 seq_lengths, emo_labels)
        visual_feat_recon, audio_feat_recon = self.AVE(
            visual_proj, audio_proj, U_v, U_a, seq_lengths, return_loss=False)

        u_l, _ = self.lstm_l(U_l)
        u_l = torch.relu(u_l)
        u_av = self.linear_av(torch.cat([audio_feat_recon, visual_feat_recon], dim=-1))

        node_features, edge_indices = build_heterogeneous_graph(
            u_l, u_av, seq_lengths, qmask, win_aux=self.win_aux, no_cuda=self.no_cuda)

        features = self.graphnet_l_av(node_features, edge_indices)
        emotions = extract_main_features(features, seq_lengths)

        all_final_out = self.smax_fc(emotions)
        all_log_prob = F.log_softmax(all_final_out, 1)

        return all_log_prob, contrastive_loss, emotions
