import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, alpha=None, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, log_probs, labels):
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if log_probs.dim() > 3:
            log_probs = log_probs.contiguous().view(log_probs.size(0), log_probs.size(1),
                                                    log_probs.size(2), -1)
            log_probs = log_probs.transpose(2, 3)
            log_probs = log_probs.contiguous().view(-1, log_probs.size(1), log_probs.size(3)).squeeze()

        labels_length = log_probs.size(1)
        seq_length = log_probs.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length],
                                   device=log_probs.device).scatter_(1, new_label, 1)

        pt = torch.exp(log_probs) * label_onehot
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                alpha_t = torch.tensor(self.alpha, device=log_probs.device, dtype=log_probs.dtype)
                if alpha_t.dim() == 1:
                    alpha_t = alpha_t.unsqueeze(0).expand(seq_length, -1)
                focal_weight = focal_weight * alpha_t

        fl = -focal_weight * log_probs * label_onehot
        fl = fl.sum(dim=1)
        return fl.mean() if self.size_average else fl.sum()


class MultimodalContrastiveLearning(nn.Module):
    def __init__(self, feature_size, num_classes, temperature=0.1):
        super().__init__()
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.temperature = temperature

        self.projection = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.LayerNorm(feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size),
            nn.LayerNorm(feature_size),
        )

    def mask_padding(self, features, dia_len):
        max_len, batch_size, _ = features.shape
        mask = torch.zeros((max_len, batch_size), dtype=torch.bool, device=features.device)
        for i, length in enumerate(dia_len):
            mask[:length, i] = True
        return mask

    def emotion_contrastive_loss(self, visual_feat, audio_feat, emo_labels, mask):
        max_len, batch_size, feat_size = visual_feat.shape

        all_features = torch.cat([visual_feat, audio_feat], dim=0)
        mask_expanded = mask.repeat(2, 1)
        all_features = all_features.reshape(-1, feat_size)[mask_expanded.reshape(-1)]

        projected_features = F.normalize(self.projection(all_features), dim=1)

        emo_labels_masked = emo_labels.reshape(-1)[mask.reshape(-1)]
        all_labels = torch.cat([emo_labels_masked, emo_labels_masked], dim=0)

        similarity_matrix = torch.matmul(projected_features, projected_features.T) / self.temperature

        labels_expanded = all_labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        positive_mask = positive_mask - torch.eye(positive_mask.size(0), device=positive_mask.device)

        exp_sim = torch.exp(similarity_matrix)
        positive_sim = torch.sum(exp_sim * positive_mask, dim=1)
        negative_sim = torch.sum(exp_sim, dim=1) - torch.exp(torch.diag(similarity_matrix))

        loss = -torch.log(positive_sim / (negative_sim + 1e-8))
        return loss.mean()

    def modality_contrastive_loss(self, visual_feat, audio_feat, mask):
        visual_proj = F.normalize(self.projection(visual_feat[mask]), dim=1)
        audio_proj = F.normalize(self.projection(audio_feat[mask]), dim=1)

        similarity = torch.matmul(visual_proj, audio_proj.T) / self.temperature
        labels = torch.arange(visual_proj.size(0), device=visual_proj.device)

        loss_visual = F.cross_entropy(similarity, labels)
        loss_audio = F.cross_entropy(similarity.T, labels)
        return (loss_visual + loss_audio) / 2

    def forward(self, visual_feat, audio_feat, dia_len, emo_labels):
        mask = self.mask_padding(visual_feat, dia_len)
        emotion_loss = self.emotion_contrastive_loss(visual_feat, audio_feat, emo_labels, mask)
        modality_loss = self.modality_contrastive_loss(visual_feat, audio_feat, mask)
        return emotion_loss + modality_loss
