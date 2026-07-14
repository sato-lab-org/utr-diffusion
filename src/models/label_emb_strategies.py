import torch
import torch.nn as nn

def build_label_encoder(label_emb_mode, num_label, emb_dim):
    if label_emb_mode == "concat":
        return ConcatLabelEmbedding(num_label, emb_dim)
    elif label_emb_mode == "per_label":
        return PerLabelAdaptorEmbedding(num_label, emb_dim)
    elif label_emb_mode == "per_label_shared_null":
        return PerLabelAdaptorEmbeddingSharedNull(num_label, emb_dim)
    else:
        raise ValueError(f"Unknown label_emb_mode: {label_emb_mode}")

class ConcatLabelEmbedding(nn.Module):
    def __init__(self, num_labels, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_labels * 2, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, label_values, label_is_presence):
        x = torch.cat([label_values, label_is_presence], dim=-1)   # [B, 2N] label: [3.0, NaN, 1.5] -> [3.0, 0.0, 1.5], [1, 0, 1]
        return self.net(x) # cat[label.unsqueeze(1), mask.unsequeeze(1)]


class PerLabelAdaptorEmbedding(nn.Module):
    def __init__(self, num_labels, emb_dim):
        super().__init__()
        self.num_labels = num_labels
        self.null_emb = nn.Embedding(num_labels, emb_dim)
        self.adaptors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, emb_dim),
                nn.SiLU(),
                nn.Linear(emb_dim, emb_dim),
            )
            for _ in range(num_labels)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, label_values, label_is_presence):
        B = label_values.shape[0] # Batch and num_label

        m = label_is_presence.bool().unsqueeze(-1) # [B, N, 1]

        label_idx = torch.arange(self.num_labels, device=label_values.device, dtype=torch.long).unsqueeze(0).expand(B, -1) # [N] -> [1, N] -> [B, N]

        uncond = self.null_emb(label_idx) # [B, N, D]
        cond_list = []
        for i in range(self.num_labels):
            v_i = label_values[:, i:i+1] # [B, 1]
            cond_i = self.adaptors[i](v_i) #[B, D]
            cond_list.append(cond_i)
        cond = torch.stack(cond_list, dim=1) # [B, N ,D]
        out = torch.where(m, cond, uncond) # if  m == 1 get cond, if m == 0 get uncond
        out = out.mean(dim=1) # [B, D]
        return self.out_proj(out)


class PerLabelAdaptorEmbeddingSharedNull(nn.Module):
    def __init__(self, num_labels, emb_dim):
        super().__init__()
        self.num_labels = num_labels
        self.null_emb = nn.Embedding(1, emb_dim)
        self.adaptors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, emb_dim),
                nn.SiLU(),
                nn.Linear(emb_dim, emb_dim),
            )
            for _ in range(num_labels)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, label_values, label_is_presence):
        m = label_is_presence.bool() # [B, N]

        cond_list = []
        for i in range(self.num_labels):
            v_i = label_values[:, i:i+1] # [B, 1]
            cond_i = self.adaptors[i](v_i) #[B, D]
            cond_list.append(cond_i)
        cond = torch.stack(cond_list, dim=1) # [B, N ,D]
        mask_cond = cond * m.unsqueeze(-1)
        mask_cond_sum = mask_cond.sum(dim=1)
        num_valid = m.sum(dim=1, keepdim=True) # [B, 1]
        out = mask_cond_sum / num_valid.clamp(min=1.0) #

        has_valid = (num_valid.squeeze(1) > 0)
        if has_valid.any():
            out[has_valid] = self.out_proj(out[has_valid]).to(out.dtype)

        all_missing = ~has_valid
        if all_missing.any():
            out[all_missing] = self.null_emb.weight[0]

        return out