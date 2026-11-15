"""Tri-modal cross-attention fusion model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .physical_features import compute_physical_features


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError(f"PyTorch is required for fusion model: {type(exc).__name__}: {exc}") from exc
    return torch, nn, DataLoader, TensorDataset


def resolve_device(device: str = "auto") -> str:
    torch, _, _, _ = _require_torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def make_fusion_model(phy_dim: int, text_dim: int, graph_dim: int, embed_dim: int = 128):
    torch, nn, _, _ = _require_torch()

    class TriModalAttentionFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.phy_proj = nn.Linear(phy_dim, embed_dim)
            self.text_proj = nn.Linear(text_dim, embed_dim)
            self.graph_proj = nn.Linear(graph_dim, embed_dim)
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.attn_norm = nn.LayerNorm(embed_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )
            self.output_norm = nn.LayerNorm(embed_dim)
            self.fc = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, phy_x, text_x, graph_x):
            q = self.phy_proj(phy_x).unsqueeze(1)
            text = self.text_proj(text_x).unsqueeze(1)
            graph = self.graph_proj(graph_x).unsqueeze(1)
            kv = torch.cat([text, graph], dim=1)
            fused, _ = self.cross_attn(q, kv, kv)
            fused = self.attn_norm(q + fused)
            fused = self.output_norm(fused + self.feed_forward(fused)).squeeze(1)
            pred = self.fc(fused)
            return pred, fused

    return TriModalAttentionFusion()


def _load_modal_arrays(df: pd.DataFrame, bert_csv: Path, graph_csv: Path):
    phy = compute_physical_features(df)
    bert = pd.read_csv(bert_csv).fillna(0.0)
    graph = pd.read_csv(graph_csv).fillna(0.0)
    if len(phy) != len(bert) or len(phy) != len(graph):
        raise ValueError(
            f"Modal feature row mismatch: physical={len(phy)}, bert={len(bert)}, graph={len(graph)}"
        )
    return phy, bert, graph


def train_fusion_model(
    train_df: pd.DataFrame,
    bert_train_csv: Path,
    graph_train_csv: Path,
    target_col: str,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 0.005,
    random_state: int = 42,
    device: str = "auto",
):
    torch, nn, DataLoader, TensorDataset = _require_torch()
    device = resolve_device(device)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)

    phy, bert, graph = _load_modal_arrays(train_df, bert_train_csv, graph_train_csv)
    y = pd.to_numeric(train_df[target_col], errors="coerce")
    valid = y.notna() & (y > 0)
    if int(valid.sum()) < 2:
        raise ValueError(f"Need at least 2 valid rows for target {target_col}.")

    model = make_fusion_model(phy.shape[1], bert.shape[1], graph.shape[1]).to(device)
    dataset = TensorDataset(
        torch.tensor(phy.loc[valid].to_numpy(dtype=np.float32), dtype=torch.float32),
        torch.tensor(bert.loc[valid].to_numpy(dtype=np.float32), dtype=torch.float32),
        torch.tensor(graph.loc[valid].to_numpy(dtype=np.float32), dtype=torch.float32),
        torch.tensor(y.loc[valid].to_numpy(dtype=np.float32), dtype=torch.float32).view(-1, 1),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = []

    model.train()
    for epoch in range(epochs):
        losses = []
        maes = []
        for b_phy, b_text, b_graph, b_y in loader:
            b_phy, b_text, b_graph, b_y = (
                b_phy.to(device),
                b_text.to(device),
                b_graph.to(device),
                b_y.to(device),
            )
            optimizer.zero_grad()
            pred, _ = model(b_phy, b_text, b_graph)
            loss = criterion(pred, b_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            maes.append(float(torch.abs(pred - b_y).mean().detach().cpu()))
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            mse = float(np.mean(losses))
            history.append(
                {
                    "epoch": epoch + 1,
                    "mse": mse,
                    "rmse": float(np.sqrt(mse)),
                    "mae": float(np.mean(maes)),
                }
            )
    return model, {
        "target_col": target_col, "valid_train_count": int(valid.sum()),
        "phy_dim": phy.shape[1], "text_dim": bert.shape[1], "graph_dim": graph.shape[1],
        "physical_features": phy.columns.tolist(),
        "history": history,
    }


def save_fusion_model(model, path: Path, metadata: dict) -> Path:
    torch, _, _, _ = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    return path


def load_fusion_model(path: Path, phy_dim: int, text_dim: int, graph_dim: int, device: str = "auto"):
    torch, _, _, _ = _require_torch()
    device = resolve_device(device)
    payload = torch.load(path, map_location=device, weights_only=True)
    model = make_fusion_model(phy_dim, text_dim, graph_dim).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("metadata", {})


def extract_fusion_features(
    model,
    df: pd.DataFrame,
    bert_csv: Path,
    graph_csv: Path,
    output_csv: Path,
    device: str = "auto",
) -> Path:
    torch, _, _, _ = _require_torch()
    device = resolve_device(device)
    phy, bert, graph = _load_modal_arrays(df, bert_csv, graph_csv)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        _, fused = model(
            torch.tensor(phy.to_numpy(dtype=np.float32), dtype=torch.float32).to(device),
            torch.tensor(bert.to_numpy(dtype=np.float32), dtype=torch.float32).to(device),
            torch.tensor(graph.to_numpy(dtype=np.float32), dtype=torch.float32).to(device),
        )
    arr = fused.detach().cpu().numpy()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(arr, columns=[f"fusion_emb_{i}" for i in range(arr.shape[1])]).to_csv(output_csv, index=False)
    return output_csv

