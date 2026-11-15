"""Element interaction graph feature extraction."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import prepare_base_dataframe

ELEMENT_PROPS = {
    "Ti": [147.0, 1.54, 4.0],
    "Mo": [139.0, 2.16, 6.0],
    "Al": [143.0, 1.61, 3.0],
    "Sn": [140.0, 1.96, 4.0],
    "V": [134.0, 1.63, 5.0],
    "Zr": [160.0, 1.33, 4.0],
    "Cr": [128.0, 1.66, 6.0],
    "Nb": [146.0, 1.60, 5.0],
    "Ta": [146.0, 1.50, 5.0],
    "Fe": [126.0, 1.83, 8.0],
    "W": [139.0, 2.36, 6.0],
    "Si": [111.0, 1.90, 4.0],
    "O": [60.0, 3.44, 6.0],
    "C": [70.0, 2.55, 4.0],
    "N": [65.0, 3.04, 5.0],
    "H": [53.0, 2.20, 1.0],
    "Ni": [124.0, 1.91, 10.0],
    "Cu": [128.0, 1.90, 11.0],
    "B": [85.0, 2.04, 3.0],
    "Mn": [127.0, 1.55, 7.0],
    "Y": [180.0, 1.22, 3.0],
    "Zn": [134.0, 1.65, 12.0],
    "Hf": [159.0, 1.30, 4.0],
}


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GATConv, global_max_pool
    except Exception as exc:
        raise RuntimeError(f"PyTorch/torch_geometric is required for GNN features: {type(exc).__name__}: {exc}") from exc
    return torch, nn, Data, DataLoader, GATConv, global_max_pool


def _make_model(node_dim: int = 4, hidden_dim: int = 32, output_dim: int = 64):
    torch, nn, _, _, GATConv, global_max_pool = _require_torch()

    class ElementInteractionGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.gat1 = GATConv(node_dim, hidden_dim, heads=4, concat=True)
            self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=False)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = self.gat1(x, edge_index)
            x = torch.relu(x)
            x = self.gat2(x, edge_index)
            x = global_max_pool(x, batch)
            return self.fc(x)

    return ElementInteractionGNN()


def _dataset_from_df(df: pd.DataFrame):
    torch, _, Data, _, _, _ = _require_torch()
    prepared = prepare_base_dataframe(df)
    dataset = []
    for _, row in prepared.iterrows():
        nodes = []
        for el, props in ELEMENT_PROPS.items():
            col = f"{el} (wt%)"
            if col in prepared.columns:
                wt = row[col]
                if pd.notna(wt) and float(wt) > 0:
                    nodes.append(props + [float(wt)])
        if not nodes:
            nodes.append(ELEMENT_PROPS["Ti"] + [100.0])
        x = torch.tensor(nodes, dtype=torch.float32)
        n_nodes = len(nodes)
        edge_index = [[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j] or [[0, 0]]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        dataset.append(Data(x=x, edge_index=edge_index))
    return dataset


def _extract_with_model(df: pd.DataFrame, output_csv: Path, model, batch_size: int = 32) -> Path:
    torch, _, _, DataLoader, _, _ = _require_torch()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(_dataset_from_df(df), batch_size=batch_size, shuffle=False)
    features = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features.append(model(batch).detach().cpu().numpy())
    arr = np.vstack(features)
    pd.DataFrame(arr, columns=[f"feat_{i}" for i in range(arr.shape[1])]).to_csv(output_csv, index=False)
    return output_csv


def fixed_graph_model(seed: int = 42):
    """Create a frozen GAT encoder."""
    torch, _, _, _, _, _ = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = _make_model()
    model.requires_grad_(False)
    model.eval()
    return model


def extract_graph_features(datasets: dict, output_dir: Path, seed: int = 42) -> dict:
    torch, _, _, _, _, _ = _require_torch()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = fixed_graph_model(seed)
        return {name: _extract_with_model(data, output_dir / f"{name}_graph.csv", model)
                for name, data in datasets.items()}
    finally:
        torch.set_num_threads(previous_threads)
