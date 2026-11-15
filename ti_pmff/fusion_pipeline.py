"""Train fusion models and transform held-out samples and design candidates."""
import json
from pathlib import Path

import pandas as pd
import torch

from .bert_features import extract_bert_cls_features
from .config import TARGET_TO_COLUMN
from .fusion_model import (
    extract_fusion_features, load_fusion_model, save_fusion_model, train_fusion_model,
)
from .graph_features import extract_graph_features


class TiPMFFFeaturePipeline:
    def __init__(self, output_dir, checkpoint, bert_config, vocab_file,
                 device="cpu", bert_device="cuda", fusion_epochs=20, seed=42):
        self.output_dir = Path(output_dir)
        self.settings = {
            "checkpoint": str(Path(checkpoint).resolve()),
            "bert_config": str(Path(bert_config).resolve()),
            "vocab_file": str(Path(vocab_file).resolve()),
            "seed": seed,
        }
        self.device, self.bert_device = device, bert_device
        self.fusion_epochs = fusion_epochs

    def modal_features(self, datasets, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        text = {}
        for name, data in datasets.items():
            text[name] = extract_bert_cls_features(
                data, output_dir / f"{name}_bert.csv", self.settings["checkpoint"],
                self.settings["bert_config"], self.settings["vocab_file"],
                layer=11, device=self.bert_device,
            )
        graph = extract_graph_features(datasets, output_dir, self.settings["seed"])
        return text, graph

    def fit_transform(self, train, test, candidates=None):
        datasets = {"train": train, "test": test}
        if candidates is not None:
            datasets["candidate"] = candidates
        text, graph = self.modal_features(datasets, self.output_dir)
        for target, column in TARGET_TO_COLUMN.items():
            model, metadata = train_fusion_model(
                train, text["train"], graph["train"], column,
                epochs=self.fusion_epochs, device=self.device,
                random_state=self.settings["seed"],
            )
            name = target.lower()
            save_fusion_model(model, self.output_dir / f"fusion_{name}.pt", metadata)
            for split, data in datasets.items():
                extract_fusion_features(
                    model, data, text[split], graph[split],
                    self.output_dir / f"{name}_{split}_fusion.csv", self.device,
                )
        (self.output_dir / "feature_settings.json").write_text(
            json.dumps(self.settings, indent=2) + "\n")

    @classmethod
    def from_directory(cls, feature_dir, device="cpu", bert_device="cuda"):
        settings = json.loads((Path(feature_dir) / "feature_settings.json").read_text())
        return cls(feature_dir, device=device, bert_device=bert_device, **settings)

    def transform_candidates(self, candidates, output_dir):
        output_dir = Path(output_dir)
        text, graph = self.modal_features({"candidate": candidates}, output_dir)
        features = {}
        for target in TARGET_TO_COLUMN:
            name = target.lower()
            model_file = self.output_dir / f"fusion_{name}.pt"
            metadata = torch.load(model_file, map_location="cpu", weights_only=True)["metadata"]
            model, _ = load_fusion_model(
                model_file, metadata["phy_dim"], metadata["text_dim"], metadata["graph_dim"],
                self.device,
            )
            path = output_dir / f"{name}_candidate_fusion.csv"
            extract_fusion_features(model, candidates, text["candidate"], graph["candidate"],
                                    path, self.device)
            features[target] = pd.read_csv(path)
        return features
