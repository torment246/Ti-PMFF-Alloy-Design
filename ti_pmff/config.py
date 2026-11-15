"""Default configuration for Ti-PMFF wrappers."""

from __future__ import annotations

ELEMENT_COLS = [
    "Ti (wt%)",
    "Mo (wt%)",
    "Al (wt%)",
    "Sn (wt%)",
    "V (wt%)",
    "Zr (wt%)",
    "Cr (wt%)",
    "Nb (wt%)",
    "Ta (wt%)",
    "Fe (wt%)",
    "W (wt%)",
    "Si (wt%)",
    "O (wt%)",
    "C (wt%)",
    "N (wt%)",
    "H (wt%)",
    "Ni (wt%)",
    "Cu (wt%)",
    "B (wt%)",
    "Mn (wt%)",
    "Y (wt%)",
    "Zn (wt%)",
    "TiB (wt%)",
]

UTS_COL = "Strength (MPa)"
EL_COL = "Elongation (%)"
PROCESS_COL = "Process"
TRANSITION_TEMP_COL = "transition temperature (°C)"

DEFAULT_BERT_CONFIG = "configs/bert_demo.json"
DEFAULT_VOCAB = "data/demo/vocab.txt"
DEFAULT_CHECKPOINT_PREFIX = "outputs/01_ti_bert/model.ckpt-8"
DEFAULT_COMPOSITION_DATA = "data/demo/composition.csv"
DEFAULT_COMPOSITION_EMBEDDINGS = "outputs/02_prediction/composition/train_bert.csv"
DEFAULT_PROCESSING_DATA = "data/demo/processing.csv"
DEFAULT_TARGETS = ["UTS", "EL"]
DEFAULT_DOWNSTREAM = "ridge"
DEFAULT_RANDOM_STATE = 42

TARGET_TO_COLUMN = {
    "UTS": UTS_COL,
    "EL": EL_COL,
}

TARGET_TO_CACHE_NAME = {
    "UTS": "uts",
    "EL": "el",
}
