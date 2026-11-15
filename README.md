# Ti-PMFF

Code for **Data-driven upcycling of low-grade Ti–6Al–4V scrap into an ultrastrong
and ductile titanium alloy**.

## Overview

Ti-PMFF combines Ti-BERT text embeddings, element-interaction graph features and
five physical descriptors: MoEq, AlEq, valence electron concentration, atomic-size
mismatch and beta-transus temperature. Cross-attention fuses these features, and
GA selects descriptor subsets for UTS and elongation prediction.

The workflow includes:

1. Ti-BERT pretraining
2. Ti-PMFF property prediction
3. GA feature selection
4. Composition design
5. Processing optimization for the selected composition
6. Two-stage design figures and tables

The supplied data are synthetic examples for running this workflow. The complete
training datasets and pretrained model weights are available from the corresponding
authors upon reasonable request.

## System requirements and installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install "tensorflow[and-cuda]==2.21.0"
```

## Input data

The default input files are in `data/demo/`:

| File | Contents |
| --- | --- |
| `composition.csv` | Alloy compositions and UTS/EL labels |
| `processing.csv` | Alloy compositions, processing descriptions and UTS/EL labels |
| `candidates.csv` | Candidate compositions for Stage I |
| `corpus.txt` | Sentence-separated text for Ti-BERT pretraining |
| `vocab.txt` | WordPiece vocabulary |

Demo parameters are in `configs/`:

- `bert_demo.json`: BERT architecture
- `bert_training_demo.json`: pretraining settings
- `ga_demo.json`: GA and downstream regressors
- `physical_ranges_demo.json`: composition-screening intervals
- `design_demo.json`: processing conditions and search grid

## 1. Ti-BERT pretraining

Input: corpus, vocabulary and BERT model/training configurations.

```bash
python 01_train_ti_bert.py \
  --corpus data/demo/corpus.txt \
  --vocab data/demo/vocab.txt \
  --bert-config configs/bert_demo.json \
  --training-config configs/bert_training_demo.json \
  --output-dir outputs/01_ti_bert
```

Typical outputs:

```text
outputs/01_ti_bert/
  pretraining_examples.tfrecords
  model.ckpt-8.*
```

## 2. Ti-PMFF property prediction

Input: the two property CSV files, composition candidates and Ti-BERT checkpoint.

```bash
python 02_predict_properties.py \
  --composition-data data/demo/composition.csv \
  --processing-data data/demo/processing.csv \
  --candidates data/demo/candidates.csv \
  --checkpoint outputs/01_ti_bert/model.ckpt-8 \
  --output-dir outputs/02_prediction
```

Typical outputs:

```text
outputs/02_prediction/
  composition/       Split records, text/graph/fused features and fusion weights
  processing/        Split records, text/graph/fused features and fusion weights
  prediction_metrics.csv
  prediction_predictions.csv
```

## 3. GA feature selection

Input: the split records and fused features from Step 2.

```bash
python 03_select_features.py \
  --feature-dir outputs/02_prediction \
  --config configs/ga_demo.json \
  --output-dir outputs/03_feature_selection
```

Typical outputs:

```text
outputs/03_feature_selection/
  composition_UTS/
  composition_EL/
  process_UTS/
  process_EL/
  metrics.csv
```

## 4. Composition design: Stage I

Input: Step-2 composition candidates/features and Step-3 GA models.

```bash
python 04_design_composition.py \
  --feature-dir outputs/02_prediction/composition \
  --selection-dir outputs/03_feature_selection \
  --physical-ranges configs/physical_ranges_demo.json \
  --output-dir outputs/04_composition
```

Typical outputs:

```text
outputs/04_composition/
  predictions.csv
  composition_ranked.csv
  composition_topk.csv
  physical_ranges.json
```

## 5. Processing optimization: Stage II

Input: Step-4 composition results, processing-grid settings, Step-2 fusion models
and Step-3 processing GA models.

```bash
python 05_optimize_processing.py \
  --composition-results outputs/04_composition/composition_topk.csv \
  --design-config configs/design_demo.json \
  --feature-dir outputs/02_prediction/processing \
  --selection-dir outputs/03_feature_selection \
  --output-dir outputs/05_processing
```

Typical outputs:

```text
outputs/05_processing/
  design_settings.json
  process_candidates.csv
  features/
  predictions.csv
  process_ranked.csv
  process_topk.csv
```

## 6. Two-stage design figures and tables

Input: the rankings and predictions from Steps 4 and 5.

```bash
python 06_plot_results.py \
  --results-dir outputs \
  --output-dir outputs/06_results
```

Outputs:

```text
outputs/06_results/
  figures/
    Fig_S6.png
    Fig_S7.png
  tables/
    Table_S5.csv
    Table_S6.csv
```

## Suggested running order

Run scripts 01–06 in order, or use the complete workflow:

```bash
python run_demo.py --figures
```

## Code organization

- `pretraining/`: BERT model, tokenizer, example preparation and training
- `ti_pmff/`: physical, text, graph, fusion, prediction and plotting components
- `ga/`: feature selection
- `stage1/`: composition scoring and ranking
- `stage2/`: processing candidates and scoring
- `data/generate_demo.py`: regenerate the supplied synthetic examples

## Demo data and availability

The included data are synthetic examples for demonstrating the workflow.
Prediction metrics, selected features and recommended candidates can differ from
the full research results; the Demo does not reproduce all manuscript metrics.

The complete training datasets and pretrained model weights are available from
the corresponding authors upon reasonable request.

## License and third-party code

Copyright © 2026 the authors. The author-owned code is provided for peer-review
evaluation and reproducibility checking. Redistribution, modification and commercial
use require prior written permission; see [LICENSE](LICENSE).

The Ti-BERT implementation includes Google AI Language Team code under the
[Apache 2.0 license](licenses/Apache-2.0.txt), with original copyright and license
headers retained.
