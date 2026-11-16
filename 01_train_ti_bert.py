"""Train Ti-BERT with masked-language modelling and next-sentence prediction."""
import argparse
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS_FILE = ROOT / "data/demo/corpus.txt"
VOCAB_FILE = ROOT / "data/demo/vocab.txt"
BERT_CONFIG = ROOT / "configs/bert_demo.json"
TRAINING_CONFIG = ROOT / "configs/bert_training_demo.json"
OUTPUT_DIR = ROOT / "outputs/01_ti_bert"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_FILE)
    parser.add_argument("--vocab", type=Path, default=VOCAB_FILE)
    parser.add_argument("--bert-config", type=Path, default=BERT_CONFIG)
    parser.add_argument("--training-config", type=Path, default=TRAINING_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--steps", type=int, help="Override the number of training steps")
    parser.add_argument("--dupe-factor", type=int, default=2)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.device == "cuda" else "-1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = json.loads(args.training_config.read_text())

    from ti_pmff.bert_features import patch_tensorflow_v1_compat
    patch_tensorflow_v1_compat()
    from pretraining import tokenization
    from pretraining.create_pretraining_data import create_training_instances, write_instance_to_example_files
    from pretraining.train import train

    # Documents are separated by blank lines; each nonempty line is one sentence.
    tokenizer = tokenization.FullTokenizer(str(args.vocab), do_lower_case=True)
    instances = create_training_instances(
        [str(args.corpus)], tokenizer, settings["max_seq_length"],
        dupe_factor=args.dupe_factor, short_seq_prob=0.1, masked_lm_prob=0.15,
        max_predictions_per_seq=settings["max_predictions_per_seq"],
        rng=random.Random(12345),
    )
    record_file = args.output_dir / "pretraining_examples.tfrecords"
    write_instance_to_example_files(
        instances, tokenizer, settings["max_seq_length"],
        settings["max_predictions_per_seq"], [str(record_file)],
    )

    settings.update({
        "bert_config_file": str(args.bert_config),
        "input_file": str(record_file),
        "output_dir": str(args.output_dir),
        "device": args.device,
        "seed": 42,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "resume": args.resume,
    })
    if args.steps is not None:
        settings["num_train_steps"] = args.steps
    print(f"Corpus: {args.corpus}")
    print(f"Training examples: {len(instances)}")
    print(f"BERT config: {args.bert_config}")
    print(f"Training device: {args.device}")
    train(argparse.Namespace(**settings))
    print(f"Checkpoint: {args.output_dir / ('model.ckpt-' + str(settings['num_train_steps']))}")


if __name__ == "__main__":
    main()
