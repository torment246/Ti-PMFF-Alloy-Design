"""Train BERT with masked-language modelling and next-sentence prediction."""

import json
from pathlib import Path

from .compat import configure_modeling


def input_dataset(tf, files, args, training):
    schema = {
        'input_ids': tf.io.FixedLenFeature([args.max_seq_length], tf.int64),
        'input_mask': tf.io.FixedLenFeature([args.max_seq_length], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([args.max_seq_length], tf.int64),
        'masked_lm_positions': tf.io.FixedLenFeature([args.max_predictions_per_seq], tf.int64),
        'masked_lm_ids': tf.io.FixedLenFeature([args.max_predictions_per_seq], tf.int64),
        'masked_lm_weights': tf.io.FixedLenFeature([args.max_predictions_per_seq], tf.float32),
        'next_sentence_labels': tf.io.FixedLenFeature([1], tf.int64),
    }

    def decode(record):
        example = tf.io.parse_single_example(record, schema)
        return {name: tf.cast(value, tf.int32) if value.dtype == tf.int64 else value
                for name, value in example.items()}

    if training:
        dataset = tf.data.Dataset.from_tensor_slices(files).repeat()
        dataset = dataset.shuffle(len(files), seed=args.seed)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=min(args.num_cpu_threads, len(files)),
            num_parallel_calls=min(args.num_cpu_threads, len(files)),
            deterministic=args.seed is not None,
        ).shuffle(100, seed=args.seed)
    else:
        dataset = tf.data.TFRecordDataset(files).repeat()
    dataset = dataset.map(decode, num_parallel_calls=args.num_cpu_threads)
    batch_size = args.train_batch_size if training else args.eval_batch_size
    options = tf.data.Options()
    options.threading.private_threadpool_size = args.num_cpu_threads
    return dataset.batch(batch_size, drop_remainder=True).with_options(options).prefetch(1)


def build_model(config, features, training):
    tf, modeling = configure_modeling()
    from .objectives import get_masked_lm_output, get_next_sentence_output

    model = modeling.BertModel(
        config=config, is_training=training,
        input_ids=features['input_ids'], input_mask=features['input_mask'],
        token_type_ids=features['segment_ids'], use_one_hot_embeddings=False, scope='bert',
    )
    mlm_loss, mlm_example_loss, mlm_log_probs = get_masked_lm_output(
        config, model.get_sequence_output(), model.get_embedding_table(),
        features['masked_lm_positions'], features['masked_lm_ids'], features['masked_lm_weights'],
    )
    nsp_loss, _, nsp_log_probs = get_next_sentence_output(
        config, model.get_pooled_output(), features['next_sentence_labels'],
    )
    return {
        'loss': mlm_loss + nsp_loss, 'masked_lm_loss': mlm_loss,
        'next_sentence_loss': nsp_loss, 'mlm_example_loss': mlm_example_loss,
        'mlm_log_probs': mlm_log_probs, 'nsp_log_probs': nsp_log_probs,
    }


def evaluate(tf, session, iterator, features, model, steps):
    import numpy as np

    session.run(iterator.initializer)
    losses, masked_loss_sum, masked_correct, masked_count, nsp_correct, nsp_count = [], 0., 0., 0., 0, 0
    for _ in range(steps):
        values, labels = session.run([model, features])
        weights = labels['masked_lm_weights'].reshape(-1)
        predictions = values['mlm_log_probs'].argmax(axis=-1)
        masked_correct += float(np.sum((predictions == labels['masked_lm_ids'].reshape(-1)) * weights))
        masked_loss_sum += float(np.sum(values['mlm_example_loss'] * weights))
        masked_count += float(weights.sum())
        nsp_correct += int(np.sum(values['nsp_log_probs'].argmax(axis=-1) == labels['next_sentence_labels'].reshape(-1)))
        nsp_count += labels['next_sentence_labels'].size
        losses.append([values['loss'], values['next_sentence_loss']])
    means = np.mean(losses, axis=0)
    return {
        'loss': float(means[0]), 'masked_lm_loss': masked_loss_sum / max(masked_count, 1.),
        'masked_lm_accuracy': masked_correct / max(masked_count, 1.),
        'next_sentence_loss': float(means[1]), 'next_sentence_accuracy': nsp_correct / nsp_count,
    }


def train(args):
    tf, modeling = configure_modeling()
    from .optimization import create_optimizer

    config = modeling.BertConfig.from_json_file(args.bert_config_file)
    if args.max_seq_length > config.max_position_embeddings:
        raise ValueError('max-seq-length exceeds the model positional embeddings')
    files = [path for pattern in args.input_file.split(',') for path in sorted(tf.io.gfile.glob(pattern))]
    if not files:
        raise FileNotFoundError(f'No TFRecord inputs matched {args.input_file}')
    output = Path(args.output_dir)
    previous = tf.train.latest_checkpoint(str(output))
    if args.resume and not previous:
        raise FileNotFoundError('--resume requires a checkpoint in output-dir')
    if previous and not args.resume:
        raise FileExistsError('Output already contains a checkpoint; use --resume or a new output-dir')
    if args.device == 'cuda' and not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('--device cuda requested but TensorFlow cannot see a GPU')
    if args.device == 'auto':
        args.device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
    output.mkdir(parents=True, exist_ok=True)
    graph = tf.Graph()
    with graph.as_default(), tf.device('/GPU:0' if args.device == 'cuda' else '/CPU:0'):
        if args.seed is not None:
            tf.compat.v1.set_random_seed(args.seed)
        iterator = tf.compat.v1.data.make_initializable_iterator(input_dataset(tf, files, args, True))
        train_model = build_model(config, iterator.get_next(), True)
        if args.init_checkpoint:
            variables = tf.compat.v1.trainable_variables()
            tf.compat.v1.train.init_from_checkpoint(args.init_checkpoint, {v.op.name: v for v in variables})
        train_op = create_optimizer(train_model['loss'], args.learning_rate, args.num_train_steps,
                                    args.num_warmup_steps, use_tpu=False)
        with tf.control_dependencies([train_op]):
            step_after_train = tf.identity(tf.compat.v1.train.get_or_create_global_step())
        if args.max_eval_steps:
            eval_iterator = tf.compat.v1.data.make_initializable_iterator(input_dataset(tf, files, args, False))
            eval_features = eval_iterator.get_next()
            with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=True):
                eval_model = build_model(config, eval_features, False)
        saver = tf.compat.v1.train.Saver(max_to_keep=1)
        initialize = tf.compat.v1.global_variables_initializer()
        global_step = tf.compat.v1.train.get_or_create_global_step()
    session_config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=args.num_cpu_threads, inter_op_parallelism_threads=2,
        device_count={'GPU': 0} if args.device == 'cpu' else {}, allow_soft_placement=True,
    )
    session_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(graph=graph, config=session_config) as session:
        if args.resume:
            saver.restore(session, previous)
        else:
            session.run(initialize)
        session.run(iterator.initializer)
        step = int(session.run(global_step))
        if step >= args.num_train_steps:
            raise ValueError(f'Checkpoint step {step} is already at or beyond num-train-steps')
        print(f'Training from step {step} to {args.num_train_steps}', flush=True)
        while step < args.num_train_steps:
            step, loss, mlm, nsp = session.run([
                step_after_train, train_model['loss'], train_model['masked_lm_loss'], train_model['next_sentence_loss'],
            ])
            if step == 1 or step % args.log_every_steps == 0 or step == args.num_train_steps:
                print(f'step={step} loss={loss:.6f} MLM={mlm:.6f} NSP={nsp:.6f}', flush=True)
            if step % args.save_checkpoints_steps == 0 or step == args.num_train_steps:
                path = saver.save(session, str(output / 'model.ckpt'), global_step=step)
                print(f'Saved {path}', flush=True)
                if args.max_eval_steps:
                    metrics = evaluate(tf, session, eval_iterator, eval_features, eval_model, args.max_eval_steps)
                    print('Evaluation on training input: ' + json.dumps(metrics, sort_keys=True), flush=True)
    print(f'Final checkpoint: {output / ("model.ckpt-" + str(step))}', flush=True)
