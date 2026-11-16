"""Extract token embeddings or attention matrices from a BERT checkpoint."""

import argparse
import json

import numpy as np

from ti_pmff.bert_features import (
    patch_tensorflow_v1_compat, _legacy_layer_norm, _reset_layer_norm_counter,
)
from ti_pmff.config import DEFAULT_BERT_CONFIG, DEFAULT_VOCAB, DEFAULT_CHECKPOINT_PREFIX
from ti_pmff.paths import resolve_input_path, resolve_output_path, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=('embedding', 'attention'), default='embedding')
    parser.add_argument('--layers', default='11')
    parser.add_argument('--bert_config_file', default=DEFAULT_BERT_CONFIG)
    parser.add_argument('--vocab_file', default=DEFAULT_VOCAB)
    parser.add_argument('--input_file', required=True, help='One text record per line; paths are relative to the repository root')
    parser.add_argument('--output_file_embedding', default='outputs/features/token_embeddings.jsonl')
    parser.add_argument('--output_file_attention', default='outputs/features/attention.jsonl')
    parser.add_argument('--init_checkpoint', default=DEFAULT_CHECKPOINT_PREFIX)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', choices=('cpu', 'auto'), default='cpu')
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_seq_length < 2:
        parser.error('batch_size must be positive and max_seq_length must be at least 2')
    input_file = resolve_input_path(args.input_file)
    config_file = resolve_input_path(args.bert_config_file)
    vocab_file = resolve_input_path(args.vocab_file)
    checkpoint = resolve_path(args.init_checkpoint)
    output = resolve_output_path(args.output_file_embedding if args.target == 'embedding' else args.output_file_attention)
    if input_file == output:
        parser.error('Input and output must be different files')
    tf = patch_tensorflow_v1_compat()
    from . import modeling, tokenization, input_examples

    config = modeling.BertConfig.from_json_file(str(config_file))
    layers = [int(index) for index in args.layers.split(',')]
    if not layers or any(index < -config.num_hidden_layers or index >= config.num_hidden_layers for index in layers):
        parser.error('Layer indices exceed the BERT configuration')
    if args.max_seq_length > config.max_position_embeddings:
        parser.error('Sequence length exceeds the positional embeddings')
    tokenizer = tokenization.FullTokenizer(str(vocab_file), do_lower_case=True)
    examples = input_examples.read_examples(str(input_file))
    if not examples:
        parser.error('Input contains no text records')
    features = input_examples.convert_examples_to_features(examples, args.max_seq_length, tokenizer)
    graph = tf.Graph()
    with graph.as_default():
        _reset_layer_norm_counter()
        modeling.layer_norm = _legacy_layer_norm
        ids = tf.compat.v1.placeholder(tf.int32, [None, args.max_seq_length])
        mask = tf.compat.v1.placeholder(tf.int32, [None, args.max_seq_length])
        types = tf.compat.v1.placeholder(tf.int32, [None, args.max_seq_length])
        model = modeling.BertModel(config, False, ids, mask, types, use_one_hot_embeddings=False)
        all_layers = model.get_all_encoder_layers() if args.target == 'embedding' else model.get_all_encoder_ws()
        tensors = [all_layers[index] for index in layers]
        variables = tf.compat.v1.trainable_variables()
        saver = tf.compat.v1.train.Saver(var_list=variables)
    session_config = tf.compat.v1.ConfigProto(
        device_count={'GPU': 0} if args.device == 'cpu' else {},
        intra_op_parallelism_threads=4, inter_op_parallelism_threads=2,
    )
    session_config.gpu_options.allow_growth = True
    output.parent.mkdir(parents=True, exist_ok=True)
    with tf.compat.v1.Session(graph=graph, config=session_config) as session, output.open('w', encoding='utf-8') as stream:
        saver.restore(session, str(checkpoint))
        for start in range(0, len(features), args.batch_size):
            batch = features[start:start + args.batch_size]
            values = session.run(tensors, feed_dict={
                ids: np.asarray([item.input_ids for item in batch], dtype=np.int32),
                mask: np.asarray([item.input_mask for item in batch], dtype=np.int32),
                types: np.asarray([item.input_type_ids for item in batch], dtype=np.int32),
            })
            for row, feature in enumerate(batch):
                payload = {'linex_index': feature.unique_id}
                if args.target == 'embedding':
                    payload['features'] = [
                        {'token': token, 'layers': [
                            {'index': index, 'values': [round(float(value), 6) for value in tensor[row, position]]}
                            for index, tensor in zip(layers, values)
                        ]} for position, token in enumerate(feature.tokens)
                    ]
                else:
                    payload['attention_ws'] = [
                        {'index': index, 'attention_ws': {
                            f'head_{head}': weights.tolist() for head, weights in enumerate(tensor[row])
                        }} for index, tensor in zip(layers, values)
                    ]
                stream.write(json.dumps(payload) + '\n')
    print(f'Wrote {len(features)} {args.target} records to {output}')


if __name__ == '__main__':
    main()
