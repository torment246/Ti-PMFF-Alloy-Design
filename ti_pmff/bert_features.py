"""Ti-BERT CLS feature extraction."""
from pathlib import Path
import types

import numpy as np
import pandas as pd

from .text_input import build_text_inputs

_LAYER_NORM_COUNTER = 0


def _reset_layer_norm_counter():
    global _LAYER_NORM_COUNTER
    _LAYER_NORM_COUNTER = 0

def _legacy_dense(inputs, units, activation=None, kernel_initializer=None, name=None, use_bias=True, **kwargs):
    import tensorflow as tf

    input_shape = inputs.shape.as_list()
    if not input_shape or input_shape[-1] is None:
        raise ValueError("Legacy dense requires a known final input dimension.")
    scope_name = name or "dense"
    with tf.compat.v1.variable_scope(scope_name):
        kernel = tf.compat.v1.get_variable(
            "kernel",
            shape=[input_shape[-1], units],
            initializer=kernel_initializer or tf.compat.v1.glorot_uniform_initializer(),
        )
        flat = tf.reshape(inputs, [-1, input_shape[-1]])
        output = tf.matmul(flat, kernel)
        if use_bias:
            bias = tf.compat.v1.get_variable("bias", shape=[units], initializer=tf.zeros_initializer())
            output = tf.nn.bias_add(output, bias)
        output_shape = tf.concat([tf.shape(inputs)[:-1], [units]], axis=0)
        output = tf.reshape(output, output_shape)
        if activation is not None:
            output = activation(output)
        return output

def _legacy_layer_norm(input_tensor, name=None):
    import tensorflow as tf

    global _LAYER_NORM_COUNTER
    hidden_size = input_tensor.shape.as_list()[-1]
    if hidden_size is None:
        raise ValueError("Legacy layer norm requires a known final dimension.")
    if name is None:
        scope_name = "layer_normalization" if _LAYER_NORM_COUNTER == 0 else f"layer_normalization_{_LAYER_NORM_COUNTER}"
        _LAYER_NORM_COUNTER += 1
    else:
        scope_name = name
    with tf.compat.v1.variable_scope(scope_name):
        gamma = tf.compat.v1.get_variable("gamma", shape=[hidden_size], initializer=tf.ones_initializer())
        beta = tf.compat.v1.get_variable("beta", shape=[hidden_size], initializer=tf.zeros_initializer())
        mean, variance = tf.nn.moments(input_tensor, axes=[-1], keepdims=True)
        normalized = (input_tensor - mean) / tf.sqrt(variance + 0.001)
        return normalized * gamma + beta

def patch_tensorflow_v1_compat():
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    aliases = {
        "variable_scope": tf.compat.v1.variable_scope,
        "get_variable": tf.compat.v1.get_variable,
        "get_variable_scope": tf.compat.v1.get_variable_scope,
        "truncated_normal_initializer": tf.compat.v1.truncated_normal_initializer,
        "trainable_variables": tf.compat.v1.trainable_variables,
        "global_variables_initializer": tf.compat.v1.global_variables_initializer,
        "Session": tf.compat.v1.Session,
        "ConfigProto": tf.compat.v1.ConfigProto,
        "gfile": tf.compat.v1.gfile,
        "logging": tf.compat.v1.logging,
        "flags": tf.compat.v1.flags,
        "app": tf.compat.v1.app,
        "assert_less_equal": tf.compat.v1.assert_less_equal,
        "trainable_variables": tf.compat.v1.trainable_variables,
    }
    for name, value in aliases.items():
        if not hasattr(tf, name):
            setattr(tf, name, value)
    tf.compat.v1.layers = types.SimpleNamespace(dense=_legacy_dense)
    if not hasattr(tf.train, "list_variables"):
        tf.train.list_variables = tf.compat.v1.train.list_variables
    if not hasattr(tf.train, "init_from_checkpoint"):
        tf.train.init_from_checkpoint = tf.compat.v1.train.init_from_checkpoint
    return tf


def extract_bert_cls_features(df, output_csv, checkpoint_prefix, bert_config, vocab_file,
                             max_seq_length=128, layer=11, batch_size=32, device="cpu"):
    tf = patch_tensorflow_v1_compat()
    from pretraining import input_examples, modeling, tokenization

    if device == "auto":
        device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"
    if device == "cuda" and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("TensorFlow cannot see a GPU. Use --bert-device cpu for the CPU demo.")
    tokenizer = tokenization.FullTokenizer(str(vocab_file), do_lower_case=True)
    examples = [input_examples.InputExample(i, line, None) for i, line in enumerate(build_text_inputs(df))]
    features = input_examples.convert_examples_to_features(examples, max_seq_length, tokenizer)
    config = modeling.BertConfig.from_json_file(str(bert_config))

    graph = tf.Graph()
    with graph.as_default(), tf.device("/GPU:0" if device == "cuda" else "/CPU:0"):
        _reset_layer_norm_counter()
        modeling.layer_norm = _legacy_layer_norm
        input_ids = tf.compat.v1.placeholder(tf.int32, [None, max_seq_length])
        input_mask = tf.compat.v1.placeholder(tf.int32, [None, max_seq_length])
        input_types = tf.compat.v1.placeholder(tf.int32, [None, max_seq_length])
        model = modeling.BertModel(config, False, input_ids, input_mask, input_types,
                                   use_one_hot_embeddings=False)
        cls = model.get_all_encoder_layers()[int(layer)][:, 0, :]
        saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.trainable_variables())

    session_config = tf.compat.v1.ConfigProto(
        device_count={"GPU": 0} if device == "cpu" else {},
        intra_op_parallelism_threads=4, inter_op_parallelism_threads=2,
        allow_soft_placement=True,
    )
    session_config.gpu_options.allow_growth = True
    rows = []
    with tf.compat.v1.Session(graph=graph, config=session_config) as session:
        saver.restore(session, str(checkpoint_prefix))
        for start in range(0, len(features), batch_size):
            batch = features[start:start + batch_size]
            values = session.run(cls, feed_dict={
                input_ids: np.asarray([f.input_ids for f in batch], dtype=np.int32),
                input_mask: np.asarray([f.input_mask for f in batch], dtype=np.int32),
                input_types: np.asarray([f.input_type_ids for f in batch], dtype=np.int32),
            })
            rows.extend([[round(float(value), 6) for value in row] for row in values])
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv
