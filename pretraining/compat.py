"""TensorFlow compatibility helpers for BERT graph execution."""

from ti_pmff.bert_features import patch_tensorflow_v1_compat


def configure_modeling():
    tf = patch_tensorflow_v1_compat()
    from . import modeling

    counter = 0

    def layer_norm(inputs, name=None):
        nonlocal counter
        scope = name or ('layer_normalization' if counter == 0 else f'layer_normalization_{counter}')
        counter += 1
        width = inputs.shape.as_list()[-1]
        with tf.compat.v1.variable_scope(scope):
            gamma = tf.compat.v1.get_variable('gamma', [width], initializer=tf.ones_initializer())
            beta = tf.compat.v1.get_variable('beta', [width], initializer=tf.zeros_initializer())
            mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
            return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 0.001)

    def dropout(inputs, probability):
        if not probability:
            return inputs
        # TF1 dropout uses a keep probability.
        return tf.compat.v1.nn.dropout(inputs, keep_prob=1.0 - probability)

    modeling.layer_norm = layer_norm
    modeling.dropout = dropout
    return tf, modeling
