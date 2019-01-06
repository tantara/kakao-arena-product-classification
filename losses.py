import tensorflow as tf

def norm(x):
    s = range(1, len(x.get_shape()))
    alpha = tf.reduce_max(tf.abs(x), s, keepdims=True) + 1e-12
    l2_norm = alpha * tf.sqrt(tf.reduce_sum(tf.pow(x / alpha, 2), s, keepdims=True) + 1e-6)
    return x / l2_norm

def add_adv_noise(x, loss, eps):
    grad = tf.gradients(loss, x, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)[0]
    grad = tf.stop_gradient(grad)
    return x + eps * norm(grad)

def cal_class_loss(labels, logits, ignore_label=None, loss_weight=1.0):
    loss_fn = tf.losses.sparse_softmax_cross_entropy

    if ignore_label != None:
        not_ignore_mask = tf.to_float(tf.not_equal(labels, ignore_label)) * loss_weight
        labels = tf.math.maximum(labels, 0)
        return loss_fn(labels, logits, weights=not_ignore_mask)
    else:
        return loss_fn(labels, logits)

def cal_l2_loss():
    train_vars = tf.trainable_variables()
    l2_vars = []
    for i, tvar in enumerate(train_vars):
        if 'Logit' in tvar.name:
            l2_vars.append(tvar)
    l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in l2_vars ]) * 0.0002
    return l2_loss
