from easydict import EasyDict as edict

import tensorflow as tf

def ModelConfig(FLAGS, train=True):
    config = edict()

    config.is_quant = FLAGS.is_quant
    config.weight_bits = FLAGS.weight_bits
    config.activation_bits = FLAGS.activation_bits

    config.model = FLAGS.model
    config.bmsd = FLAGS.bmsd
    config.use_b = FLAGS.use_b
    config.use_m = FLAGS.use_m
    # input
    config.embd_size = FLAGS.embd_size
    config.vocab_size = FLAGS.vocab_size
    config.embd_dropout_rate = FLAGS.embd_dropout_rate
    config.embd_avg_type = FLAGS.embd_avg_type
    config.embd_init_type = FLAGS.embd_init_type
    if train:
        config.norm_mean = FLAGS.norm_mean
        config.norm_std = FLAGS.norm_std
    config.zero_based = FLAGS.zero_based

    # classifier
    config.noise_type = FLAGS.noise_type
    config.unigram_adv_noise = FLAGS.unigram_adv_noise
    config.img_adv_noise = FLAGS.img_adv_noise

    config.fc_img_feat = FLAGS.fc_img_feat
    config.fc_dropout_rate = FLAGS.fc_dropout_rate
    config.fc_layers = [{'size': s, 'act': tf.nn.relu} for s in FLAGS.fc_layers]

    config.l2_regularizer = False # FIXME
    if train:
        config.optimizer = FLAGS.optimizer
        config.base_lr = FLAGS.base_lr
        config.decay_steps = FLAGS.decay_steps
        config.decay_rate = FLAGS.decay_rate
        config.staircase = FLAGS.staircase
        config.momentum = FLAGS.momentum

    return config
