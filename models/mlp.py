import tensorflow as tf
import random

from input_generator import input_generator
from losses import cal_class_loss, add_adv_noise, cal_l2_loss

class Classifier(object):
    def __init__(self, files, reuse=False, train=True, config=None, test=False):
        assert config != None, 'need model_config to build'

        self.embd_size = config.embd_size
        self.vocab_size = config.vocab_size
        self.config = config
        self.test = test

        self.it = input_generator(files)
        self.feats, self.pid, self.gts = self.it.get_next()
        self.num_classes = 4215
        self.labels = self.feats['label']

        bmsd_labels = None
        bmsd_labels = [self.feats['b'], self.feats['m'], self.feats['s'], self.feats['d']]
        self.bmsd_labels = bmsd_labels

        self.num_word = None
        with tf.variable_scope('MLP', reuse=reuse):
            uni_embd, img_feat = self.build_input(self.feats, reuse=reuse)

            noise_type = self.config.noise_type
            adv_loss = None
            #logits, cls_loss = self.build_normal_graph(uni_embd, img_feat, reuse=reuse, train=train)
            #adv_loss = self.build_adv_graph(uni_embd, img_feat, cls_loss, reuse=True, train=train)
            if self.config.bmsd:
                logits, cls_loss, \
                b_logits, m_logits, s_logits, d_logits, \
                b_loss, m_loss, s_loss, d_loss, \
                    = self.build_normal_graph(uni_embd, img_feat, reuse=reuse, train=train, bmsd=True)
            else:
                logits, cls_loss = self.build_normal_graph(uni_embd, img_feat, reuse=reuse, train=train)

            if train:
                if noise_type == 'adv':
                    adv_loss = self.build_adv_graph(uni_embd, img_feat, cls_loss, reuse=True, train=train)

            bmsd_loss = None
            bmsd_logits = None
            if self.config.bmsd:
                bmsd_loss = [b_loss, m_loss, s_loss, d_loss]
                bmsd_logits = [b_logits, m_logits, s_logits, d_logits]

            self.build_graph(self.labels, logits, cls_loss, adv_loss=adv_loss, train=train,
                bmsd_labels=bmsd_labels, bmsd_logits=bmsd_logits, bmsd_loss=bmsd_loss)

        self.logits = logits
        if self.config.bmsd:
            self.logits_b = b_logits
            self.logits_m = m_logits
            self.logits_s = s_logits
            self.logits_d = d_logits

    def build_normal_graph(self, uni_embd, img_feat, reuse, train, bmsd=False):
        embedding = self.build_embedding(uni_embd, img_feat, reuse=reuse, train=train)

        if bmsd:
            logits, logits_b, logits_m, logits_s, logits_d = self.build_logits(embedding, reuse=reuse, train=train, bmsd=True)

            loss = cal_class_loss(self.labels, logits)
            unshift = 1 if self.config.zero_based else 0
            loss_b = cal_class_loss(self.bmsd_labels[0]-unshift, logits_b)
            loss_m = cal_class_loss(self.bmsd_labels[1]-unshift, logits_m)
            loss_s = cal_class_loss(self.bmsd_labels[2]-unshift, logits_s, ignore_label=-1-unshift)
            loss_d = cal_class_loss(self.bmsd_labels[3]-unshift, logits_d, ignore_label=-1-unshift)

            return logits, loss, logits_b, logits_m, logits_s, logits_d, loss_b, loss_m, loss_s, loss_d
        else:
            logits = self.build_logits(embedding, reuse=reuse, train=train)
            loss = cal_class_loss(self.labels, logits)

            return logits, loss

    def build_adv_graph(self, uni_embd, img_feat, cls_loss, reuse, train):
        uni_adv = add_adv_noise(uni_embd, cls_loss, self.config.unigram_adv_noise)
        img_adv = add_adv_noise(img_feat, cls_loss, self.config.img_adv_noise)

        _, adv_loss = self.build_normal_graph(uni_adv, img_adv, reuse=reuse, train=train)
        return adv_loss

    def build_input(self, feats, reuse):
        with tf.variable_scope('Input', reuse=reuse):
            # Unigram
            if self.config.embd_init_type == 'uniform':
                uni_embd_var = tf.get_variable('uni_embd', [self.vocab_size, self.embd_size], initializer=tf.random_uniform_initializer(-1., 1.))
            elif self.config.embd_init_type == 'random_normal':
                seed = random.randint(1, 10000)
                uni_embd_var = tf.get_variable('uni_embd', [self.vocab_size, self.embd_size], initializer=tf.random_normal_initializer(self.config.norm_mean, self.config.norm_std, seed=seed))
            elif self.config.embd_init_type == 'truncated_normal':
                seed = random.randint(1, 10000)
                uni_embd_var = tf.get_variable('uni_embd', [self.vocab_size, self.embd_size], initializer=tf.truncated_normal_initializer(self.config.norm_mean, self.config.norm_std, seed=seed))
            else:
                uni_embd_var = tf.get_variable('uni_embd', [self.vocab_size, self.embd_size])
            uni_embd = tf.nn.embedding_lookup(uni_embd_var, tf.abs(feats['unigram'])) # -1 -> 1

            # Img Feature
            img_feat = tf.clip_by_value(feats['img_feat'], -100, 100)

        return uni_embd, img_feat

    def build_embedding(self, uni_embd, img_feat, reuse, train):
        with tf.variable_scope('Embedding', reuse=reuse):
            uni_mask = tf.not_equal(self.feats['unigram'], 0)
            uni_len = tf.reduce_sum(tf.to_int32(uni_mask), axis=1)
            uni_len = tf.expand_dims(uni_len, 1)
            uni_len = tf.math.maximum(uni_len, 1)

            # uni_embd_masked = uni_embd * tf.to_float(tf.expand_dims(uni_mask, 2))
            uni_embd_masked = uni_embd
            if self.config.embd_avg_type == 'divide_by_valid':
                uni_sum = tf.math.reduce_sum(uni_embd_masked, axis=1)
                uni_average = tf.math.divide(uni_sum, tf.to_float(uni_len))
            elif self.config.embd_avg_type == 'simple_mean':
                uni_average = tf.math.reduce_mean(uni_embd_masked, axis=1)
            elif self.config.embd_avg_type == 'simple_sum':
                uni_sum = tf.math.reduce_sum(uni_embd_masked, axis=1)
                #uni_average = tf.math.divide(uni_sum, tf.to_float(self.embd_size))
                uni_average = uni_sum
            else:
                raise

            if self.config.fc_img_feat > 0:
                img_dense = tf.layers.dense(img_feat,
                                           self.config.fc_img_feat,
                                           activation=tf.nn.relu,
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
            else:
                img_dense = img_feat

            word_sum = uni_average
            self.num_word = uni_len

            word_bn = tf.layers.batch_normalization(word_sum, training=train)
            word_dropout = tf.layers.dropout(word_bn, rate=self.config.embd_dropout_rate, training=train)
            embedding = tf.concat([word_dropout, img_dense], 1)

        return embedding

    def build_logits(self, embedding, reuse, train, bmsd=False):
        with tf.variable_scope('Logit', reuse=reuse) as scope:
            output = embedding

            for i, fc_layer in enumerate(self.config.fc_layers):
                output = tf.layers.dense(output, fc_layer["size"], activation=fc_layer["act"])
                output = tf.layers.batch_normalization(output, training=train)
                output = tf.layers.dropout(output, rate=self.config.fc_dropout_rate, training=train)

            logits = tf.layers.dense(output, self.num_classes)
            if bmsd:
                shift = 0 if self.config.zero_based else 1
                logits_b = tf.layers.dense(output, 57+shift, name='logits_b')
                logits_m = tf.layers.dense(output, 552+shift, name='logits_m')

                if self.config.use_b and self.config.use_m:
                    output_b = logits_b
                    output_b = tf.layers.batch_normalization(output_b, training=train)
                    output_b = tf.layers.dropout(output_b, rate=self.config.fc_dropout_rate, training=train)
                    output_m = logits_m
                    output_m = tf.layers.batch_normalization(output_m, training=train)
                    output_m = tf.layers.dropout(output_m, rate=self.config.fc_dropout_rate, training=train)
                    prev_output = tf.concat([output, output_b, output_m], 1)
                elif self.config.use_b:
                    output_b = logits_b
                    output_b = tf.layers.batch_normalization(output_b, training=train)
                    output_b = tf.layers.dropout(output_b, rate=self.config.fc_dropout_rate, training=train)
                    prev_output = tf.concat([output, output_b], 1)
                elif self.config.use_m:
                    output_m = logits_m
                    output_m = tf.layers.batch_normalization(output_m, training=train)
                    output_m = tf.layers.dropout(output_m, rate=self.config.fc_dropout_rate, training=train)
                    prev_output = tf.concat([output, output_m], 1)
                else:
                    prev_output = output

                logits_s = tf.layers.dense(prev_output, 3190+shift, name='logits_s')
                logits_d = tf.layers.dense(prev_output, 404+shift, name='logits_d')

                return logits, logits_b, logits_m, logits_s, logits_d
            else:
                return logits

    def build_loss(self, cls_loss, adv_loss, train, bmsd_loss=None):
        if self.config.l2_regularizer:
            l2_loss = cal_l2_loss()
            total_cls_loss = cls_loss + l2_loss
        else:
            total_cls_loss = cls_loss


        if self.config.bmsd:
            b_loss, m_loss, s_loss, d_loss = bmsd_loss
            total_cls_loss += (b_loss + m_loss + s_loss + d_loss)
            #total_cls_loss += (b_loss + 1.2*m_loss + 1.3*s_loss + 1.4*d_loss)

        noise_type = self.config.noise_type
        if train and noise_type == 'adv':
            total_adv_loss = adv_loss
        else:
            total_adv_loss = 0.0

        total_loss = tf.cond(tf.constant(train),
                       lambda: total_cls_loss + total_adv_loss,
                       lambda: total_cls_loss)

        return total_loss


    def build_graph(self, labels, logits, cls_loss, adv_loss, train=True,
        bmsd_labels=None, bmsd_logits=None, bmsd_loss=None):

        total_loss = self.build_loss(cls_loss, adv_loss=adv_loss, train=train, bmsd_loss=bmsd_loss)
        self.total_loss = total_loss

        noise_type = self.config.noise_type
        self.loss_metric = tf.metrics.mean(total_loss)
        self.cls_loss_metric = tf.metrics.mean(cls_loss)
        self.acc_metric = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1))
        self.num_word_metric = tf.metrics.mean(self.num_word)

        if self.config.bmsd:
            unshift = 1 if self.config.zero_based else 0
            self.loss_b_metric = tf.metrics.mean(bmsd_loss[0])
            self.acc_b_metric = tf.metrics.accuracy(labels=bmsd_labels[0], predictions=tf.argmax(bmsd_logits[0], 1)+unshift)
            self.loss_m_metric = tf.metrics.mean(bmsd_loss[1])
            self.acc_m_metric = tf.metrics.accuracy(labels=bmsd_labels[1], predictions=tf.argmax(bmsd_logits[1], 1)+unshift)
            self.loss_s_metric = tf.metrics.mean(bmsd_loss[2])
            s_weights = tf.to_float(tf.not_equal(bmsd_labels[2], -1))
            self.acc_s_metric = tf.metrics.accuracy(labels=bmsd_labels[2], predictions=tf.argmax(bmsd_logits[2], 1)+unshift, weights=s_weights)
            self.loss_d_metric = tf.metrics.mean(bmsd_loss[3])
            d_weights = tf.to_float(tf.not_equal(bmsd_labels[3], -1))
            self.acc_d_metric = tf.metrics.accuracy(labels=bmsd_labels[3], predictions=tf.argmax(bmsd_logits[3], 1)+unshift, weights=d_weights)

        if train:
            self.adv_loss_metric = tf.metrics.mean(0.0)
            if noise_type == 'adv':
                self.adv_loss_metric = tf.metrics.mean(adv_loss)

        if train and self.config.is_quant:
            #tvars = tf.trainable_variables()
            tf.contrib.quantize.experimental_create_training_graph(weight_bits=self.config.weight_bits, activation_bits=self.config.activation_bits, scope='MLP')
        elif self.test and self.config.is_quant:
            #tvars = tf.trainable_variables()
            tf.contrib.quantize.experimental_create_eval_graph(weight_bits=self.config.weight_bits, activation_bits=self.config.activation_bits, scope='MLP')
        #tvars = tf.global_variables()
        #for i, tvar in enumerate(tvars):
        #    print(i, tvar.name, tvar)

    def build_optimizer(self, train=True):
        if train:
            global_step = tf.Variable(0, trainable=False)
            self.lr = tf.train.exponential_decay(self.config.base_lr, global_step, self.config.decay_steps, self.config.decay_rate, staircase=self.config.staircase)
            if self.config.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.config.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self.lr, self.config.momentum)
            elif self.config.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.lr)
            else:
                raise

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.op = optimizer.minimize(self.total_loss, global_step=global_step)
