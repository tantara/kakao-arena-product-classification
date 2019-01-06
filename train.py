import itertools
import os
import sys

import tensorflow as tf
from tensorflow.keras.utils import Progbar
import numpy as np

from models import mlp
from utils import ModelConfig
from misc import evaluate, evaluate_bmsd
from logger import QuickLogger

flags = tf.app.flags

FLAGS = flags.FLAGS

# common options
flags.DEFINE_string('root_dir', '/data', 'root directory')
flags.DEFINE_string('output_dir', None, 'output directory')
flags.DEFINE_string('exp_dir', None, 'experiment directory')
flags.DEFINE_string('pretrained', None, 'pretrained weights')
flags.DEFINE_boolean('is_quant', False, 'quantization or not')
flags.DEFINE_integer('weight_bits', 8, 'the number of bits of weight')
flags.DEFINE_integer('activation_bits', 8, 'the number of bits of activation')
flags.DEFINE_integer('seed', 2019, 'seed')

# options for dataset
flags.DEFINE_enum('tokenizer', 'okt', ['okt', 'mecab', 'whitespace', 'okt_sub'],
                  'tokenizer used to generate .tfrecord')
flags.DEFINE_integer('vocab_size', 150000, 'embedding size')
flags.DEFINE_integer('max_len', 20, 'maximum word count per product')
flags.DEFINE_string('postfix', '', 'postfix appended to .tfrecord')
flags.DEFINE_integer('total_chunk', 20, 'the number of all tfrecords')
flags.DEFINE_integer('train_chunk', 19, 'the number of tfrecords for training')
flags.DEFINE_integer('val_chunk', 1, 'the number of tfrecords for validation')
flags.DEFINE_integer('num_train_steps', None, 'the number of steps per epoch for training')
flags.DEFINE_integer('batch_size', 1024, 'batch size')
flags.DEFINE_boolean('zero_based', False, 'label is zero-based')

# options for network
flags.DEFINE_enum('model', 'mlp', ['mlp', 'lstm'], 'type of classifier')
flags.DEFINE_boolean('bmsd', False, 'whether bmsd is used or not')
flags.DEFINE_boolean('use_b', False, 'whether b is used or not')
flags.DEFINE_boolean('use_m', False, 'whether m is used or not')
flags.DEFINE_integer('embd_size', 256, 'embedding size')
flags.DEFINE_enum('embd_avg_type', 'divide_by_valid', ['divide_by_valid', 'simple_mean', 'simple_sum'], 'type of averaged embedding')
flags.DEFINE_enum('embd_init_type', 'none', ['none', 'uniform', 'random_normal', 'truncated_normal'], 'type of initializer for embedding')
flags.DEFINE_float('norm_mean', 0.0, 'mean of normal distribution')
flags.DEFINE_float('norm_std', 0.1, 'mean of normal distribution')
flags.DEFINE_enum('noise_type', 'adv', ['none', 'adv'], 'type of classifier')
flags.DEFINE_float('unigram_adv_noise', 5e-1, 'adv noise strength for unigram')
flags.DEFINE_float('img_adv_noise', 5e0, 'adv noise strength for img feature')
flags.DEFINE_float('embd_dropout_rate', 0.5, 'dropout rate for embedding')
flags.DEFINE_float('fc_dropout_rate', 0.5, 'dropout rate for fc')
flags.DEFINE_integer('fc_img_feat', 0, 'the size of fc layer for image feat')
flags.DEFINE_multi_integer('fc_layers', [4096], 'the size of fc layer for classifiers')

# options for training
flags.DEFINE_integer('epochs', 30, 'training epochs')
flags.DEFINE_integer('val_per_epoch', 2, 'per-epoch to validate')
flags.DEFINE_enum('optimizer', 'adam', ['adam', 'momentum', 'rmsprop'], 'optimizer')
flags.DEFINE_float('base_lr', 1e-3, 'base learning rate')
flags.DEFINE_integer('decay_steps', 1000, 'decay steps for lr decay')
flags.DEFINE_float('decay_rate', 0.999, 'decay rate for lr decay')
flags.DEFINE_boolean('staircase', True, 'staircase for lr decay')
flags.DEFINE_float('momentum', 0.9, 'momentum for optimizer')

logger = QuickLogger(log_dir=FLAGS.exp_dir).get_logger()

def train(sess, train_vars):
    sess.run([train_vars.it.initializer, tf.local_variables_initializer()])

    train_progbar = Progbar(FLAGS.num_train_steps, stateful_metrics=['loss', 'acc', 'num_word', 'lr', 'cls_loss', 'adv_loss'])
    feed_dict = [train_vars.op, train_vars.loss_metric, train_vars.acc_metric, train_vars.num_word_metric, train_vars.lr,
                 train_vars.cls_loss_metric, train_vars.adv_loss_metric]
    if FLAGS.bmsd:
        feed_dict.append(train_vars.loss_b_metric)
        feed_dict.append(train_vars.loss_m_metric)
        feed_dict.append(train_vars.loss_s_metric)
        feed_dict.append(train_vars.loss_d_metric)
        feed_dict.append(train_vars.acc_b_metric)
        feed_dict.append(train_vars.acc_m_metric)
        feed_dict.append(train_vars.acc_s_metric)
        feed_dict.append(train_vars.acc_d_metric)

    try:
        step = 0
        while FLAGS.num_train_steps == None or step < FLAGS.num_train_steps:
            if FLAGS.bmsd:
                _, (loss, _), (acc, _), (num_word, _), cur_lr, (cls_loss, _), (adv_loss, _), \
                (loss_b, _), (loss_m, _), (loss_s, _), (loss_d, _), (acc_b, _), (acc_m, _), (acc_s, _), (acc_d, _), \
                    = sess.run(feed_dict)
                updates = [('loss', loss), ('acc', acc), ('num_word', num_word), ('lr', cur_lr),
                                        ('cls_loss', cls_loss), ('adv_loss', adv_loss), ]
                updates.append(('loss_b', loss_b))
                updates.append(('loss_m', loss_m))
                updates.append(('loss_s', loss_s))
                updates.append(('loss_d', loss_d))
                updates.append(('acc_b', acc_b))
                updates.append(('acc_m', acc_m))
                updates.append(('acc_s', acc_s))
                updates.append(('acc_d', acc_d))

            else:
                _, (loss, _), (acc, _), (num_word, _), cur_lr, (cls_loss, _), (adv_loss, _)  = sess.run(feed_dict)
                updates = [('loss', loss), ('acc', acc), ('num_word', num_word), ('lr', cur_lr),
                                        ('cls_loss', cls_loss), ('adv_loss', adv_loss), ]

            train_progbar.update(step, updates)

            if step % 1000 == 0 or (FLAGS.num_train_steps != None and step == FLAGS.num_train_steps-1):
                if FLAGS.bmsd:
                    print_str = "[*] Step %d - Loss: %.3f, Acc: %.4f, Num Word: %.2f, LR: %.6f, Loss(bmsd)(%.3f, %.3f, %3f, %3f), Acc(bmsd)(%.4f, %.4f, %.4f, %.4f)" % (step, loss, acc, num_word, cur_lr, loss_b, loss_m, loss_s, loss_d, acc_b, acc_m, acc_s, acc_d)
                else:
                    print_str = "[*] Step %d - Loss: %.3f, Acc: %.4f, Num Word: %.2f, LR: %.6f" % (step, loss, acc, num_word, cur_lr)
                logger.info(print_str)
            step+=1
    except tf.errors.OutOfRangeError:
        logger.info("Total step is %d" % step)
        print("Total step is %d" % step)
        FLAGS.num_train_steps = step
    

def val(sess, val_vars):
    sess.run(val_vars.it.initializer)

    preds = []
    preds_b = []
    preds_m = []
    preds_s = []
    preds_d = []
    gts = []
    indices = tf.argmax(val_vars.logits, 1)
    feed_dict = [val_vars.loss_metric, val_vars.acc_metric, val_vars.num_word_metric, indices, val_vars.gts]

    if FLAGS.bmsd:
        unshift = 1 if FLAGS.zero_based else 0
        indices_b = tf.argmax(val_vars.logits_b, 1)+unshift
        indices_m = tf.argmax(val_vars.logits_m, 1)+unshift
        indices_s = tf.argmax(val_vars.logits_s, 1)+unshift
        indices_d = tf.argmax(val_vars.logits_d, 1)+unshift
        feed_dict.append(indices_b)
        feed_dict.append(indices_m)
        feed_dict.append(indices_s)
        feed_dict.append(indices_d)

    try:
        step = 0
        while True:
            if FLAGS.bmsd:
                (loss, _), (acc, _), (num_word, _), pred, gt, pred_b, pred_m, pred_s, pred_d = sess.run(feed_dict)

                preds_b.append(pred_b)
                preds_m.append(pred_m)
                preds_s.append(pred_s)
                preds_d.append(pred_d)
            else:
                (loss, _), (acc, _), (num_word, _), pred, gt = sess.run(feed_dict)

            preds.append(pred)
            gts.append(gt)
    
    except tf.errors.OutOfRangeError:
        logger.info('[*] Validation loss: %.3f, acc: %.4f, num_word: %.2f' % (loss, acc, num_word))
        print('[*] Validation loss: %.3f, acc: %.4f, num_word: %.2f' % (loss, acc, num_word))

        if FLAGS.bmsd:
            flat_preds = list(itertools.chain(*preds))
            flat_gts = list(itertools.chain(*gts))
            flat_preds_b = list(itertools.chain(*preds_b))
            flat_preds_m = list(itertools.chain(*preds_m))
            flat_preds_s = list(itertools.chain(*preds_s))
            flat_preds_d = list(itertools.chain(*preds_d))

            score, base_scores = evaluate(flat_gts, flat_preds)
            print('\n[!] Label Score: %.4f' % (score))

            bmsd_score, bmsd_scores = evaluate_bmsd(flat_gts, flat_preds_b, flat_preds_m, flat_preds_s, flat_preds_d)
            print('BMSD Score: %.4f' % (bmsd_score))

            final_score = 0.0
            for r, s1, s2 in zip([1.0,1.2,1.3,1.4], base_scores, bmsd_scores):
                final_score += r * max(s1, s2)
            final_score /= 4.0

            print('[!] Final Score: %.4f' % (final_score))
        else:
            flat_preds = list(itertools.chain(*preds))
            flat_gts = list(itertools.chain(*gts))
            score, base_scores = evaluate(flat_gts, flat_preds)
            print('\n[!] Label Score: %.4f' % (score))

        return score

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    np.random.seed(FLAGS.seed)

    tf.gfile.MakeDirs(FLAGS.exp_dir)

    DATA_DIR = os.path.join(FLAGS.root_dir, "output_%s" % FLAGS.tokenizer)
    TFRECORD_FORMAT = DATA_DIR + ("/train%s" % (FLAGS.postfix)) + ".%02d.tfrecord"

    #assert FLAGS.train_chunk <= FLAGS.total_chunk
    #assert FLAGS.train_chunk + FLAGS.val_chunk <= FLAGS.total_chunk

    train_files = [TFRECORD_FORMAT % i for i in range(1, FLAGS.train_chunk+1)] # 1~19
    val_files = [TFRECORD_FORMAT % i for i in range(FLAGS.total_chunk-FLAGS.val_chunk+1, FLAGS.total_chunk+1)] # 20

    model_config = ModelConfig(FLAGS)
    logger.info('[!] FLAGS.*')
    logger.info(FLAGS.flag_values_dict())
    logger.info('[!] model_config.*')
    logger.info(model_config)
    print(model_config)

    with tf.Graph().as_default() as graph:
        tf.set_random_seed(FLAGS.seed)
        max_score = 0.0
    
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as sess:
            train_vars = mlp.Classifier(train_files, train=True, config=model_config)
            val_vars = mlp.Classifier(val_files, reuse=True, train=False, config=model_config)
            train_vars.build_optimizer()
            saver = tf.train.Saver(max_to_keep=50)
            gvars = tf.global_variables()
            tvars = []
            for gvar in gvars:
                if 'quant' not in gvar.name:
                    tvars.append(gvar)
            load_saver = tf.train.Saver(var_list=tvars)

            sess.run(tf.global_variables_initializer())
            if FLAGS.pretrained:
                load_saver.restore(sess, FLAGS.pretrained)
            #sess.run(tf.initialize_all_variables())
            train_total_steps = None

            for e in range(1, FLAGS.epochs+1):
                logger.info('\n[*] Epoch %d/%d' % (e, FLAGS.epochs))
                print('\n[*] Epoch %d/%d' % (e, FLAGS.epochs))

                # Training
                logger.info('[!] Training...')
                train(sess, train_vars)

                # Validation
                if e % FLAGS.val_per_epoch == 0:
                    logger.info('[!] Validation...')
                    score = val(sess, val_vars)

                    # Save a checkpoint
                    logger.info('[*] Save the current model')
                    output_path = os.path.join(FLAGS.exp_dir, "model.ckpt")
                    #if e >= 20:
                    #    save_path = saver.save(sess, output_path, global_step=e)
                    save_path = saver.save(sess, output_path, global_step=e)

                    if score > max_score:
                        max_score = score
                        logger.info('[*] Save the best model at %d(score: %.4f)' % (e, max_score))
                        output_path = os.path.join(FLAGS.exp_dir, "best.ckpt")
                        save_path = saver.save(sess, output_path)

if __name__ == '__main__':
    flags.mark_flag_as_required('output_dir')
    flags.mark_flag_as_required('exp_dir')
    flags.mark_flag_as_required('model')
    #tf.app.run()
    main(sys.argv[1])
