import itertools
import os
import sys

import tensorflow as tf
from tensorflow.keras.utils import Progbar
import numpy as np

from models import mlp
from utils import ModelConfig
from misc import evaluate, to_cate_ids, evaluate_bmsd, evaluate_lfirst
from logger import QuickLogger

flags = tf.app.flags

FLAGS = flags.FLAGS

# common options
flags.DEFINE_string('root_dir', '/data', 'root directory')
flags.DEFINE_string('output_dir', None, 'output directory')
flags.DEFINE_string('exp_dir', None, 'experiment directory')
flags.DEFINE_boolean('is_quant', False, 'quantization or not')
flags.DEFINE_integer('weight_bits', 8, 'the number of bits of weight')
flags.DEFINE_integer('activation_bits', 8, 'the number of bits of activation')
flags.DEFINE_enum('data_split', 'dev', ['dev', 'test'], 'data split')
flags.DEFINE_integer('seed', 2019, 'seed')
flags.DEFINE_integer('val_epoch', None, 'validation epoch')

# options for dataset
flags.DEFINE_enum('tokenizer', 'okt', ['okt', 'mecab', 'whitespace', 'okt_sub'],
                  'tokenizer used to generate .tfrecord')
flags.DEFINE_integer('vocab_size', 150000, 'embedding size')
flags.DEFINE_integer('max_len', 20, 'maximum word count per product')
flags.DEFINE_string('postfix', '', 'postfix appended to .tfrecord')
flags.DEFINE_integer('total_chunk', 1, 'the number of tfrecords for validation')
flags.DEFINE_integer('batch_size', 1024, 'batch size')
flags.DEFINE_boolean('zero_based', False, 'label is zero-based')

# options for network
flags.DEFINE_enum('model', 'mlp', ['mlp', 'lstm'], 'type of classifier')
flags.DEFINE_boolean('bmsd', False, 'whether bmsd is used or not')
flags.DEFINE_enum('bmsd_eval_type', 'label_only', ['label_only', 'label_first', 'acc_first', 'bmsd_only'], 'whether bmsd is used or not')
flags.DEFINE_boolean('use_b', False, 'whether b is used or not')
flags.DEFINE_boolean('use_m', False, 'whether m is used or not')
flags.DEFINE_integer('embd_size', 256, 'embedding size')
flags.DEFINE_enum('embd_avg_type', 'divide_by_valid', ['divide_by_valid', 'simple_mean', 'simple_sum'], 'type of averaged embedding')
flags.DEFINE_enum('embd_init_type', 'none', ['none', 'uniform'], 'type of initializer for embedding')
flags.DEFINE_enum('noise_type', 'adv', ['none', 'adv'], 'type of classifier')
flags.DEFINE_float('unigram_adv_noise', 5e-1, 'adv noise strenpidh for unigram')
flags.DEFINE_float('img_adv_noise', 5e0, 'adv noise strenpidh for img feature')
flags.DEFINE_float('embd_dropout_rate', 0.5, 'dropout rate for embedding')
flags.DEFINE_float('fc_dropout_rate', 0.5, 'dropout rate for fc')
flags.DEFINE_integer('fc_img_feat', 0, 'the size of fc layer for image feat')
flags.DEFINE_multi_integer('fc_layers', [4096], 'the size of fc layer for classifiers')

logger = QuickLogger(log_dir=FLAGS.exp_dir, log_name='pred_logs.txt').get_logger()

def val(sess, val_vars):
    sess.run([val_vars.it.initializer, tf.local_variables_initializer()])

    val_progbar = Progbar(None, stateful_metrics=['num_word',])

    preds = []
    preds_b = []
    preds_m = []
    preds_s = []
    preds_d = []
    pids = []
    indices = tf.argmax(val_vars.logits, 1)

    feed_dict = [val_vars.num_word_metric, indices, val_vars.pid]
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
                (num_word, _), pred, pid, pred_b, pred_m, pred_s, pred_d = sess.run(feed_dict)
                preds_b.append(pred_b)
                preds_m.append(pred_m)
                preds_s.append(pred_s)
                preds_d.append(pred_d)
            else:
                (num_word, _), pred, pid = sess.run(feed_dict)

            val_progbar.update(step, [('num_word', num_word),])
            preds.append(pred)
            pids.append(pid)
            step += 1

    except tf.errors.OutOfRangeError:
        flat_preds = list(itertools.chain(*preds))
        flat_pids = list(itertools.chain(*pids))

        if FLAGS.bmsd and FLAGS.bmsd_eval_type != 'label_only':
            flat_preds_b = list(itertools.chain(*preds_b))
            flat_preds_m = list(itertools.chain(*preds_m))
            flat_preds_s = list(itertools.chain(*preds_s))
            flat_preds_d = list(itertools.chain(*preds_d))

        f = open(os.path.join(FLAGS.exp_dir, "submission.%s-from-%d-%s.tsv" % (FLAGS.data_split, FLAGS.val_epoch, FLAGS.bmsd_eval_type)), "w")
        for i, (pid, yhat) in enumerate(zip(flat_pids, flat_preds)):
            row = [pid]

            if FLAGS.bmsd:
                b, m, s, d = to_cate_ids[yhat]
                if FLAGS.bmsd_eval_type == 'label_only':
                    row.append(str(b))
                    row.append(str(m))
                    row.append(str(s))
                    row.append(str(d))
                elif FLAGS.bmsd_eval_type == 'label_first':
                    row.append(str(b))
                    row.append(str(m))
                    if s == -1:
                        row.append(str(flat_preds_s[i]))
                    else:
                        row.append(str(s))
                    if d == -1:
                        row.append(str(flat_preds_d[i]))
                    else:
                        row.append(str(d))
                elif FLAGS.bmsd_eval_type == 'bmsd_only':
                    row.append(str(flat_preds_b[i]))
                    row.append(str(flat_preds_m[i]))
                    row.append(str(flat_preds_s[i]))
                    row.append(str(flat_preds_d[i]))
                elif FLAGS.bmsd_eval_type == 'acc_first':
                    row.append(str(b))
                    row.append(str(m))
                    #row.append(str(-1))
                    #row.append(str(-1))
                    row.append(str(flat_preds_s[i]))
                    row.append(str(flat_preds_d[i]))
                else:
                    row += list(map(str, to_cate_ids[yhat]))
            else:
                row += list(map(str, to_cate_ids[yhat]))

            print >>f, "\t".join(row)
        f.close()
    
def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    np.random.seed(FLAGS.seed)

    tf.gfile.MakeDirs(FLAGS.exp_dir)

    DATA_DIR = os.path.join(FLAGS.root_dir, "output_%s" % FLAGS.tokenizer)
    TFRECORD_FORMAT = DATA_DIR + ("/%s%s" % (FLAGS.data_split, FLAGS.postfix)) + ".%02d.tfrecord"

    val_files = [TFRECORD_FORMAT % i for i in range(1, FLAGS.total_chunk+1)]

    model_config = ModelConfig(FLAGS, train=False)
    logger.info('[!] FLAGS.*')
    logger.info(FLAGS.flag_values_dict())
    logger.info('[!] model_config.*')
    logger.info(model_config)
    print(model_config)

    with tf.Graph().as_default() as graph:
        tf.set_random_seed(FLAGS.seed)

        val_vars = mlp.Classifier(val_files, train=False, test=True, config=model_config)
        saver = tf.train.Saver(max_to_keep=30)
        ckpt_path = os.path.join(FLAGS.exp_dir, "model.ckpt-%d" % (FLAGS.val_epoch))
    
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt_path)
            logger.info('[!] Validation...')
            val(sess, val_vars)

if __name__ == '__main__':
    flags.mark_flag_as_required('output_dir')
    flags.mark_flag_as_required('exp_dir')
    flags.mark_flag_as_required('model')
    flags.mark_flag_as_required('val_epoch')
    #tf.app.run()
    main(sys.argv[1])
