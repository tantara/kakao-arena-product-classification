import os
import sys
import h5py 
import argparse
import multiprocessing
import time

import numpy as np
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf8')

from misc import Option, to_cate_ids

parser = argparse.ArgumentParser(description='Generate tfrecords')
parser.add_argument('--split', default='train', help='train, dev, test')
parser.add_argument('--num_chunk', type=int, default=20, help='the number of chunks (train: 20, dev: 1, test: 1)')
parser.add_argument('--input_root', default='/data/output/tmp', help='folder to load shuffled chunks')
parser.add_argument('--output_root', default='/data/output', help='folder to save tfrecords')
parser.add_argument('--shuffle', type=lambda x: (str(x).lower() == 'true'), default=True, help='shuffle indices in chunks')
args = parser.parse_args()

if not os.path.exists(args.output_root):
    os.makedirs(args.output_root)

opt = Option("./config.json") 

final_format = "%s_splitted.chunk.%02d"
if args.shuffle:
    final_format = "%s_shuffled.chunk.%02d"
tfrecord_format = "%s" + ("-%d-max%d" % (opt.unigram_hash_size, opt.max_len)) + ".%02d.tfrecord"


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def parse_data(texts, i):
    product = " ".join([text[i].strip() for text in texts]).split()
    words = [w.strip() for w in product]
    words = [w for w in words if len(w) >= opt.min_word_length and len(w) <= opt.max_word_length]
    def unique_words(words):
        uniq = []
        for word in words:
            if word not in uniq:
                uniq.append(word)
        return uniq
    #words = list(set(words))
    words = unique_words(words)

    unigram_res = np.full(opt.max_len, 0, dtype=np.int)
    unigram_indices = [(hash(w) % (opt.unigram_hash_size - 1) + 1) for w in words][:opt.max_len]
    unigram_res[:len(unigram_indices)] = unigram_indices

    return unigram_res


def append(texts, img_feat, label, pid, writer):
    for i in range(len(img_feat)):
        unigram_data = parse_data(texts, i)
        if args.split == "train":
            b, m, s, d = to_cate_ids[label[i]]
        else:
            b, m, s, d = (-1, -1, -1, -1)
        feature = {
            'img_feat': _float_feature(img_feat[i]), 
            'unigram': _int64_feature(unigram_data),
            'label': _int64_feature([label[i]]),
            'b': _int64_feature([b]),
            'm': _int64_feature([m]),
            's': _int64_feature([s]),
            'd': _int64_feature([d]),
            'pid': _bytes_feature(pid[i])
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString()) 


def convert(i):
    chunk_id = i+1

    print("[*] Convert %s %02d" % (args.split, chunk_id))
    start = time.time()

    h_in = h5py.File(os.path.join(args.input_root, final_format % (args.split, chunk_id)), 'r')
    writer = tf.python_io.TFRecordWriter(os.path.join(args.output_root, tfrecord_format % (args.split, chunk_id)))

    texts = [h_in[col][()] for col in ['product', 'brand', 'model']] 
    img_feat = h_in['img_feat'][()]
    label = h_in['label'][()]
    pid = h_in['pid'][()]

    append(texts, img_feat, label, pid, writer)
    print("[*] Done %02d %.2fsec" % (chunk_id, time.time() - start))


def main():
    workers = []
    for i in range(args.num_chunk):
        t = multiprocessing.Process(target=convert, args=(i,))
        workers.append(t)
        t.start()

    for worker in workers:
        worker.join()


if __name__ == '__main__':
    print('args', args)
    main()
