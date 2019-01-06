import os
import sys
import h5py 
import cPickle 
import argparse
import multiprocessing
import time

import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')

from utils import normalize, Tokenizer

parser = argparse.ArgumentParser(description='Tokenize datasets')
parser.add_argument('--split', default='train', help='train, dev, test')
parser.add_argument('--tagger', default='okt', help='whitespace, okt, mecab')
parser.add_argument('--num_chunk', type=int, default=9, help='the number of chunks (train: 9, dev: 1, test: 2)')
parser.add_argument('--vocab_root', default='/base/data', help='folder to load bmsd_vocab.cPickle')
parser.add_argument('--data_root', default='/data', help='folder to load input chunks')
parser.add_argument('--output_root', default='/data/output/tmp', help='folder to save tokenized chunks')
args = parser.parse_args()

if not os.path.exists(args.output_root):
    os.makedirs(args.output_root)

columns = ['product', 'brand', 'model']
bmsd_vocab = cPickle.loads(open(os.path.join(args.vocab_root, "bmsd_vocab.cPickle")).read()) 

input_format = "%s.chunk.%02d"
output_format = "%s_tokenized.chunk.%02d"


def append(h_in, h_out, split): 
    tkn = Tokenizer(args.tagger)
    data = h_in[split] # train, dev, test
    cur_size = len(data['product'])

    bcateid = data['bcateid'][()]
    mcateid = data['mcateid'][()]
    scateid = data['scateid'][()]
    dcateid = data['dcateid'][()]

    def get_label(i, vocab_type="bmsd"):
        b = bcateid[i]
        m = mcateid[i]
        s = scateid[i]
        d = dcateid[i]

        if split == 'train':
            if vocab_type == "bmsd":
                y = bmsd_vocab['%s>%s>%s>%s' % (b, m, s, d)] 
            else:
                raise
            return y
        else:
            return -1

    h_out['img_feat'] = data['img_feat'][:]
    h_out['pid'] = data['pid'][:]
    h_out['label'] = [get_label(i, vocab_type="bmsd") for i in range(cur_size)]

    for col in columns:
        result = []
        for i in range(cur_size): 
            txt = normalize(data[col][i], col_type=col)
            words = tkn.tokenize(txt)
            result.append(np.string_(words))

        h_out[col] = np.array(result, dtype="S1000")


def convert(i):
    chunk_id = i+1

    print("[*] Convert %s %02d" % (args.split, chunk_id))
    start = time.time()

    h_in = h5py.File(os.path.join(args.data_root, input_format % (args.split, chunk_id)), 'r')
    h_out = h5py.File(os.path.join(args.output_root, output_format % (args.split, chunk_id)), 'w')

    append(h_in, h_out, split=args.split)
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
