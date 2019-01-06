import os
import sys
import h5py 
import argparse
import multiprocessing
import time

import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')

parser = argparse.ArgumentParser(description='Split&Shuffle datasets')
parser.add_argument('--split', default='train', help='train, dev, test')
parser.add_argument('--num_input_chunk', type=int, default=9, help='the number of chunks (train: 9, dev: 1, test: 2)')
parser.add_argument('--num_output_chunk', type=int, default=20, help='the number of chunks (train: 20, dev: 1, test: 1)')
parser.add_argument('--input_root', default='/data/output/tmp', help='folder to load tokenized chunks')
parser.add_argument('--output_root', default='/data/output/tmp', help='folder to save shuffled chunks')
parser.add_argument('--shuffle', type=lambda x: (str(x).lower() == 'true'), default=True, help='shuffle indices in chunks')
args = parser.parse_args()

if not os.path.exists(args.output_root):
    os.makedirs(args.output_root)

columns = ['img_feat', 'pid', 'label', 'product', 'brand', 'model']

input_format = "%s_tokenized.chunk.%02d"
split_format = "%s_splitted.chunk.%02d"
shuffle_format = "%s_shuffled.chunk.%02d"

chunk_indices = {}


def split_chunk(i):
    chunk_id = i+1

    print("[*] Split %s %d" % (args.split, chunk_id))
    start = time.time()
    h_out = h5py.File(os.path.join(args.output_root, split_format % (args.split, chunk_id)), 'a')

    output_offset = 0
    for j in range(args.num_input_chunk):
        input_chunk_id = j+1

        h_in = h5py.File(os.path.join(args.output_root, input_format % (args.split, input_chunk_id)), 'r')
        cur_size = len(h_in['product'])
        per_chunk = cur_size // args.num_output_chunk
    
        indices = chunk_indices[str(input_chunk_id)]

        cache = {}
        for col in columns:
            cache[col] = h_in[col][()]

        input_offset = i*per_chunk
        cur_indices = indices[input_offset:input_offset+per_chunk]
        for col in columns:
            h_out[col][output_offset:output_offset+per_chunk] = cache[col][cur_indices]

        output_offset += per_chunk
        del cache
        h_in.close()

    h_out.close()
    print("[*] Done %d %.2fsec" % (chunk_id, time.time() - start))


def shuffle_chunk(i):
    chunk_id = i+1

    print("[*] Shuffle %s %d" % (args.split, chunk_id))
    start = time.time()
    h_in = h5py.File(os.path.join(args.output_root, split_format % (args.split, chunk_id)), 'r')
    h_out = h5py.File(os.path.join(args.output_root, shuffle_format % (args.split, chunk_id)), 'w')

    cur_size = len(h_in['product'])
    indices = np.arange(cur_size)
    np.random.shuffle(indices) 

    for col in columns:
        cache = h_in[col][()]
        h_out[col] = cache[indices]

    print("[*] Done %d %.2fsec" % (chunk_id, time.time() - start))


def split():
    workers = []
    for i in range(args.num_output_chunk):
        t = multiprocessing.Process(target=split_chunk, args=(i,))
        workers.append(t)
        t.start()
                
    for worker in workers:
        worker.join()            


def shuffle():
    workers = []
    for i in range(args.num_output_chunk):
        t = multiprocessing.Process(target=shuffle_chunk, args=(i,))
        workers.append(t)
        t.start()
                
    for worker in workers:
        worker.join()            
                
                
if __name__ == '__main__':
    print('args', args)
    total_size = 0
    for i in range(args.num_input_chunk):
        chunk_id = i+1

        print("[*] Count %s %d" % (args.split, chunk_id))
        h_in = h5py.File(os.path.join(args.input_root, input_format % (args.split, chunk_id)), 'r')
        cur_size = len(h_in['product'])
        total_size += cur_size

        indices = np.arange(cur_size)
        if args.shuffle:
            np.random.shuffle(indices)
        chunk_indices[str(chunk_id)] = list(indices)
        h_in.close()
    
    for i in range(args.num_output_chunk):
        chunk_id = i+1
        per_chunk = total_size // args.num_output_chunk
        # TODO
        # if chunk == args.num_output_chunk-1:
        #     per_chunk += total_size % args.num_output_chunk

        print("[*] Create %s %d" % (args.split, chunk_id))
        h_out = h5py.File(os.path.join(args.output_root, split_format % (args.split, chunk_id)), 'w')
        h_out.create_dataset("img_feat", (per_chunk, 2048), chunks=True, dtype=np.float32)
        h_out.create_dataset("pid", (per_chunk,), chunks=True, dtype="S1000") 
        h_out.create_dataset("label", (per_chunk,), chunks=True, dtype=np.int) 
        h_out.create_dataset("product", (per_chunk,), chunks=True, dtype="S1000")
        h_out.create_dataset("brand", (per_chunk,), chunks=True, dtype="S1000")
        h_out.create_dataset("model", (per_chunk,), chunks=True, dtype="S1000")
        h_out.close()

    split()
    if args.shuffle:
        shuffle()
