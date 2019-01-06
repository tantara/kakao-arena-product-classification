# -*- coding: utf-8 -*-
# Copyright 2017 Kakao, Recommendation Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import traceback
from multiprocessing import Pool

import tqdm
import fire
import h5py
import numpy as np
import six
from six.moves import cPickle

from misc import Option
opt = Option('./config.json')

class Reader(object):
    def __init__(self, div):
        self.div = div

    def get_bmsd(self, h, i):
        b = h['bcateid'][i]
        m = h['mcateid'][i]
        s = h['scateid'][i]
        d = h['dcateid'][i]
        return '%s>%s>%s>%s' % (b, m, s, d)

    def get_y_vocab(self, data_path, vocab_type):
        y_vocab = {}
        h = h5py.File(data_path, 'r')[self.div]
        sz = h['pid'].shape[0]
        for i in tqdm.tqdm(range(sz), mininterval=1):
            if vocab_type == "bmsd":
                class_name = self.get_bmsd(h, i)
            else:
                print('unsupported vocab type: %s' % vocab_type)
                raise

            if class_name not in y_vocab:
                y_vocab[class_name] = len(y_vocab)
        return y_vocab


def build_y_vocab(data):
    try:
        data_path, div, vocab_type = data
        reader = Reader(div)
        y_vocab = reader.get_y_vocab(data_path, vocab_type)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))
    return y_vocab


class Data:
    y_vocab_path = {'bmsd': 'bmsd_vocab.cPickle'}

    def __init__(self):
        pass

    def build_y_vocab(self, vocab_type):
        pool = Pool(opt.num_workers)
        try:
            rets = pool.map_async(build_y_vocab,
                                  [(data_path, 'train', vocab_type)
                                   for data_path in opt.train_data_list]).get(99999999)
            pool.close()
            pool.join()
            y_vocab = set()
            for _y_vocab in rets:
                for k in six.iterkeys(_y_vocab):
                    y_vocab.add(k)
            self.y_vocab = {y: idx for idx, y in enumerate(y_vocab)}
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        print('size of %s vocab: %d' % (vocab_type, len(self.y_vocab)))
        cPickle.dump(self.y_vocab, open(self.y_vocab_path[vocab_type], 'wb'), 2)

    def build_bmsd_vocab(self):
        self.build_y_vocab(vocab_type="bmsd")


if __name__ == '__main__':
    data = Data()
    fire.Fire({'bmsd': data.build_bmsd_vocab})
