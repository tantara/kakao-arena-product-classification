# -*- coding: utf-8 -*-

import re
re_sc = re.compile('[\!@#$%\^&\*\(\)\-=_\[\]\{\}\.,/\?~\+\'"|]')

import sys
sys.setdefaultencoding('utf8')

class Tokenizer(object):
    def __init__(self, tagger):
        if tagger == 'okt':
            from konlpy.tag import Okt
            self.tagger = Okt()
            self.filters = ['Noun', 'Foreign', 'Alpha']
        elif tagger == 'mecab':
            from konlpy.tag import Mecab
            self.tagger = Mecab()
            self.filters = ['NNG', 'NNP', 'SL', 'SH', 'SN']
        elif tagger == 'whitespace':
            self.tagger = None
        else:
            raise

    def tokenize(self, txt):
        if self.tagger == None:
            words = [w.strip() for w in txt if not w.strip().isdigit()]
        else:
            words = [w.strip() for w in txt]
        words = " ".join(words)
        words = words.encode('utf-8')

        if self.tagger != None:
            pos = self.tagger.pos(words, norm=True, stem=True)
            words = " ".join([word for (word, tag) in filter(lambda (word, tag): tag in self.filters, pos)]) 
        return words


def normalize(txt, col_type):
    txt = txt.encode('utf-8')
    txt = txt.lower()
    txt = re_sc.sub(' ', txt).strip()
    return txt.split()
