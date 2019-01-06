import cPickle
import numpy as np

#y_vocab = cPickle.loads(open("data/y_vocab.cPickle").read())
y_vocab = cPickle.loads(open("data/bmsd_vocab.cPickle").read())
to_cate_ids = {v: map(int, k.split(">")) for k, v in y_vocab.iteritems()} 

def evaluate(y_gt, y_pred, print_result=True):
    correct = {'b': 0, 'm': 0, 's': 0, 'd': 0}
    n = {'b': 0, 'm': 0, 's': 0, 'd': 0}
    miss = {'b': 0, 'm': 0, 's': 0, 'd': 0}

    for i, (gt, pred) in enumerate(zip(y_gt, y_pred)):
        gt = to_cate_ids[gt]
        pred = to_cate_ids[pred]
        for depth, _g, _p in zip(['b', 'm', 's', 'd'], gt, pred):
            if _g == -1:
                continue
            n[depth] += 1
            if _p == _g:
                correct[depth] += 1
            if _p == -1:
                miss[depth] += 1
                
    score = sum([correct[d] / (float(n[d])+1e-6) * w
                 for d, w in zip(['b', 'm', 's', 'd'], [1.0, 1.2, 1.3, 1.4])]) / 4.0

    if print_result:
        print('\n[*] Label')
        for d in ['b', 'm', 's', 'd']:
            print('%s-Accuracy: %.3f(%s/%s), Miss: %.3f(%s/%s)' % (d, correct[d] / (float(n[d])+1e-6), correct[d], n[d], miss[d] / (float(n[d])+1e-6), miss[d], n[d]))
        print('label-Accuracy: %.3f(%s/%s)' % (np.mean(np.array(y_gt) == np.array(y_pred)), np.sum(np.array(y_gt) == np.array(y_pred)), n['b']))
        print('Score: %.4f' % score)

    bmsd_score = [correct[d] / (float(n[d])+1e-6) for d in ['b', 'm', 's', 'd']]
    return score, bmsd_score

def evaluate_bmsd(y_gt, y_pred_b, y_pred_m, y_pred_s, y_pred_d, print_result=True):
    correct = {'b': 0, 'm': 0, 's': 0, 'd': 0}
    n = {'b': 0, 'm': 0, 's': 0, 'd': 0}

    for i in range(len(y_gt)):
        gt = y_gt[i]
        pred_b = y_pred_b[i]
        pred_m = y_pred_m[i]
        pred_s = y_pred_s[i]
        pred_d = y_pred_d[i]
        pred = [pred_b, pred_m, pred_s, pred_d]

        gt = to_cate_ids[gt]
        for depth, _g, _p in zip(['b', 'm', 's', 'd'], gt, pred):
            if _g == -1:
                continue
            n[depth] += 1
            if _p == _g:
                correct[depth] += 1
                
    score = sum([correct[d] / (float(n[d])+1e-6) * w
                 for d, w in zip(['b', 'm', 's', 'd'], [1.0, 1.2, 1.3, 1.4])]) / 4.0

    if print_result:
        print('\n[*] BMSD')
        for d in ['b', 'm', 's', 'd']:
            print('%s-Accuracy: %.3f(%s/%s)' % (d, correct[d] / (float(n[d])+1e-6), correct[d], n[d]))
        print('Score: %.4f' % score)

    bmsd_score = [correct[d] / (float(n[d])+1e-6) for d in ['b', 'm', 's', 'd']]
    return score, bmsd_score

def evaluate_lfirst(y_gt, y_pred, y_pred_b, y_pred_m, y_pred_s, y_pred_d, print_result=True):
    correct = {'b': 0, 'm': 0, 's': 0, 'd': 0}
    n = {'b': 0, 'm': 0, 's': 0, 'd': 0}

    for i in range(len(y_gt)):
        gt = y_gt[i]
        pred = y_pred[i]
        pred_b = y_pred_b[i]
        pred_m = y_pred_m[i]
        pred_s = y_pred_s[i]
        pred_d = y_pred_d[i]
        pred_bmsd = [pred_b, pred_m, pred_s, pred_d]

        gt = to_cate_ids[gt]
        pred = to_cate_ids[pred]
        for depth, _g, _p, _bmsd in zip(['b', 'm', 's', 'd'], gt, pred, pred_bmsd):
            if _g == -1:
                continue
            n[depth] += 1
            if _p == -1:
                _p = _bmsd
            if _p == _g:
                correct[depth] += 1
                
    score = sum([correct[d] / (float(n[d])+1e-6) * w
                 for d, w in zip(['b', 'm', 's', 'd'], [1.0, 1.2, 1.3, 1.4])]) / 4.0

    if print_result:
        print('\n[*] Label First')
        for d in ['b', 'm', 's', 'd']:
            print('%s-Accuracy: %.3f(%s/%s)' % (d, correct[d] / (float(n[d])+1e-6), correct[d], n[d]))
        print('Score: %.4f' % score)

    bmsd_score = [correct[d] / (float(n[d])+1e-6) for d in ['b', 'm', 's', 'd']]
    return score, bmsd_score

