import numpy as np
import torch

def entropy(scores):
    return -(scores * np.log(scores + 1e-6)).sum(-1)

def get_energy_scores(logits):
    return -torch.logsumexp(torch.tensor(logits), -1).numpy()

def get_max_logit_scores(logits):
    return -np.max(logits, -1)

def msp(scores):
    return scores.max(-1)

def get_ood_scores(scores, logits, m, others=None):
    if m == 'entropy':
        return entropy(scores)
    if m == 'energy':
        return get_energy_scores(logits)
    if m == 'msp':
        return msp(scores)


def get_ood_scores_for_datasets(data_map, methods=[]):
    out = {}
    for d, v in data_map.items():
        if d not in out.keys():
            out[d] = {}
        for m in methods:
            out[d][m] = get_ood_scores(v['scores'], v['logits'], m, others=v)
    return out

def eval_ood_all_in_one(normal_map, ood_map, methods=[]):
    assert normal_map.keys() == ood_map.keys()

    normal_out = get_ood_scores_for_datasets({
        'normal': normal_map
    })
    ood_out = get_ood_scores_for_datasets(ood_map)

    res = {}
    for d, out in ood_out.items():
        if d not in res.keys():
            res[d] = {}
        for m in methods:
            res[d][m] = eval_ood(normal_out['normal'][m], out[d][m])
    
    return res

from sklearn.metrics import average_precision_score, roc_auc_score

def eval_ood(normal_scores, ood_scores):
    gt = np.zeros(len(normal_scores) + len(ood_scores))
    gt[:len(normal_scores)] = 1

    # larger, normaler
    auroc = roc_auc_score(gt, normal_scores.tolist() + ood_scores.tolist() )

    return {
        'auroc': auroc 
    }

def eval_ood_for_datasets(normal_scores, ood_scores_map):
    out = {}
    for d, scores in ood_scores_map.items():
        out[f'ood_auroc.{d}'] = eval_ood(normal_scores, scores)['auroc']
    
    return out
