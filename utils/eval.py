import numpy as np
from pretrain_learner.data_util import eval_res
from pretrain_learner.cal_util import ECE, MCE, TemperatureScaling, evaluate, TemperatureScalingWithOthers
from .metrics import AdaptiveECELoss


def eval_all_in_one(test_scores, test_gtlabels, test_corrects, ood_info = None, normalize=True, config=None):
    # calibration + misclassification

    # test_scores = softmax(test_logits)
    test_msp = test_scores.max(1)

    mis_res = eval_res(test_msp, test_corrects)
    if config is None:
        eval_bins = 15
    else: 
        eval_bins = config.get('cal_eval_bins', 15)
    _cal_res = evaluate(test_scores, np.array(test_gtlabels), verbose=True, normalize=normalize, bins=eval_bins)

    cal_res = {
        'Test Error': _cal_res[0],
        'ECE': _cal_res[1],
        'MCE': _cal_res[2],
        'NLL': _cal_res[3],
        'Brier score': _cal_res[4],
    }


    return {
        'mis_detect': mis_res,
        'calibration': cal_res,
    }

def eval_all_in_one_w_conf(test_confs, test_gtlabels, test_corrects, test_preds):
    # calibration + misclassification

    # test_scores = softmax(test_logits)
    cal_res = get_cal_res_with_confs(test_confs, test_gtlabels, test_corrects, test_preds)
    mis_res = get_mis_detect_res(test_confs, test_corrects)

    return {
        'mis_detect': mis_res,
        'calibration': cal_res
    }

def get_cal_res_with_scores(test_scores, test_gtlabels, test_corrects):
    _cal_res = evaluate(test_scores, np.array(test_gtlabels), verbose=True, normalize=True)


    return {
        'Test Error': _cal_res[0],
        'ECE': _cal_res[1],
        'MCE': _cal_res[2],
        'NLL': _cal_res[3],
        'Brier score': _cal_res[4],
    }

import sklearn.metrics as metrics
import torch

def get_cal_res_with_confs(test_confs, test_gtlabels, test_corrects, test_preds):
    bins = 15

    # test_error = (1 - sum(test_corrects) / len(test_corrects)) * 100

    accuracy = metrics.accuracy_score(test_gtlabels, test_preds) * 100
    test_error = 100 - accuracy

    eceloss = ECE(test_confs, test_preds, test_gtlabels, bin_size=1/bins)
    mceloss = MCE(test_confs, test_preds, test_gtlabels, bin_size=1/bins)

    adaece_criterion = AdaptiveECELoss(bins)
    p_adaece = adaece_criterion(None, torch.tensor(np.array(test_gtlabels).astype(int)), confidences=torch.tensor(np.array(test_confs).astype(float)), predictions=torch.tensor(np.array(test_preds).astype(int))).item()

    # toplabel_ece = eval_more_ece(test_confs, test_gtlabels, test_preds)

    return {
        'Test Error': test_error,
        'ECE': eceloss,
        'MCE': mceloss,
        'Adaptive ECE': p_adaece,
        # 'TopLabel ECE': toplabel_ece
    }

def get_mis_detect_res(test_msp, test_corrects):
    mis_res = eval_res(test_msp, test_corrects)

    return mis_res


def get_temperature_scores_with_aux(test_logits, val_logits, val_gtlabels, test_aux, val_aux, loss='log_loss', others=None):

    fit_dim = val_aux.shape[1]

    # use all parameters and temperature scaling once
    if loss == 'log_loss':
        tscaler_aux = TemperatureScalingWithOthers(maxiter=300, solver='SLSQP', temp=np.ones(fit_dim + 1))

    x0 = None
    if others is not None and 'temp' in others:
        x0 = np.zeros(val_aux.shape[1]+1) / 2
        x0[0] = 1 / others['temp']

    fit_res = tscaler_aux.fit(np.array(val_logits), np.array(val_gtlabels), val_aux, kwargs={
    })
    

    aux_ts_scores = tscaler_aux.predict(np.array(test_logits), test_aux)
    aux_ts_logits = tscaler_aux.predict(np.array(test_logits), test_aux, return_logit=True)
    aux_ts_msp = aux_ts_scores.max(1)

    return {
        'scores': aux_ts_scores,
        'logits': aux_ts_logits,
        'msp': aux_ts_msp,
        'learner': tscaler_aux
    }

def get_temperature_scores(test_logits, val_logits, val_gtlabels):
    tscaler = TemperatureScaling(maxiter=300, solver='SLSQP')
    fit_res = tscaler.fit(np.array(val_logits), np.array(val_gtlabels))
    ts_scores = tscaler.predict(np.array(test_logits))
    ts_logits = tscaler.predict(np.array(test_logits), return_logit=True)
    ts_msp = ts_scores.max(1)

    return {
        'scores': ts_scores,
        'logits': ts_logits,
        'msp': ts_msp,
        'learner': tscaler,
        'temp': tscaler.temp
    }

import copy
def select_aux_features(pack, mode, configs={}):
    val_sup_pariwise_sim_scores = pack['val_sup_pariwise_sim_scores']
    sup_pariwise_sim_scores = pack['sup_pariwise_sim_scores']
    val_pre_pariwise_sim_scores = pack['val_pre_pariwise_sim_scores']
    pre_pariwise_sim_scores = pack['pre_pariwise_sim_scores']

    
    
    use_raw_ndcg_scores = False
    if 'ens' in configs.get('method_loss', 'log_loss'):
        use_raw_ndcg_scores = True
    if configs.get('use_raw', False):
        use_raw_ndcg_scores = True
        

    if mode == 'pre':


        val_task_features_arr = [
            val_pre_pariwise_sim_scores.mean(1, keepdims=True)
        ]
        task_features_arr = [
            pre_pariwise_sim_scores.mean(1, keepdims=True)
        ]

        
        val_task_features_arr = np.concatenate(val_task_features_arr, 1)
        task_features_arr = np.concatenate(task_features_arr, 1)


    # parameterized temperature scaling with original ndcg scores
    if use_raw_ndcg_scores:
        val_task_features_arr = val_pre_pariwise_sim_scores
        task_features_arr = pre_pariwise_sim_scores

    
    return val_task_features_arr, task_features_arr
