'''
    evaluate performance with the extracted features
'''
import numpy as np
import pickle
import logging
import datetime
import os
import argparse
from pretrain_learner.data_util import load_yaml, load_features_for_pre, load_features_for_sup, load_features_for_sup_shift, load_features_for_pre_shift
from pretrain_learner.util import show_dist
from pathlib import Path
import logging

import warnings
warnings.filterwarnings("ignore", message=r"ndcg_score\sshould\snot\sbe\sused\son\snegative\sy_true\svalues", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"deprecated\sfunction", category=UserWarning)

from utils.ood_base.utils import entropy, get_energy_scores, get_max_logit_scores

env_configs = load_yaml('./env_config.yaml')
feature_dir = env_configs['feature_dir']


def main(configs, logger):
    dataset = configs['dataset']
    class_num = configs['class_num']
    n_neighbors = configs['K']
    score_metric = configs['score']
    method_loss = configs['method_loss']

    if configs.get('eval_shift', False):
        base_info = load_features_for_sup_shift(feature_dir, dataset, configs['base'], configs)
    else:
        base_info = load_features_for_sup(feature_dir, dataset, configs['base'], configs)


    base_val_features =  list(base_info['val_info']['feats'].values())[0]
    base_test_features =  list(base_info['test_info']['feats'].values())[0]
    base_train_features = None
    if base_info['train_info'] is not None:
        base_train_features =  list(base_info['train_info']['feats'].values())[0]

    base_unlabeled_features = None
    if base_info['unlabel_info'] is not None:
        base_unlabeled_features = list(base_info['unlabel_info']['feats'].values())[0]

    _info = base_info['test_info']
    logger.info('avg confience: {:.2%}; acc: {:.2%}({}/{})'.format(_info['msp'].max(1).mean(), sum(_info['corrects'])/len(_info['corrects']), sum(_info['corrects']), len(_info['corrects'])))


    val_idx = base_info['val_idx']
    test_idx = base_info['test_idx']
    train_idx = base_info['train_idx']
    test_size = len(base_test_features)
    val_size = len(val_idx)
    if train_idx is not None:
        train_size = len(train_idx)
        print('train_size', train_size)



    sup_infos = {}
    sup_val_features = []
    sup_test_features = []
    sup_train_features = []
    sup_unlabeled_features = []
    for sup_key in configs['other_sups']:

        if configs.get('eval_shift', False):
            sup_infos[sup_key] = load_features_for_sup_shift(feature_dir, dataset, sup_key, configs)
        else:
            sup_infos[sup_key] = load_features_for_sup(feature_dir, dataset, sup_key, configs)


        # print('sup_key', sup_key, )
        assert np.all(sup_infos[sup_key]['val_idx'] == val_idx)
        sup_val_features.append( list(sup_infos[sup_key]['val_info']['feats'].values())[0] )
        sup_test_features.append( list(sup_infos[sup_key]['test_info']['feats'].values())[0] )
        sup_train_features.append( list(sup_infos[sup_key]['train_info']['feats'].values())[0] )

        _info = sup_infos[sup_key]['test_info']
        logger.info('sup key: {}'.format(sup_key))
        logger.info('avg confience: {:.2%}; acc: {:.2%}({}/{})'.format(_info['msp'].max(1).mean(), sum(_info['corrects'])/len(_info['corrects']), sum(_info['corrects']), len(_info['corrects'])))

        if sup_infos[sup_key]['unlabel_info'] is not None:
            sup_unlabeled_features.append( list(sup_infos[sup_key]['unlabel_info']['feats'].values())[0] )



    pre_infos = {}
    pre_val_features = []
    pre_test_features = []
    pre_train_features = []
    pre_unlabeled_features = []
    for pre_key in configs['pretrain']:

        if configs.get('eval_shift', False):
            pre_infos[pre_key] = load_features_for_pre_shift(feature_dir, dataset, pre_key, {
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx,
                **configs
            })
        else:
            pre_infos[pre_key] = load_features_for_pre(feature_dir, dataset, pre_key, {
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx,
                **configs
            })


        # the candidate pool size should be same
        assert len(list(pre_infos[pre_key]['val_info']['feats'].values())[0]) ==  len(base_val_features)
        assert len(list(pre_infos[pre_key]['test_info']['feats'].values())[0]) ==  len(base_test_features)

        if base_train_features is not None and pre_infos[pre_key]['train_info'] is not None:
            assert len(list(pre_infos[pre_key]['train_info']['feats'].values())[0]) ==  len(base_train_features)


        pre_val_features.append( list(pre_infos[pre_key]['val_info']['feats'].values())[0] )
        pre_test_features.append( list(pre_infos[pre_key]['test_info']['feats'].values())[0] )

        if pre_infos[pre_key]['train_info'] is not None:
            pre_train_features.append( list(pre_infos[pre_key]['train_info']['feats'].values())[0] )


        if pre_infos[pre_key]['unlabel_info'] is not None:
            pre_unlabeled_features.append( list(pre_infos[pre_key]['unlabel_info']['feats'].values())[0] )


    from pretrain_learner.util import get_def_scores

    if configs.get('neighbor_source') == 'unlabeled' and len(pre_unlabeled_features) > 0:
        
        sample_num = len(base_unlabeled_features)
        sub_size = min(configs.get('neighbor_pool_size', sample_num), sample_num)
        sub_seed = configs.get('split_seed', 0)

        _indexs = np.arange(sample_num)
        np.random.seed(sub_seed)
        np.random.shuffle(_indexs)
        sub_indexs = _indexs[:sub_size]

        base_neighbor_features = base_unlabeled_features[sub_indexs]
        sup_neighbor_features = [sup_feats[sub_indexs] for sup_feats in sup_unlabeled_features]
        pre_neighbor_features = [pre_feats[sub_indexs] for pre_feats in pre_unlabeled_features]
        
        logger.info('using unlabeled data as neighborhood')

    elif configs.get('neighbor_source') == 'train' and len(pre_train_features) > 0:
        
        train_sample_num = len(base_train_features)
        train_sub_size = min(configs.get('neighbor_pool_size', train_sample_num), train_sample_num)
        train_sub_seed = configs.get('split_seed', 0)

        assert train_sample_num == train_size

        print('train_sample_num', train_sample_num)
        print('train_sub_size', train_sub_size)

        train_indexs = np.arange(train_sample_num)
        np.random.seed(train_sub_seed)
        np.random.shuffle(train_indexs)
        train_sub_indexs = train_indexs[:train_sub_size]

        base_neighbor_features = base_train_features[train_sub_indexs]
        sup_neighbor_features = [sup_feats[train_sub_indexs] for sup_feats in sup_train_features]
        pre_neighbor_features = [pre_feats[train_sub_indexs] for pre_feats in pre_train_features]
        
        # pre_neighbor_features = pre_train_features
    else:
        base_neighbor_features = base_val_features
        sup_neighbor_features = sup_val_features
        pre_neighbor_features = pre_val_features

    logger.info('base_neighbor_features.shape: {}'.format(base_neighbor_features.shape))
    logger.info('pre_neighbor_features[0].shape: {}'.format(pre_neighbor_features[0].shape) )


    sup_sim_scores, sup_pariwise_sim_scores = get_def_scores(n_neighbors,
        [base_test_features, *sup_test_features],
        [base_neighbor_features, *sup_neighbor_features],
        search_mask=np.ones(test_size, dtype=bool),
        metric=score_metric,
        return_list=True,
    )

    val_sup_sim_scores, val_sup_pariwise_sim_scores = get_def_scores(n_neighbors,
        [base_val_features, *sup_val_features],
        [base_neighbor_features, *sup_neighbor_features],
        search_mask=np.ones(val_size, dtype=bool),
        metric=score_metric,
        return_list=True,
    )


    pre_sim_scores, pre_pariwise_sim_scores = get_def_scores(n_neighbors,
        [base_test_features, *pre_test_features],
        [base_neighbor_features, *pre_neighbor_features],
        search_mask=np.ones(test_size, dtype=bool),
        metric=score_metric,
        return_list=True,
        # verbose=True
    )

    val_pre_sim_scores, val_pre_pariwise_sim_scores = get_def_scores(n_neighbors,
        [base_val_features, *pre_val_features],
        [base_neighbor_features, *pre_neighbor_features],
        search_mask=np.ones(val_size, dtype=bool),
        metric=score_metric,
        return_list=True,
    )


    #################### BASE INFORMATION ########################
    res = {}
    # acc
    test_corrects = base_info['test_info']['corrects']
    test_logits = base_info['test_info']['pred_logits']
    test_gtlabels = base_info['test_info']['gt_labels']
    test_scores = base_info['test_info']['msp']
    test_msp = test_scores.max(1)
    test_labels = np.argmax(test_scores, axis=1)


    val_corrects = base_info['val_info']['corrects']
    val_logits = base_info['val_info']['pred_logits']
    val_gtlabels = base_info['val_info']['gt_labels']
    val_scores = base_info['val_info']['msp']
    val_msp = val_scores.max(1)
    val_labels = np.argmax(val_scores, axis=1)


    logger.info('avg confience: {:.2%}; acc: {:.2%}({}/{})'.format(test_msp.mean(), sum(test_corrects)/len(test_corrects), sum(test_corrects), len(test_corrects)))


    ## base performance
    from utils.eval import eval_all_in_one, eval_all_in_one_w_conf, get_cal_res_with_scores, get_mis_detect_res, get_temperature_scores, get_temperature_scores_with_aux, select_aux_features



    base_res = eval_all_in_one(test_scores, test_gtlabels, test_corrects, config=configs)
    logger.info(base_res['mis_detect'])



    res['msp'] = {**base_res['mis_detect']}



    ts_out = get_temperature_scores(test_logits, val_logits, val_gtlabels)
    val_ts_out = get_temperature_scores(val_logits, val_logits, val_gtlabels)


    ts_res = eval_all_in_one(ts_out['scores'], test_gtlabels, test_corrects, config=configs)
    logger.info('Temperature Scaling:')
    logger.info('avg confience: {:.2%}; acc: {:.2%}({}/{})'.format(ts_out['msp'].mean(), sum(test_corrects)/len(test_corrects), sum(test_corrects), len(test_corrects)))
    logger.info(ts_res['mis_detect'])
    res['temp.scaling'] = {**ts_res['mis_detect']}


    # entropy
    entropy_scores = -entropy(test_scores)

    res['entropy'] = {**get_mis_detect_res(entropy_scores, test_corrects)}
    logger.info('entropy res:')
    logger.info(res['entropy'])

    # energy score
    energy_scores = -get_energy_scores(test_logits)
    res['energy'] = {**get_mis_detect_res(energy_scores, test_corrects)}
    logger.info('energy res:')
    logger.info(res['energy'])
    
    # max logits
    maxlogit_scores = -get_max_logit_scores(test_logits)

    res['maxlogit'] = {**get_mis_detect_res(maxlogit_scores, test_corrects)}
    logger.info('maxlogit res:')
    logger.info(res['maxlogit'])


    logger.info('========Our method========')

    mode = 'pre'

    pack = {
        'sup_pariwise_sim_scores': sup_pariwise_sim_scores,
        'val_sup_pariwise_sim_scores': val_sup_pariwise_sim_scores,
        'pre_pariwise_sim_scores': pre_pariwise_sim_scores,
        'val_pre_pariwise_sim_scores': val_pre_pariwise_sim_scores,
    }
    val_task_features_arr, task_features_arr = select_aux_features(pack, mode, configs)

    aux_ts_out = get_temperature_scores_with_aux(test_logits, val_logits, val_gtlabels, task_features_arr, val_task_features_arr, loss=method_loss, others={'temp': ts_out['temp']})
    val_aux_ts_out = get_temperature_scores_with_aux(val_logits, val_logits, val_gtlabels, val_task_features_arr, val_task_features_arr, loss=method_loss, others={'temp': ts_out['temp']})

    logger.info('our method performance on validation data:')
    val_aux_ts_res = eval_all_in_one(val_aux_ts_out['scores'], val_gtlabels, val_corrects)
    logger.info(val_aux_ts_res['mis_detect'])
    logger.info(val_aux_ts_res['calibration'])

    print("aux_ts_out['scores']", aux_ts_out['scores'].shape)

    aux_ts_res = eval_all_in_one(aux_ts_out['scores'], test_gtlabels, test_corrects, config=configs)
    logger.info('temperature scaling with ndcg')
    logger.info('avg confience: {:.2%}; acc: {:.2%}({}/{})'.format(aux_ts_out['msp'].mean(), sum(test_corrects)/len(test_corrects), sum(test_corrects), len(test_corrects)))
    logger.info(aux_ts_res['mis_detect'])


    res['temp.scaling.ndcg_all'] = {**aux_ts_res['mis_detect']}

    logger.info('========End Our method========')


    import pandas as pd
    import csv

    today = datetime.date.today()
    res_file = 'res/{}/{}/{}.csv'.format(configs.get('exp_tag', ''), today, configs['exp_name'])
    os.makedirs(os.path.dirname(res_file), exist_ok=True)

    logger.info(res)

    fields = ['method', *list(res['msp'].keys()) ]
    print('fields', fields)
    def mergedict(a,b):
        a.update(b)
        return a

    with open(res_file, "w") as f:
        w = csv.DictWriter( f, fields )
        w.writeheader()
        for k,d in res.items():
            print(k, d)
            w.writerow({'method': k, **d})


    logger.info('exported at {}!'.format(res_file))


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, default=None, help="Path for config yaml")

    args = parser.parse_args()
    print('args', args)


    configs = load_yaml(args.config_path)
    configs['exp_name'] = os.path.basename(args.config_path)

    main(configs)