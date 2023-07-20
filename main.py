from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
import os
import eval

def build_config(raw_config):
    # adjust config a bit
    config = {}
    cfg = raw_config['run']
    config['debug'] = raw_config['debug']

    config['dataset'] = cfg.get('dataset', {}).get('full_name') or cfg.get('dataset', {}).get('name')
    config['class_num'] = cfg.get('dataset', {}).get('class_num', -1)

    # base model
    config['base'] = cfg.get('base_model', {}).get('name', '')
    config['base_type'] = cfg.get('base_model', {}).get('type', '')

    # pretrain group setting
    config['pretrain'] = cfg.get('pretrain', {}).get('models', [])
    config['pre_group_name'] = cfg.get('pretrain', {}).get('name', '')

 
    config['K'] = cfg.get('algo', {}).get('K', -1)
    config['K_ratio'] = cfg.get('algo', {}).get('K_ratio', -1)
    # distance measure choice
    config['score'] = cfg.get('algo', {}).get('score', '')


    # neighborhood pool source: valid set / train set
    config['neighbor_source'] = cfg.get('set', {}).get('neighbor_source', '')
    # the max neighbor pool size, min(valid_size/train_size, neighbor_pool_size)
    config['neighbor_pool_size'] = cfg.get('set', {}).get('neighbor_pool_size', '')

    # the configs in set auto assigned
    for key, value in cfg.get('set').items():
        config[key] = value

    # exp_name
    config['exp_name'] = '{}_model={}_pre={}_source={}_nbhsize={}'.format(
            config['dataset'], 
            config['base'],
            config['pre_group_name'],
            config['neighbor_source'],
            config['neighbor_pool_size'],
        )
    
    if config['K_ratio'] != -1:
        config['exp_name'] += '_Kratio={}'.format(config['K_ratio'])
        config['K'] = int(config['neighbor_pool_size'] * config['K_ratio'])
    else:
        config['exp_name'] += '_K={}'.format(config['K'])

    
    if config['score'] != 'ndcg_score':
        config['exp_name'] += '_score={}'.format(config['score'])

    if config['pre_group_name'] == 'single':
        # models only contain one
        assert len(config['pretrain']) == 1
        config['exp_name'] += '_premodname={}'.format(
            config['pretrain'][0],
        )

    if config.get('split_seed', None) is not None:
        config['exp_name'] += '_splitseed={}'.format(config.get('split_seed'))
        
    if config.get('other'):
        config['exp_name'] += '_other={}'.format(
            config['other'],
        )
    if config.get('method_loss'):
        config['exp_name'] += '_methodloss={}'.format(
            config['method_loss'],
        )

    if config.get('debug', False):
        config['exp_name'] += '_debug=True'

    return config



logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:

    _raw_config = OmegaConf.to_yaml(cfg, resolve=True)
    _conf = OmegaConf.create(_raw_config)
    main_conf = build_config(_conf)

    hydra_context = HydraConfig.get()
    
    logger.info('begin exp:')
    print('log saved at {}'.format(hydra_context.runtime.output_dir))
    # log config
    logger.info('config:\n' + OmegaConf.to_yaml(main_conf, resolve=True) )


    eval.main(main_conf, logger)

    print('log saved at {}'.format(hydra_context.runtime.output_dir))
    logger.info('end exp.')


if __name__ == "__main__":
    run()
