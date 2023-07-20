import copy
import os
import torch
from pathlib import Path

# from train_base.models.resnet import resnet152, resnet18, resnet34
# from train_base.models.resnet_cifar10 import resnet152, resnet18, resnet34
from torchvision.models import resnet18, resnet34, resnet152,resnet50
# from train_base.models.resnet_dp_1 import resnet18_dp
import torch.nn as nn
import torchvision.transforms as trn
import numpy as np
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler
from types import SimpleNamespace
import torchvision
from tqdm import tqdm

def entropy(scores):
    return -(scores * np.log(scores + 1e-6)).sum(-1)

def retrieve_data(dataset_name, dataset, indices=[]):
    if dataset_name == 'cifar10' or dataset_name == 'cifar100':
        if getattr(dataset, 'data'):
            return dataset.data, dataset.targets
        else:
            if len(indices) <= 0:
                imgs = np.array([ np.asarray(dataset.loader(sample[0])) for sample in dataset.samples])
                targets = dataset.targets
            else:
                imgs = np.array([ np.asarray(dataset.loader(sample[0])) for i, sample in enumerate(tqdm(dataset.samples)) if i in indices])
                targets = np.array(dataset.targets)[indices].tolist()

            return imgs, targets
    if dataset_name == 'svhn' or dataset_name == 'stl10':
        return dataset.data, dataset.labels
    if dataset_name in ['mnist', 'fmnist', 'mnist_wor']:
        return dataset.data, dataset.targets
    if dataset_name in ['cinic10', 'imagenet', 'tinyimagenet']:
        if len(indices) <= 0:
            imgs = np.array([ np.asarray(dataset.loader(sample[0])) for sample in dataset.samples])
            targets = dataset.targets
        else:
            imgs = np.array([ np.asarray(dataset.loader(sample[0])) for i, sample in enumerate(tqdm(dataset.samples)) if i in indices])
            targets = np.array(dataset.targets)[indices].tolist()

        return imgs, targets

def get_model_name(config_path):
    if os.path.exists(config_path) and '.yaml' in config_path:
        from confidnet.models import get_model
        from confidnet.utils.misc import load_yaml
        from confidnet.loaders import get_loader

        config_args = load_yaml(config_path)
        model_name = config_args["model"]["name"]
    elif 'imagenet' in config_path:
        dataset, model_name = config_path.split('_')
    else:
        file_name = Path(config_path).name
        prefix = file_name.split('baseline')[0].split('_')
        dataset, model_name = prefix[0], '_'.join([ _ for _ in prefix[1:] if _ != ''])

    return model_name

def get_subdataset_stat(loader, sub_classes):
    mean = 0.
    std = 0.
    total = 0
    sub_classes = torch.tensor(sub_classes)
    all_images = []
    for images, targets in loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        mask = torch.tensor([t in sub_classes for t in targets])
        # images = images[mask]
        if sum(mask) > 0 :
            all_images.append(images[mask])


    all_images = torch.cat(all_images, 0)
    mean = all_images.mean((0, 2, 3)).tolist()
    std = all_images.std((0, 2, 3)).tolist()

    return mean, std

import yaml
def process_task_config(config_path):
    with open(config_path, "r") as f:
        config_args = yaml.load(f, Loader=yaml.SafeLoader)

    return config_args


import yaml
def load_yaml(p):
    with open(p, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded



def get_data_from_ImageFolder(folder):

    samples = []
    targets = []
    print('folder loader', len(folder))
    for i in range(len(folder)):
        s, t = folder[i]
        samples.append(s)

    print('samples', len(samples))
    return samples


def get_ood_data(dataset_name):

    if dataset_name in ['svhn']:
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True)

        test_ood_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True)
        test_ood_dataset.labels = np.ones(len(test_ood_dataset))

    elif dataset_name in ['imagenet_resize', 'lsun_resize', 'imagenet_pil', 'lsun_pil', 'tiny', 'constant', 'noise', 'white', 'imagenet-o']:
        
        dir_map = {
            'imagenet_resize': "~/data/Imagenet_resize",
            'lsun_resize': "~/data/LSUN_resize",
            'imagenet_pil': "~/data/Imagenet_pil",
            'lsun_pil': "~/data/LSUN_pil",
            'tiny': "~/data/tiny",
            'constant': "~/data/constant",
            'noise': "~/data/noise",
            'white': "~/data/white",
            'imagenet-o': "~/data/imagenet-o/data",
        }
        test_ood_dataset = torchvision.datasets.ImageFolder(dir_map[dataset_name])

        if not hasattr(test_ood_dataset, 'labels'):
            setattr(test_ood_dataset, 'labels', [])
            setattr(test_ood_dataset, 'data', [])

        data = get_data_from_ImageFolder(test_ood_dataset)
        test_ood_dataset.labels = np.ones(len(data))
        test_ood_dataset.data = data

    return test_ood_dataset.data, test_ood_dataset.labels

class PartialFolder(torch.utils.data.Dataset):
    def __init__(self, parent_ds, perm, length):
        self.parent_ds = parent_ds
        self.perm = perm
        self.length = length
        super(PartialFolder, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.parent_ds[self.perm[i]]


def validation_split_folder(dataset, val_share=0.1, seed=42):
    """
       Split a (training and vaidation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).

       inputs:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation data (should be 0<val_share<1, default: 0.1)
       returns: input dataset split into test_ds, val_ds

    """
    num_train = int(len(dataset) * (1 - val_share))
    num_val = len(dataset) - num_train

    perm = np.asarray(range(len(dataset)))
    # np.random.seed(0)
    np.random.seed(seed)
    np.random.shuffle(perm)

    train_perm, val_perm = perm[:num_train], perm[num_train:]

    return PartialFolder(dataset, train_perm, num_train), PartialFolder(dataset, val_perm, num_val)

def get_save_pattern(args, model_name, dataset):
    if 'subclass' in args.config_path:
        _path_parts = args.config_path.split('/')
        subclass_key = _path_parts[['subclass' in p for p in _path_parts].index(True)]
        seed_key = _path_parts[['seed' in p for p in _path_parts].index(True)]
        save_model_path = f'saved/sub_models/{dataset}_{subclass_key}/{model_name}/{seed_key}'
        # save_img_path = f'./fig/{dataset}_{subclass_key}/{model_name}/{seed_key}'
    else:
        save_model_path = f'saved/sub_models/{dataset}/{model_name}'


    _model_name = model_name
    if vars(args).get('ssl') and args.ssl is not None:
        _model_name = f'{model_name}_{args.ssl}'

    if 'subclass' in args.config_path:
        _path_parts = args.config_path.split('/')
        subclass_key = _path_parts[['subclass' in p for p in _path_parts].index(True)]
        seed_key = _path_parts[['seed' in p for p in _path_parts].index(True)]
        pattern = f'{dataset}_{subclass_key}/{_model_name}/{seed_key}'
    else:
        pattern = f'{dataset}/{_model_name}'

    return pattern

def get_base_model_feature_extractor(model):
    features_extractor = copy.deepcopy(model)
    features_extractor.eval()
    if hasattr(features_extractor, 'fc2'):
        features_extractor.fc2 = nn.Identity()
    elif hasattr(features_extractor, 'classifier'):
        features_extractor.classifier = nn.Identity()
    else:
        features_extractor.fc = nn.Identity()
    
    return features_extractor

import clip
def get_pretrained_preprocess_func(model_name, dataset):
    # inverse to the original images
    # use pretrained transformation
    preprocess = None

    if dataset == 'cifar100' or dataset == 'cifar10':
        # mean = (0.4914, 0.4822, 0.4465) 
        # std = (0.2023, 0.1994, 0.2010)

        inv_normalize = trn.Normalize(
            mean=(-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010),
            std=(1/0.2023, 1/0.1994, 1/0.2010)
        )
    elif dataset == 'imagenet':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        inv_normalize = trn.Normalize(
            mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
            std=(1/0.229, 1/0.224, 1/0.225)
        )
    # elif dataset == 'cifar10':
        # mean = (0.4914, 0.4822, 0.4465) 
        # std = (0.2023, 0.1994, 0.2010)

    # print('model_name', model_name)
    if model_name == 'clip' or 'clip' in model_name:
        if model_name == 'clip':
            _, _preprocess = clip.load('RN50')
        else:
            version = model_name.split('_')[-1]
            mapping = {
                'resnet50': 'RN50'
            }
            _, _preprocess = clip.load(mapping[version])

            #     train_transform = trn.Compose([trn.RandomResizedCrop(224), trn.RandomHorizontalFlip(),
            #                                 trn.ToTensor(), trn.Normalize(mean, std)])
            #     test_transform = trn.Compose([trn.Resize(224), trn.ToTensor(), trn.Normalize(mean, std)])
        # remove to_tensor

        preprocess = trn.Compose([
            inv_normalize,
            trn.ToPILImage(),
            *(_preprocess.transforms)
        ])
        # print('*(_preprocess.transforms)', *(_preprocess.transforms))
        # print('using clip transform')
    # elif model_name == 'mocov2' or model_name == 'sup_pretrain_imagenet':
    elif model_name == 'mocov2' or 'suppre' in model_name:
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        preprocess = trn.Compose([
            inv_normalize,
            trn.ToPILImage(),
            trn.Resize(size=224),
            trn.CenterCrop(size=(224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=mean, std=std)
        ])
        # print('using mocov2 transform')

    elif 'simclrv2' in model_name:
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        preprocess = trn.Compose([
            inv_normalize,
            trn.ToPILImage(),
            trn.Resize(size=256),
            trn.CenterCrop(size=(224, 224)),
            trn.ToTensor(),
            # trn.Normalize(mean=mean, std=std) # add norm
        ])
    elif 'simclrv1' in model_name:
        preprocess = trn.Compose([
            inv_normalize,
            trn.ToPILImage(),
            trn.Resize(size=256),
            trn.CenterCrop(size=(256)),
            trn.ToTensor(),
            # trn.Normalize(mean=mean, std=std) # add norm
        ])

    return preprocess

def get_orig_imgs_from_samples(test_samples, dataset):
    
    
    if dataset == 'imagenet':

        # mean=[0.485, 0.456, 0.406]
        # std=[0.229, 0.224, 0.225]

        inv_normalize = trn.Normalize(
            mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
            std=(1/0.229, 1/0.224, 1/0.225)
        )
        
    all_images = []
    for item in test_samples:
        image, sample = item
        if inv_normalize is not None:
            all_images += [ inv_normalize(image).unsqueeze(0) ]
        else:
            all_images += [ image.unsqueeze(0)]

    all_images = torch.cat(all_images, 0).cpu()

    # raw_all_images = []
    # for img in all_images:
    #     im = trn.ToPILImage()(img).convert('RGB')
    #     raw_all_images += [im]

    raw_all_images = all_images
    
    return raw_all_images

def get_orig_imgs(test_loader, dataset):
    if dataset == 'cifar100' or dataset == 'cifar10' or 'cifar10-c' in dataset or 'cifar100-c' in dataset:
        # mean = (0.4914, 0.4822, 0.4465) 
        # std = (0.2023, 0.1994, 0.2010)

        inv_normalize = trn.Normalize(
            mean=(-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010),
            std=(1/0.2023, 1/0.1994, 1/0.2010)
        )
    elif dataset == 'imagenet':

        # mean=[0.485, 0.456, 0.406]
        # std=[0.229, 0.224, 0.225]

        inv_normalize = trn.Normalize(
            mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
            std=(1/0.229, 1/0.224, 1/0.225)
        )
    elif dataset in ['stanford_cars', 'cub200', 'food']:
        inv_normalize = trn.Normalize(
            mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
            std=(1/0.229, 1/0.224, 1/0.225)
        )
    elif dataset in ['svhn']:
        inv_normalize = trn.Normalize(
            mean=(-1, -1, -1),
            std=(2, 2, 2)
        )

    all_images = []
    for images, targets in tqdm(test_loader):
        if inv_normalize is not None:
            all_images += [ inv_normalize(img).unsqueeze(0) for img in images]
        else:
            all_images += [ img.unsqueeze(0) for img in images]

    all_images = torch.cat(all_images, 0).cpu()

    # raw_all_images = []
    # for img in all_images:
    #     im = trn.ToPILImage()(img).convert('RGB')
    #     raw_all_images += [im]

    raw_all_images = all_images

    return raw_all_images

from pathlib import Path
import pickle

def load_ood_features(dir_name, dataset, key, ood_datasets, configs=None, mode='sup'):
    ood_infos = {}
    if mode == 'sup' or (dataset == 'imagenet' and 'sup_' in key):
        p = Path(f'{dir_name}/{dataset}/{key}/ood')
        for ood_name in ood_datasets:
            if os.path.exists(p / ood_name / 'test.pt'):
                ood_infos[ood_name] = pickle.load(open(p / ood_name / 'test.pt', 'rb'))
            else:
                print('{} not exist'.format(p / ood_name / 'test.pt'))
    elif mode == 'pretrain':
        for ood_name in ood_datasets:
            p = Path(f'{dir_name}/{ood_name}/{key}')
            if os.path.exists(p / 'test.pt'):
                ood_infos[ood_name] = pickle.load(open(p / 'test.pt', 'rb'))
            else:
                print('{} not exist'.format(p / 'test.pt'))
    return ood_infos

def load_feature_w_train(dir_name, dataset, key, configs=None):
    p = Path(f'{dir_name}/{dataset}/{key}')

    if os.path.exists(p / 'val_idx.npy'):
        val_idx = np.load(p / 'val_idx.npy')
    else:
        val_idx = None

    if os.path.exists(p / 'train_idx.npy'):
        train_idx = np.load(p / 'train_idx.npy')
    else:
        train_idx = None
    
    val_info = None
    if dataset in ['cifar10', 'cifar100'] or 'cifar10-c' in dataset or 'cifar100-c' in dataset or 'cifar10.1' in dataset:
        try:
            val_info = pickle.load(open(p / 'val.pt', 'rb'))
        except:
            print('{} no val.pt'.format(key))
        test_info = pickle.load(open(p / 'test.pt', 'rb'))

    train_info = None
    if dataset in ['cifar10', 'cifar100'] or 'cifar10-c' in dataset or 'cifar100-c' in dataset:
        try:
            train_info = pickle.load(open(p / 'train.pt', 'rb'))
        except:
            print('{} no train.pt'.format(key))
    # elif dataset in ['imagenet']:
    #     # imagenet late
    #     pass
    # elif dataset in ['stanford_cars']:
    #     val_info = pickle.load(open(p / 'test.pt', 'rb'))
    #     test_info = pickle.load(open(p / 'test.pt', 'rb'))
    #     # ! todo
    unlabel_info = None
    if 'neighbor_source' in configs and configs['neighbor_source'] == 'unlabeled':
        unlabel_info = pickle.load(open(p / 'unlabeled.pt', 'rb'))
    
    return {
        'val_info': val_info,
        'test_info': test_info,
        'val_idx': val_idx,
        'test_idx': None,
        'unlabel_info': unlabel_info,
        'train_info': train_info,
        'train_idx': train_idx
    }



def load_feature(dir_name, dataset, key, configs=None):
    p = Path(f'{dir_name}/{dataset}/{key}')

    if os.path.exists(p / 'val_idx.npy'):
        val_idx = np.load(p / 'val_idx.npy')
    else:
        val_idx = None
    
    val_info = None
    if dataset in ['cifar10', 'cifar100'] or 'cifar10-c' in dataset or 'cifar100-c' in dataset or 'cifar10.1' in dataset:
        try:
            val_info = pickle.load(open(p / 'val.pt', 'rb'))
        except:
            print('{} no val.pt'.format(key))

            # if distribution shift dataset, use normal data as validation set
            import re
            orig_dataset = re.split('\.|-', dataset)[0]
            new_p = Path(f'{dir_name}/{orig_dataset}/{key}')
            val_info = pickle.load( open(new_p / 'val.pt', 'rb'))

            if os.path.exists(new_p / 'val_idx.npy'):
                val_idx = np.load(new_p / 'val_idx.npy')
                print('load val index from {}'.format(new_p))

            print('load val from {}'.format(new_p))

    test_info = pickle.load(open(p / 'test.pt', 'rb'))
    # elif dataset in ['imagenet']:
    #     # imagenet late
    #     pass
    # elif dataset in ['stanford_cars']:
    #     val_info = pickle.load(open(p / 'test.pt', 'rb'))
    #     test_info = pickle.load(open(p / 'test.pt', 'rb'))
    #     # ! todo
    unlabel_info = None
    if 'neighbor_source' in configs and configs['neighbor_source'] == 'unlabeled':
        unlabel_info = pickle.load(open(p / 'unlabeled.pt', 'rb'))

    if os.path.exists(p / 'train_idx.npy'):
        train_idx = np.load(p / 'train_idx.npy')
    else:
        train_idx = None

    train_info = None
    if train_idx is not None:
        train_info = pickle.load(open(p / 'train.pt', 'rb'))
    else:
        try:
            # see if train.pt exist first
            train_info = pickle.load(open(p / 'train.pt', 'rb'))
        except:
            train_info = pickle.load(open(p / 'val.pt', 'rb'))
            print('using val.pt as train info')

    
    return {
        'val_info': val_info,
        'test_info': test_info,
        'val_idx': val_idx,
        'test_idx': None,
        'unlabel_info': unlabel_info,
        'train_info': train_info,
        'train_idx': train_idx
    }

def load_feature_split(dir_name, dataset, key, configs=None):
    p = Path(f'{dir_name}/{dataset}/{key}')

    assert configs['is_val_split_from_test']

    # if dataset in ['stanford_cars', 'imagenet']:
    # val_info = pickle.load(open(p / 'test.pt', 'rb'))
    full_test_info = pickle.load(open(p / 'test.pt', 'rb'))

    all_idx = np.arange(len( list(full_test_info['feats'].values())[0]))
    np.random.seed(configs['split_seed'])
    np.random.shuffle(all_idx)
    val_idx = all_idx[:int(configs['split_ratio'] * len(all_idx))]
    # test_idx = np.delete(all_idx, val_idx)
    test_idx = list(set(all_idx) - set(val_idx))
    assert len(set(test_idx).intersection(set(val_idx))) == 0

    val_info = {}
    test_info = {}
    for k, v in full_test_info.items():
        # print(k, v)
        if k == 'feats':
            _k = list(full_test_info['feats'].keys())[0]
            val_info[k] = {
                f'{_k}': v[_k][val_idx]
            }
            test_info[k] = {
                f'{_k}': v[_k][test_idx]
            }
        else:
            val_info[k] = v[val_idx]
            test_info[k] = v[test_idx]

    # unlabel_info = None
    # if 'neighbor_source' in configs and configs['neighbor_source'] == 'unlabeled':
    #     unlabel_info = pickle.load(open(p / 'unlabeled.pt', 'rb'))

    unlabel_info = None
    if 'neighbor_source' in configs and configs['neighbor_source'] == 'unlabeled':
        unlabel_info = pickle.load(open(p / 'unlabeled.pt', 'rb'))
    
    if os.path.exists(p / 'train_idx.npy'):
        train_idx = np.load(p / 'train_idx.npy')
    else:
        train_idx = None

    train_info = None
    if train_idx is not None:
        train_info = pickle.load(open(p / 'train.pt', 'rb'))
    else:
        try:
            # see if train.pt exist first
            train_info = pickle.load(open(p / 'train.pt', 'rb'))
        except:
            train_info = pickle.load(open(p / 'val.pt', 'rb'))
            print('using val.pt as train info')


    
    return {
        'val_info': val_info,
        'test_info': test_info,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'unlabel_info': unlabel_info,
        'train_info': train_info,
        'train_idx': train_idx
    }

import re
def load_features_for_sup_shift(dir_name, dataset, key, configs=None):
    assert 'cifar10-c' in dataset or 'cifar100-c' in dataset or 'cifar10.1' in dataset\
        or 'imagenetv2' in dataset or 'imagenet-a' in dataset or 'imagenet-sketch' in dataset or 'imagenet-c' in dataset

    shift_maps = {
        'cifar10-c-.*': 'cifar10',
        'cifar100-c-.*': 'cifar100',
        'cifar10\.1.*': 'cifar10',
        'imagenetv2': 'imagenet',
        'imagenet-a': 'imagenet',
        'imagenet-sketch': 'imagenet',
        'imagenet-c-.*': 'imagenet',
    }

    # only test data load from shifted dataset
    # others load from normal dataset
    p = Path(f'{dir_name}/{dataset}/{key}')

    for s_k, s_v in shift_maps.items():
        regrex = re.compile(s_k)
        if regrex.match(dataset):
            normal_p = Path(f'{dir_name}/{s_v}/{key}')
    
    test_info = pickle.load( open(p / 'test.pt', 'rb') ) if os.path.exists(p / 'test.pt') else None
    test_idx = np.load(p / 'test_idx.npy') if os.path.exists(p / 'test_idx.npy') else np.arange(len(test_info['gt_labels'])) # default whole test set

    orig_test_info = pickle.load( open(normal_p / 'test.pt', 'rb') ) if os.path.exists(normal_p / 'test.pt') else None
    train_info = pickle.load( open(normal_p / 'train.pt', 'rb') ) if os.path.exists(normal_p / 'train.pt') else None
    val_info = pickle.load( open(normal_p / 'val.pt', 'rb') ) if os.path.exists(normal_p / 'val.pt') else None
    unlabel_info = pickle.load( open(normal_p / 'unlabeled.pt', 'rb') ) if os.path.exists(normal_p / 'unlabeled.pt') else None

    train_idx = np.load(normal_p / 'train_idx.npy') if os.path.exists(normal_p / 'train_idx.npy') else None
    val_idx = np.load(normal_p / 'val_idx.npy') if os.path.exists(normal_p / 'val_idx.npy') else None

    if configs.get('is_val_split_from_orig_test'):

        # must set val_source from test
        assert configs.get('val_source') == 'orig_test'

        full_test_info = copy.deepcopy(orig_test_info)

        all_idx = np.arange(len( list(full_test_info['feats'].values())[0]))
        np.random.seed(configs['split_seed'])
        np.random.shuffle(all_idx)
        orig_val_idx = all_idx[:int(configs['split_ratio'] * len(all_idx))]
        # test_idx = np.delete(all_idx, val_idx)
        orig_test_idx = list(set(all_idx) - set(orig_val_idx))
        assert len(set(orig_test_idx).intersection(set(orig_val_idx))) == 0

        val_info, _ = get_split_infos(full_test_info, orig_val_idx, orig_test_idx)
        val_idx = orig_val_idx

    return {
        'val_info': val_info,
        'test_info': test_info,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'unlabel_info': unlabel_info,
        'train_info': train_info,
        'train_idx': train_idx
    }


def load_features_for_pre_shift(dir_name, dataset, key, configs=None):
    # assert 'cifar10-c' in dataset or 'cifar100-c' in dataset or 'cifar10.1' in dataset or 'imagenetv2' in dataset or 'imagenet-a' in dataset
    assert 'cifar10-c' in dataset or 'cifar100-c' in dataset or 'cifar10.1' in dataset\
        or 'imagenetv2' in dataset or 'imagenet-a' in dataset or 'imagenet-sketch' in dataset or 'imagenet-c' in dataset

    shift_maps = {
        'cifar10-c-.*': 'cifar10',
        'cifar100-c-.*': 'cifar100',
        'cifar10\.1.*': 'cifar10',
        'imagenetv2': 'imagenet',
        'imagenet-a': 'imagenet',
        'imagenet-sketch': 'imagenet',
        'imagenet-c-.*': 'imagenet',
    }

    # only test data load from shifted dataset
    # others load from normal dataset
    p = Path(f'{dir_name}/{dataset}/{key}')

    for s_k, s_v in shift_maps.items():
        regrex = re.compile(s_k)
        if regrex.match(dataset):
            normal_p = Path(f'{dir_name}/{s_v}/{key}')

    test_info = pickle.load( open(p / 'test.pt', 'rb') ) if os.path.exists(p / 'test.pt') else None

    train_info = pickle.load( open(normal_p / 'train.pt', 'rb') ) if os.path.exists(normal_p / 'train.pt') else None
    val_info = pickle.load( open(normal_p / 'val.pt', 'rb') ) if os.path.exists(normal_p / 'val.pt') else None
    unlabel_info = pickle.load( open(normal_p / 'unlabeled.pt', 'rb') ) if os.path.exists(normal_p / 'unlabeled.pt') else None

    assert configs.get('val_source')

    # load with index
    if configs.get('val_source') == 'train':
        train_info, val_info = get_split_infos(train_info, configs['train_idx'], configs['val_idx'])
    elif configs.get('val_source') == 'orig_test' and configs.get('is_val_split_from_orig_test'):
        orig_test_info = pickle.load( open(normal_p / 'test.pt', 'rb') ) if os.path.exists(normal_p / 'test.pt') else None
        full_test_info = copy.deepcopy(orig_test_info)

        val_info, _ = get_split_infos(full_test_info, configs['val_idx'], [])
        
        if train_info is not None:
            train_info, _ = get_split_infos(train_info, configs['train_idx'], [])

    else:
        # reordered the train info by provided train_idx
        if train_info is not None:
            train_info, _ = get_split_infos(train_info, configs['train_idx'], [])

    

    # elif configs.get('val_source') == 'test':
    #     test_info, val_info = get_split_infos(test_info, configs['test_idx'], configs['val_idx'])
        
    return {
        'train_info': train_info,
        'test_info': test_info,
        'val_info': val_info,
        'unlabel_info': unlabel_info,
    }



def load_features_for_sup(dir_name, dataset, key, configs=None):
    p = Path(f'{dir_name}/{dataset}/{key}').resolve()

    train_info = pickle.load( open(p / 'train.pt', 'rb') ) if os.path.exists(p / 'train.pt') else None
    val_info = pickle.load( open(p / 'val.pt', 'rb') ) if os.path.exists(p / 'val.pt') else None
    test_info = pickle.load( open(p / 'test.pt', 'rb') ) if os.path.exists(p / 'test.pt') else None
    unlabel_info = pickle.load( open(p / 'unlabeled.pt', 'rb') ) if os.path.exists(p / 'unlabeled.pt') else None

    train_idx = np.load(p / 'train_idx.npy') if os.path.exists(p / 'train_idx.npy') else None
    val_idx = np.load(p / 'val_idx.npy') if os.path.exists(p / 'val_idx.npy') else None
    test_idx = np.load(p / 'test_idx.npy') if os.path.exists(p / 'test_idx.npy') else np.arange(len(test_info['gt_labels'])) # default whole test set
    

    # regenerate the val and test splits
    if configs.get('is_val_split_from_test'):

        # must set val_source from test
        assert configs.get('val_source') == 'test'

        full_test_info = copy.deepcopy(test_info)

        all_idx = np.arange(len( list(full_test_info['feats'].values())[0]))
        np.random.seed(configs['split_seed'])
        np.random.shuffle(all_idx)
        val_idx = all_idx[:int(configs['split_ratio'] * len(all_idx))]
        # test_idx = np.delete(all_idx, val_idx)
        test_idx = list(set(all_idx) - set(val_idx))
        assert len(set(test_idx).intersection(set(val_idx))) == 0


        val_info, test_info = get_split_infos(full_test_info, val_idx, test_idx)

    return {
        'val_info': val_info,
        'test_info': test_info,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'unlabel_info': unlabel_info,
        'train_info': train_info,
        'train_idx': train_idx
    }

def get_split_infos(full_test_info, val_idx, test_idx):
    val_info = {}
    test_info = {}
    assert len(set(test_idx).intersection(set(val_idx))) == 0

    for k, v in full_test_info.items():
        # print(k, v)
        if k == 'feats':
            _k = list(full_test_info['feats'].keys())[0]
            val_info[k] = {
                f'{_k}': v[_k][val_idx]
            }
            test_info[k] = {
                f'{_k}': v[_k][test_idx]
            }
        else:
            val_info[k] = v[val_idx]
            test_info[k] = v[test_idx]

    return val_info, test_info

def load_features_for_pre(dir_name, dataset, key, configs=None):
    p = Path(f'{dir_name}/{dataset}/{key}').resolve()

    train_info = pickle.load( open(p / 'train.pt', 'rb') ) if os.path.exists(p / 'train.pt') else None
    val_info = pickle.load( open(p / 'val.pt', 'rb') ) if os.path.exists(p / 'val.pt') else None
    test_info = pickle.load( open(p / 'test.pt', 'rb') ) if os.path.exists(p / 'test.pt') else None
    unlabel_info = pickle.load( open(p / 'unlabeled.pt', 'rb') ) if os.path.exists(p / 'unlabeled.pt') else None

    assert configs.get('val_source')

    # load with index
    if configs.get('val_source') == 'train':
        train_info, val_info = get_split_infos(train_info, configs['train_idx'], configs['val_idx'])
    elif train_info is not None:
        # reordered the train info by provided train_idx
        train_info, _ = get_split_infos(train_info, configs['train_idx'], [])

    if configs.get('val_source') == 'test':
        test_info, val_info = get_split_infos(test_info, configs['test_idx'], configs['val_idx'])
        
    return {
        'train_info': train_info,
        'test_info': test_info,
        'val_info': val_info,
        'unlabel_info': unlabel_info,
    }

def get_is_pos(ind_scores, ood_scores, order):
    assert order in ["largest2smallest", "smallest2largest"]
    scores = np.concatenate((ind_scores, ood_scores))
    is_pos = np.concatenate((np.ones(len(ind_scores), dtype="bool"), np.zeros(len(ood_scores), dtype="bool")))
    
    # shuffle before sort
    random_idx = np.random.permutation(list(range(len(scores))))
    scores = scores[random_idx]
    is_pos = is_pos[random_idx]

    idxs = scores.argsort()
    if order == "largest2smallest":
        idxs = np.flip(idxs)
    is_pos = is_pos[idxs]
    return is_pos

def fpr_at_tpr(ind_scores, ood_scores, order, tpr = 0.95):
    assert order in ["largest2smallest", "smallest2largest"]
    is_pos = get_is_pos(ind_scores, ood_scores, order)
    P = len(ind_scores)
    N = len(ood_scores)
    TP = 0
    FP = 0
    for _is_pos in is_pos:
        if _is_pos:
            TP += 1
        else:
            FP += 1
        TPR = TP / P
        if TPR >= tpr:
            FPR = FP / N
            return FPR

def tnr_at_tpr(ind_scores, ood_scores, order, tpr = 0.95):
    assert order in ["largest2smallest", "smallest2largest"]
    is_pos = get_is_pos(ind_scores, ood_scores, order)
    P = len(ind_scores)
    N = len(ood_scores)
    TP = 0
    TN = N
    for _is_pos in is_pos:
        if _is_pos:
            TP += 1
        else:
            TN -= 1
        TPR = TP / P
        if TPR >= tpr:
            TNR = TN / N
            return TNR

from sklearn.metrics import average_precision_score, roc_auc_score
def eval_res(scores, labels, verbose=True):
    scores = np.array(scores).reshape(-1)
    labels = np.array(labels).reshape(-1)
    # AUPR Err
    err_aupr = average_precision_score((labels + 1)%2, -scores)
    # AUPR Succ
    suc_aupr = average_precision_score(labels, scores)
    # auroc
    auroc = roc_auc_score(labels, scores)

    correct_scores = scores[labels == 1]
    wrong_scores = scores[labels == 0]
    fpr = fpr_at_tpr(correct_scores, wrong_scores, 'largest2smallest', 0.95)
    tnr = tnr_at_tpr(correct_scores, wrong_scores, 'largest2smallest', 0.95)

    # print('fpr_at_tpr 95 ↓:', fpr)
    # print('tnr_at_tpr 95 ↑:', tnr)
    # print('err aupr ↑:', err_aupr)
    # print('succ aupr ↑:', suc_aupr)
    # print('auroc ↑:', auroc)
    rc_curve, aurc = AURC(~labels, scores) 

    res = {
        'fpr': fpr,
        'tnr': tnr,
        'err_aupr': err_aupr,
        'suc_aupr': suc_aupr,
        'auroc': auroc,
        'aurc': aurc
    }

    if verbose:
        print({
            'fpr ↓': fpr,
            'tnr ↑': tnr,
            'err_aupr ↑': err_aupr,
            'suc_aupr ↑': suc_aupr,
            'auroc ↑': auroc,
            'aurc ↓': aurc,
        })

    return res

def AURC(residuals, confidence):
    coverages = []
    risks = []
    n = len(residuals)
    idx_sorted = np.argsort(confidence)
    cov = n
    error_sum = sum(residuals[idx_sorted])
    coverages.append(cov/ n),
    risks.append(error_sum / n)
    weights = []
    tmp_weight = 0

    for i in range(0, len(idx_sorted) - 1):

        cov = cov-1
        error_sum = error_sum - residuals[idx_sorted[i]]
        selective_risk = error_sum /(n - 1 - i)
        tmp_weight += 1

        if i == 0 or \
        confidence[idx_sorted[i]] != confidence[idx_sorted[i - 1]]:

            coverages.append(cov / n)
            risks.append(selective_risk)
            weights.append(tmp_weight / n)
            tmp_weight = 0

    if tmp_weight > 0:

        coverages.append(0)
        risks.append(risks[-1])
        weights.append(tmp_weight / n)

    aurc = sum([(risks[i] + risks[i+1]) * 0.5 \
    * weights[i] for i in range(len(weights)) ])

    curve = (coverages, risks)
    return curve, aurc