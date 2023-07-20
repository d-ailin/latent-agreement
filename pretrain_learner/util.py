from copy import deepcopy
from ensurepip import version
from operator import index
from scipy.stats import wasserstein_distance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import torch

def show_dist(scores, test_corrects, p):
    plt.clf()
    
    test_corrects = test_corrects.astype(np.int)
    
    sns.distplot(scores[test_corrects == 1], label='success')
    sns.distplot(scores[test_corrects == 0], label='error')
    plt.legend()
    plt.savefig(p)


def show_ssl_cls_paper(pred_corrects, rot_scores, config={}, save_img_path=''):
    plt.style.use('seaborn-whitegrid')

    font = {
        # 'family' : 'normal',
        # 'weight' : 'bold',
            'size'   : 12}

    mpl.rc('font', **font)

    xlabel = config['xlabel']

    sns.color_palette("colorblind")
    # plt.figure(figsize=(3.5, 3.5))
    plt.figure(figsize=(2.8, 2.5))

    rot_scores_left_bins = np.arange(0, 1, 0.1)
    rot_scores_right_bins = np.arange(0.1, 1.1, 0.1)
    X = np.arange(len(rot_scores_left_bins))

    xtick_arr = []
    res_arr = []
    count_arr = []
    for rot_bin_left, rot_bin_right in zip(rot_scores_left_bins, rot_scores_right_bins):
        rot_mask = (rot_scores >= rot_bin_left) & (rot_scores <= rot_bin_right)
        if rot_bin_right == 1:
            rot_mask = (rot_scores >= rot_bin_left) & (rot_scores <= rot_bin_right)
        
        pred_total = len(rot_scores[rot_mask])
        pred_correct_num = sum(pred_corrects[rot_mask])
        if pred_total > 0:
            pred_acc = pred_correct_num / pred_total * 100
        else:
            pred_acc = 0
        res_arr.append(pred_acc)
        xtick_arr.append( round(rot_bin_left, 2))
        count_arr.append(pred_total)

    plt.bar(X+0.5, res_arr, width=1, alpha=config['alpha'])
    print(config['title'], xtick_arr, res_arr, count_arr)

    x_index = np.array(X.tolist() + [10])
    x_ticks = np.array(xtick_arr + [1.0])
    plt.xticks(x_index[::2], x_ticks[::2])

    ax = plt.gca()
    temp = ax.xaxis.get_ticklabels()
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 1 != 0:
            label.set_visible(False)
    # for index,data in enumerate(res_arr):
    #     if data > 0:
    #         plt.text(x=X[index]-0.05 , y =data+0.01 , s=f"{round(data,2)}" , fontdict=dict(fontsize=10))
    plt.xlim(0, 10)
    # plt.ylim(bottom=max(0, min(res_arr) - 0.1), top=1)
    plt.ylim(bottom=max(0, min(res_arr) - 10), top=100)
    # plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('Classfication Accuracy (%)')
    plt.tight_layout()
    plt.title(config['title'])

    if len(save_img_path) > 0:
    # plt.title('cifar10 {} trained epoch({}):  rot pred conf vs classification pred acc'.format(tag, epoch))
        plt.savefig('{}_{}.png'.format(save_img_path, config['title']))
        plt.savefig('{}_{}.pdf'.format(save_img_path, config['title']))
    else:
        plt.show()


def show_acc_vs_values_in_bins(jaccard_scores, bin_thresholds, pred_corrects, tag='', verbose=False):
    bin_nums = np.zeros(len(bin_thresholds))
    bin_corrects = np.zeros(len(bin_thresholds))
    bin_accs = np.zeros(len(bin_thresholds))
    bin_nums_for_incorrects = np.zeros(len(bin_thresholds))

    for i, v in enumerate(bin_thresholds):
        next_val = bin_thresholds[i+1] if i < len(bin_thresholds)-1 else 1
        
        mask = (jaccard_scores >= v) & (jaccard_scores < next_val)
        
        bin_nums[i] = sum(mask)
        bin_nums_for_incorrects[i] = (~pred_corrects)[mask].sum()
        bin_corrects[i] = pred_corrects[mask].sum()
        if sum(mask) > 0:
            bin_accs[i] = pred_corrects[mask].sum() / sum(mask).round(2)
        else:
            bin_accs[i] = 0
        if verbose:
            print(v, bin_accs[i], bin_nums[i])
    
    bar_width= .5
    # plt.figure(figsize=(9,5))
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 5))
    ax1.bar(np.arange(len(bin_thresholds)), bin_accs, bar_width, tick_label = [e.round(2) for e in bin_thresholds])
    # draw value text
    for j in range(len(bin_thresholds)):
        ax1.text(j-bar_width/2, bin_accs[j]+0.02, str(bin_accs[j].round(3)), fontsize=8, va='center')

    avg_acc = sum(pred_corrects) / len(pred_corrects)

    ax1.axhline(avg_acc, linestyle='--')
    ax1.text(-2, avg_acc, avg_acc, fontsize=10)
    ax1.set_title(f'{tag} dist. for all data' )

    bar_width= .8
    # plt.figure(figsize=(10,5))
    ax2.bar(np.arange(len(bin_thresholds)), bin_nums, bar_width, tick_label = [e.round(2) for e in bin_thresholds])

    for j in range(len(bin_thresholds)):
        ax2.text(j-.4, bin_nums[j]+0.1, str(bin_nums[j]), fontsize=8, va='center')

    ax2.set_title(f'sample num across {tag} for all data')

    plt.show()


def show_acc_vs_values_in_bins_of_classes(jaccard_scores, bin_thresholds, class_num, pred_corrects, gt_labels, class_names, tag=''):

    for c in range(class_num):
        # for each class
        c_bin_nums = np.zeros(len(bin_thresholds))
        c_bin_corrects = np.zeros(len(bin_thresholds))
        c_bin_accs = np.zeros(len(bin_thresholds))
        c_avg_acc = sum(pred_corrects[gt_labels == c]) / len(pred_corrects[gt_labels == c])
        for i, v in enumerate(bin_thresholds):
            # mask = (jaccard_scores == v) & (gt_labels == c)
            
            next_val = bin_thresholds[i+1] if i < len(bin_thresholds)-1 else 1
            mask = (jaccard_scores >= v) & (jaccard_scores < next_val) & (gt_labels == c)
            
            c_bin_nums[i] = sum(mask)
            c_bin_corrects[i] = pred_corrects[mask].sum()
            if sum(mask) == 0:
                c_bin_accs[i] = 0
            else:
                c_bin_accs[i] = pred_corrects[mask].sum() / sum(mask).round(2)
            
        bar_width= .5
        # plt.figure(figsize=(9,5))
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
        ax1.bar(np.arange(len(bin_thresholds)), c_bin_accs, bar_width, tick_label = [e.round(2) for e in bin_thresholds])
        # draw value text
        for j in range(len(bin_thresholds)):
            ax1.text(j-bar_width/2, c_bin_accs[j]+0.02, str(c_bin_accs[j].round(3)), fontsize=8, va='center')
        ax1.axhline(c_avg_acc, linestyle='--')
        ax1.text(-2, c_avg_acc, c_avg_acc, fontsize=10)
        # ax1.set_title('jaccard values dist. for class {}'.format(mapping[dataset][c]) )
        ax1.set_title('{} dist. for class {}'.format(tag, class_names[c]) )
        
        bar_width= .8
        # plt.figure(figsize=(10,5))
        ax2.bar(np.arange(len(bin_thresholds)), c_bin_nums, bar_width, tick_label = [e.round(2) for e in bin_thresholds])
        
        for j in range(len(bin_thresholds)):
            ax2.text(j-.4, c_bin_nums[j]+0.1, str(c_bin_nums[j]), fontsize=8, va='center')
            
        ax2.set_title('sample num across {} for class {}'.format(tag, class_names[c]) )

    plt.show()



import torchvision.transforms as transforms
def show_samples_in_bins(all_imgs, scores, bin_thresholds, info={}, tag=''):
    gt_labels = info['gt_labels']
    pred_scores = info['pred_scores']
    pred_labels = info['pred_labels']
    pred_corrects = info['pred_corrects']

    all_ids = np.arange(len(all_imgs))

    # for c in range(class_num):
    #     c_mask = gt_labels == c
    show_num = 5
    fig, axes = plt.subplots(len(bin_thresholds), show_num, figsize=(show_num*5, len(bin_thresholds)*2.5))
    # fig.suptitle('class:{}'.format(class_names[c]))
    plt.axis('off')
    for i, v in enumerate(bin_thresholds):
        # c_jaccard_mask = c_mask & (jaccard_scores == v)
        
        next_val = bin_thresholds[i+1] if i < len(bin_thresholds)-1 else 1
        c_jaccard_mask = (scores >= v) & (scores < next_val)
        
        # c_jaccard_mask = c_mask & (scores == v)
        
        c_imgs = all_imgs[c_jaccard_mask, :]
        c_pred_scores = pred_scores[c_jaccard_mask]
        c_corrects = pred_corrects[c_jaccard_mask]
        c_pred_labels = pred_labels[c_jaccard_mask]
        c_gt_labels = gt_labels[c_jaccard_mask]
        c_ids = all_ids[c_jaccard_mask]
        c_jaccar_scores = scores[c_jaccard_mask]
        
        final_show_num = min(show_num, len(c_imgs))
        for j in range(final_show_num):
            im = transforms.ToPILImage()(c_imgs[j]).convert('RGB')            
            axes[i][j].imshow(im)
            
            axes[i][j].set_title('id: {}; {}: {:.2f}; msp: {:.2f}'.format(
                c_ids[j], tag, c_jaccar_scores[j], c_pred_scores[j]
            ), fontsize=8)
            
            axes[i][j].set_axis_off()

    plt.show()

def show_samples_in_bins_across_classes(all_imgs, scores, bin_thresholds, class_num, class_names, info={}, tag=''):
    gt_labels = info['gt_labels']
    pred_scores = info['pred_scores']
    pred_labels = info['pred_labels']
    pred_corrects = info['pred_corrects']

    all_ids = np.arange(len(all_imgs))

    for c in range(class_num):
        c_mask = gt_labels == c
        show_num = 5
        fig, axes = plt.subplots(len(bin_thresholds), show_num, figsize=(show_num*5, len(bin_thresholds)*2.5))
        fig.suptitle('class:{}'.format(class_names[c]))
        plt.axis('off')
        for i, v in enumerate(bin_thresholds):
            # c_jaccard_mask = c_mask & (jaccard_scores == v)
            
            next_val = bin_thresholds[i+1] if i < len(bin_thresholds)-1 else 1
            c_jaccard_mask = (scores >= v) & (scores < next_val) & c_mask
            
            # c_jaccard_mask = c_mask & (scores == v)
            
            c_imgs = all_imgs[c_jaccard_mask, :]
            c_pred_scores = pred_scores[c_jaccard_mask]
            c_corrects = pred_corrects[c_jaccard_mask]
            c_pred_labels = pred_labels[c_jaccard_mask]
            c_gt_labels = gt_labels[c_jaccard_mask]
            c_ids = all_ids[c_jaccard_mask]
            c_jaccar_scores = scores[c_jaccard_mask]
            
            final_show_num = min(show_num, len(c_imgs))
            for j in range(final_show_num):
                im = transforms.ToPILImage()(c_imgs[j]).convert('RGB')            
                axes[i][j].imshow(im)
                
                axes[i][j].set_title('id: {}; c:{} gt: {}; pred: {};\n {}: {:.2f}; msp: {:.2f}'.format(
                    c_ids[j], c_corrects[j], class_names[c_gt_labels[j]], class_names[c_pred_labels[j]], tag, c_jaccar_scores[j], c_pred_scores[j]
                ), fontsize=8)
                
                axes[i][j].set_axis_off()

    plt.show()

def show_incorrect_samples_in_bins_across_classes(all_imgs, scores, bin_thresholds, class_num, class_names, info={}, tag=''):
    gt_labels = info['gt_labels']
    pred_scores = info['pred_scores']
    pred_labels = info['pred_labels']
    pred_corrects = info['pred_corrects']

    all_ids = np.arange(len(all_imgs))

    for c in range(class_num):
        c_mask = gt_labels == c
        show_num = 5
        fig, axes = plt.subplots(len(bin_thresholds), show_num, figsize=(show_num*5, len(bin_thresholds)*2.5))
        fig.suptitle('class:{}'.format(class_names[c]))
        plt.axis('off')
        for i, v in enumerate(bin_thresholds):
            # c_jaccard_mask = c_mask & (jaccard_scores == v)
            
            next_val = bin_thresholds[i+1] if i < len(bin_thresholds)-1 else 1
            c_jaccard_mask = (scores >= v) & (scores < next_val) & c_mask & ~pred_corrects
            
            # c_jaccard_mask = c_mask & (scores == v)
            
            c_imgs = all_imgs[c_jaccard_mask, :]
            c_pred_scores = pred_scores[c_jaccard_mask]
            c_corrects = pred_corrects[c_jaccard_mask]
            c_pred_labels = pred_labels[c_jaccard_mask]
            c_gt_labels = gt_labels[c_jaccard_mask]
            c_ids = all_ids[c_jaccard_mask]
            c_jaccar_scores = scores[c_jaccard_mask]
            
            final_show_num = min(show_num, len(c_imgs))
            for j in range(final_show_num):
                im = transforms.ToPILImage()(c_imgs[j]).convert('RGB')            
                axes[i][j].imshow(im)
                
                axes[i][j].set_title('id: {}; c:{} gt: {}; pred: {};\n {}: {:.2f}; msp: {:.2f}'.format(
                    c_ids[j], c_corrects[j], class_names[c_gt_labels[j]], class_names[c_pred_labels[j]], tag, c_jaccar_scores[j], c_pred_scores[j]
                ), fontsize=8)
                
                axes[i][j].set_axis_off()

    plt.show()

def get_grid_info_between_2set_scores(scores_a, scores_b, info={}, bins=20, bin_style='quantile'):
    # pred_scores = info['pred_scores']
    # pred_labels = info['pred_labels']
    pred_corrects = info['pred_corrects']

    # use quantiles as thresholds to split
    step = 1 / bins
    if bin_style == 'quantile':
        a_thres = np.quantile(scores_a, np.arange(0, 1, step))
        b_thres = np.quantile(scores_b, np.arange(0, 1, step))
    elif bin_style == 'width':
        a_thres = ((np.max(scores_a) - np.min(scores_a)) / bins * np.arange(bins) ) + np.min(scores_a)
        b_thres = ((np.max(scores_b) - np.min(scores_b)) / bins * np.arange(bins) ) + np.min(scores_b)

    grid_maps = {
        'acc': np.zeros((bins, bins)),
        'count': np.zeros((bins, bins)).astype(np.int32),
        'avg_scores_a': np.zeros((bins, bins)),
        'avg_scores_b': np.zeros((bins, bins)),
        'delta_scores': np.zeros((bins, bins)),
        'mask': [[0]*bins for _i in range(bins)],
        'index': [[0]*bins for _i in range(bins)],
    }


    for i, a_thre in enumerate(a_thres):
        next_val_a = a_thres[i+1] if i < len(a_thres)-1 else 1
        for j, b_thre in enumerate(b_thres):
            # c_jaccard_mask = c_mask & (jaccard_scores == v)
            
            next_val_b = b_thres[j+1] if j < len(b_thres)-1 else 1

            mask = (scores_a >= a_thre) & (scores_a < next_val_a) & (scores_b >= b_thre) & (scores_b < next_val_b)

            count = sum(mask)
        
            acc = round(sum(pred_corrects[mask]) / count, 3) if count > 0 else 0
            avg_scores_a = scores_a[mask].mean()
            avg_scores_b = scores_b[mask].mean()

            grid_maps['acc'][i][j] = acc
            grid_maps['count'][i][j] = count
            grid_maps['avg_scores_a'][i][j] = avg_scores_a
            grid_maps['avg_scores_b'][i][j] = avg_scores_b

            grid_maps['mask'][i][j] = mask
            grid_maps['index'][i][j] = np.where(mask == 1)


    return grid_maps


from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score, jaccard_score
from tqdm import tqdm
import faiss

def get_knn_index(all_features):
    index = faiss.IndexFlatL2(all_features.shape[1])
    index.add(all_features)

    return index


def get_knn_mat(all_features, search_features, n_neighbours, knn_index=None):
    if knn_index is None:
        index = faiss.IndexFlatL2(all_features.shape[1])
        index.add(np.ascontiguousarray(all_features))
    else:
        index = knn_index
    D, indices = index.search(np.ascontiguousarray(search_features), n_neighbours+1)
    
    is_include_same = np.array_equal(search_features, all_features[indices[:, 0]])
    
    if is_include_same:
        print('is_include_same', is_include_same)
        return indices[:, 1:]
    return indices[:, :n_neighbours]
    # return indices[:, 1:]

def get_knn_dist(all_features, search_features, n_neighbours):
    normed_all_features = all_features / np.linalg.norm(all_features, axis=1)[:, np.newaxis]
    normed_search_features = search_features / np.linalg.norm(search_features, axis=1)[:, np.newaxis]


    index = faiss.IndexFlatL2(normed_all_features.shape[1])
    # index.add(train_set)
    index.add(np.ascontiguousarray(normed_all_features))
    D, indices = index.search(np.ascontiguousarray(normed_search_features), n_neighbours+1)

    
    is_include_same = np.array_equal(normed_search_features, normed_all_features[indices[:, 0]])
    
    # return indices[:, 1:]
    # if ignore_first:
    if is_include_same:
        print('is_include_same', is_include_same)
        return D[:, 1:].mean(-1)
    return D[:, :n_neighbours].mean(-1)


from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import label_ranking_average_precision_score
from scipy.stats import kendalltau, spearmanr
from utils.cka import CudaCKA
import time


def get_def_scores(n_neighbours, search_features, all_features, search_mask, metric='auroc', return_list=False, verbose=False, knn_index=None):
    main_search_features, main_features = search_features[0], all_features[0]
    
    # get gt neighbor sets, 0/1, 1 - neighbor, 0 - not neighbor
    if verbose:
        print('main_search_features', main_search_features.shape)
        print('main_features', main_features.shape)

    
    normed_main_search_features = main_search_features / np.linalg.norm(main_search_features, axis=1)[:, np.newaxis]
    normed_main_features = main_features / np.linalg.norm(main_features, axis=1)[:, np.newaxis]

    # 2. search with normed
    gt_neighbor_indexs = get_knn_mat(normed_main_features, normed_main_search_features[search_mask], n_neighbours)
    gt_neighbor_mask = np.zeros((len(main_search_features), len(main_features)))
    for i in range(len(gt_neighbor_indexs)):
        gt_neighbor_mask[i][gt_neighbor_indexs[i]] = 1

    main_cos_sim = cosine_similarity(main_search_features, main_features)


    all_auroc_list = []
    # get the other features' cosine similarity scores
    for j in range(1, len(all_features)):
        other_search_features = search_features[j]
        other_features = all_features[j]
        
        normed_other_search_features = other_search_features / np.linalg.norm(other_search_features, axis=1)[:, np.newaxis]
        normed_other_features = other_features / np.linalg.norm(other_features, axis=1)[:, np.newaxis]
        
        # other_knn_index = get_knn_index(normed_other_features)
        
        # other_neighbor_indexs = get_knn_mat(normed_other_features, normed_other_search_features[search_mask], n_neighbours, knn_index=other_knn_index)
        other_neighbor_indexs = get_knn_mat(normed_other_features, normed_other_search_features[search_mask], n_neighbours)
        other_neighbor_mask = np.zeros((len(normed_other_search_features), len(normed_other_features))).astype(np.bool)
        for i in range(len(other_neighbor_indexs)):
            other_neighbor_mask[i][other_neighbor_indexs[i]] = 1

        cos_sim = cosine_similarity(other_search_features, other_features)
        

        auroc_list = []
        if verbose:
            range_obj = tqdm(range(len(cos_sim)))
        else:
            range_obj = range(len(cos_sim))
        for i in range_obj:
            if metric == 'ndcg_rank':
                auroc_list += [ndcg_score([gt_neighbor_mask[i]], [cos_sim[i]], k=n_neighbours)]
            elif metric == 'jaccard':
                auroc_list += [jaccard_score( gt_neighbor_mask[i].astype(np.bool), other_neighbor_mask[i].astype(np.bool) )]
            elif metric == 'cka_topk':
                a_features = main_features[ gt_neighbor_mask[i].astype(np.bool)]
                b_features = other_features[ gt_neighbor_mask[i].astype(np.bool)]
                                
                cka_obj = CudaCKA('cuda:0')
                cka_sim = cka_obj.linear_CKA(
                    torch.from_numpy(a_features).cuda(), 
                    torch.from_numpy(b_features).cuda()
                ).item()
                
                auroc_list += [cka_sim]
            elif metric == 'cka_kernel_topk':
                a_features = main_features[ gt_neighbor_mask[i].astype(np.bool)]
                b_features = other_features[ gt_neighbor_mask[i].astype(np.bool)]
                                
                cka_obj = CudaCKA('cuda:0')
                cka_sim = cka_obj.kernel_CKA(
                    torch.from_numpy(a_features).cuda(), 
                    torch.from_numpy(b_features).cuda()
                ).item()
                
                auroc_list += [cka_sim]
            elif metric == 'spearmanr':
                rank_coef, _ = spearmanr( main_cos_sim[i], cos_sim[i])
                auroc_list += [rank_coef]

        all_auroc_list.append(auroc_list)
        # print('aurco_score_list', auroc_list)
        
    all_auroc_list = np.array(all_auroc_list)
    if verbose:
        print('all_auroc_list', all_auroc_list.shape)
        if len(all_auroc_list) > 0:
            print('all_auroc_list', all_auroc_list[0, :5])
    
    # return all scores list
    if return_list:
        if all_auroc_list.shape[0] > 1:
            return all_auroc_list.mean(axis=0), all_auroc_list.transpose()
        else:
            return all_auroc_list, all_auroc_list.transpose()

    # get the means
    return all_auroc_list.mean(axis=0)



def show_samples_in_grid_bin(grid_maps, info, imgs, i, j, title, show_num=5, filter_mask=None):
    indexs = grid_maps['index'][i][j][0]

    mask = grid_maps['mask'][i][j]

    if filter_mask is not None:
        mask = mask & filter_mask


    y_score = grid_maps['avg_scores_a'][i][j]
    x_score = grid_maps['avg_scores_b'][i][j]

    pred_scores = info['pred_scores']
    pred_corrects = info['pred_corrects']
    pred_labels = info['pred_labels']
    gt_labels = info['gt_labels']
    class_names = info['class_names']

    ssl_ndcg_scores = info['ssl_ndcg_scores']
    sup_ndcg_scores = info['sup_ndcg_scores']
    ssl_ndcg_var = info['ssl_ndcg_var']
    sup_ndcg_var = info['sup_ndcg_var']

    all_ids = np.arange(len(imgs))

    # plt.figure(figsize=(10, 5))
    fig, axes = plt.subplots(1, show_num, figsize=(show_num*5, 2), squeeze=False)
    fig.suptitle('bin[{}][{}]: avg_scores_a: {}, avg_scores_b:{}'.format(i, j, y_score, x_score), y=1.3)
    plt.axis('off')

    final_show_num = min(show_num, sum(mask))

    c_imgs = imgs[mask]
    c_ids = all_ids[mask]
    c_gt_labels = gt_labels[mask]
    c_corrects = pred_corrects[mask]
    c_pred_labels = pred_labels[mask]
    c_ssl_ndcg_scores = ssl_ndcg_scores[mask]
    c_sup_ndcg_scores = sup_ndcg_scores[mask]
    c_ssl_ndcg_var = ssl_ndcg_var[mask]
    c_sup_ndcg_var = sup_ndcg_var[mask]

    c_pred_scores = pred_scores[mask]

    # print('final_show_num', final_show_num, len(indexs), indexs)

    for j in range(final_show_num):
        im = transforms.ToPILImage()(c_imgs[j]).convert('RGB')            
        axes[0][j].imshow(im)
        
        axes[0][j].set_title('id: {}; c:{} gt: {}; pred: {};\n msp: {:.2f}; ssl_ndcg: {:.2f}; sup_ndcg: {:.2f} \n ssl_ndcg_var: {:.3f}; sup_ndcg_var: {:.3f}'.format(
            c_ids[j], c_corrects[j], class_names[c_gt_labels[j]], class_names[c_pred_labels[j]], 
            c_pred_scores[j], c_ssl_ndcg_scores[j], c_sup_ndcg_scores[j],
            c_ssl_ndcg_var[j], c_sup_ndcg_var[j],
        ), fontsize=8)
        
        axes[0][j].set_axis_off()

    plt.show()

def show_samples_easy(imgs, filter_mask=[], indices=[], show_num=15):
    if len(indices) > 0:
        # filter_mask = np.zeros(len(imgs), dtype=bool)
        # filter_mask[indices] = 1
        filter_mask = indices

        final_show_num = min(show_num, len(filter_mask))
    else:
        final_show_num =  min(show_num, sum(filter_mask))

    fig, axes = plt.subplots(1, show_num, figsize=(show_num*5, 2), squeeze=False)
    # fig.suptitle('bin[{}][{}]: avg_scores_a: {}, avg_scores_b:{}'.format(i, j, y_score, x_score), y=1.3)
    plt.axis('off')

    c_imgs = imgs[filter_mask]

    for j in range(final_show_num):
        im = transforms.ToPILImage()(c_imgs[j]).convert('RGB')            
        axes[0][j].imshow(im)        
        axes[0][j].set_axis_off()

    plt.show()

def show_samples_in_mask(info, imgs, title, show_num=5, filter_mask=None):
    mask = filter_mask

    pred_scores = info['pred_scores']
    pred_corrects = info['pred_corrects']
    pred_labels = info['pred_labels']
    gt_labels = info['gt_labels']
    class_names = info['class_names']

    ssl_ndcg_scores = info['ssl_ndcg_scores']
    # sup_ndcg_scores = info['sup_ndcg_scores']
    # ssl_ndcg_var = info['ssl_ndcg_var']
    # sup_ndcg_var = info['sup_ndcg_var']

    all_ids = np.arange(len(imgs))

    # plt.figure(figsize=(10, 5))
    fig, axes = plt.subplots(1, show_num, figsize=(show_num*5, 2), squeeze=False)
    # fig.suptitle('bin[{}][{}]: avg_scores_a: {}, avg_scores_b:{}'.format(i, j, y_score, x_score), y=1.3)
    fig.suptitle(title, y= 1.3)
    plt.axis('off')

    final_show_num = min(show_num, sum(mask))

    c_imgs = imgs[mask]
    c_ids = all_ids[mask]
    c_gt_labels = gt_labels[mask]
    c_corrects = pred_corrects[mask]
    c_pred_labels = pred_labels[mask]
    c_ssl_ndcg_scores = ssl_ndcg_scores[mask]
    # c_sup_ndcg_scores = sup_ndcg_scores[mask]
    # c_ssl_ndcg_var = ssl_ndcg_var[mask]
    # c_sup_ndcg_var = sup_ndcg_var[mask]

    c_pred_scores = pred_scores[mask]

    # print('final_show_num', final_show_num, len(indexs), indexs)

    for j in range(final_show_num):
        im = transforms.ToPILImage()(c_imgs[j]).convert('RGB')            
        axes[0][j].imshow(im)


        axes[0][j].set_title('id: {}; c:{} gt: {}; pred: {};\n msp: {:.2f}; ssl_ndcg: {:.2f};'.format(
            c_ids[j], c_corrects[j], class_names[c_gt_labels[j]], class_names[c_pred_labels[j]], 
            c_pred_scores[j], c_ssl_ndcg_scores[j]
        ), fontsize=8)
        
        axes[0][j].set_axis_off()

    plt.show()
    
def get_and_show_neighbor_samples(search_features, all_features, main_imgs, search_imgs, main_mask, n_neighbours=50, info={}, search_info={}, is_plot=True):
    # find out the neighbor index
    neighbor_indexs = get_knn_mat(all_features, search_features[main_mask], n_neighbours)

    if not is_plot:
        return {
            'neighbor_indexs': neighbor_indexs
        } 

    # show main img and the neighbor imgs and their information
    main_index = np.where(main_mask == 1)[0]

    pred_scores = info['pred_scores']
    pred_corrects = info['pred_corrects']
    pred_labels = info['pred_labels']
    gt_labels = info['gt_labels']
    class_names = info['class_names']

    ssl_ndcg_scores = info['ssl_ndcg_scores']
    sup_ndcg_scores = info['sup_ndcg_scores']
    ssl_ndcg_var = info['ssl_ndcg_var']
    sup_ndcg_var = info['sup_ndcg_var']

    search_pred_scores = search_info['pred_scores']
    search_pred_corrects = search_info['pred_corrects']
    search_pred_labels = search_info['pred_labels']
    search_gt_labels = search_info['gt_labels']

    all_ids = np.arange(len(search_features))
    val_ids = np.arange(len(all_features))


    fig, axes = plt.subplots(len(main_index), n_neighbours + 1, figsize=(4 * n_neighbours, len(main_index) * 3.5))

    for i, index in enumerate(main_index):
        show_main_img = main_imgs[index]
        neighbor_index = neighbor_indexs[i]

        c_imgs = search_imgs[neighbor_index]
        c_gt_labels = search_gt_labels[neighbor_index]
        c_corrects = search_pred_corrects[neighbor_index]
        c_pred_labels = search_pred_labels[neighbor_index]
        c_pred_scores = search_pred_scores[neighbor_index]

        m_ids = all_ids[index]
        m_gt_labels = gt_labels[index]
        m_corrects = pred_corrects[index]
        m_pred_labels = pred_labels[index]
        m_ssl_ndcg_scores = ssl_ndcg_scores[index]
        m_sup_ndcg_scores = sup_ndcg_scores[index]
        m_ssl_ndcg_var = ssl_ndcg_var[index]
        m_sup_ndcg_var = sup_ndcg_var[index]
        m_pred_scores = pred_scores[index]

        im = transforms.ToPILImage()(show_main_img).convert('RGB')            
        axes[i][0].imshow(im)
        
        axes[i][0].set_title('id: {}; c:{} gt: {}; pred: {};\n msp: {:.2f}; ssl_ndcg: {:.2f}; sup_ndcg: {:.2f} \n ssl_ndcg_var: {:.3f}; sup_ndcg_var: {:.3f}'.format(
            m_ids, m_corrects, class_names[m_gt_labels], class_names[m_pred_labels], 
            m_pred_scores, m_ssl_ndcg_scores, m_sup_ndcg_scores,
            m_ssl_ndcg_var, m_sup_ndcg_var,
        ), fontsize=8)
        
        axes[i][0].set_axis_off()


        # plot main imgs
        for j in range(n_neighbours):
            im = transforms.ToPILImage()(c_imgs[j]).convert('RGB')            
            axes[i][j+1].imshow(im)
            
            axes[i][j+1].set_title('val_id:{}; c:{} gt: {}; pred: {};\n msp: {:.2f};'.format(
                neighbor_index[j],
                c_corrects[j], class_names[c_gt_labels[j]], class_names[c_pred_labels[j]], 
                c_pred_scores[j],
            ), fontsize=8)
            
            axes[i][j+1].set_axis_off()
    plt.show()

    return {
        'neighbor_indexs': neighbor_indexs
    }


def get_intersection(A, B):
    # A, B can be multiple dimensions, but same rows
    # keeping the indexes

    inter_set = []
    for i in range(A.shape[0]):
        intersection = list(set(A[i]).intersection(B[i]))

        inter_set.append(sorted(intersection))

    return inter_set

def get_pos_in_grid_map(grid_maps, index):

    bin_num = len(grid_maps['index'])
    for i in range(bin_num):
        for j in range(bin_num):
            if index in grid_maps['index'][i][j][0]:
                return (i, j)


