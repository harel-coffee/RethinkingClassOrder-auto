# -*- coding:utf-8 -*-  

""" 
@time: 1/18/22 3:07 PM 
@author: Chen He 
@site:  
@file: display_accs.py
@description:  
"""
import argparse
import os

import matplotlib

matplotlib.use('Agg')

import numpy as np

PROJ_DIR = os.path.dirname(os.path.dirname(__file__))


def calc_mean_std(result):
    mean_result = np.mean(result, axis=0)
    std_result = np.std(result, axis=0)
    return mean_result, std_result


def vis_multiple(result_dir_dict, total_cl, nb_cl, keys, MIDDLE_FOLDER=''):
    for key_idx, key in enumerate(keys):

        result_dirs = result_dir_dict[key]
        aver_acc_over_time_mul = []

        for result_dir in result_dirs:
            accs = []
            for group_idx in range((total_cl - nb_cl) // nb_cl + 1):
                conf_mat_filename = os.path.join(result_dir, 'group_%d' % (group_idx + 1), MIDDLE_FOLDER,
                                                 'conf_mat.npy')
                if not os.path.exists(conf_mat_filename):
                    raise Exception('%s not exist' % conf_mat_filename)
                conf_mat = np.load(conf_mat_filename)
                acc = np.mean(np.diag(conf_mat) * 100. / np.sum(conf_mat, axis=1))
                accs.append(acc)
            accs = np.array(accs)
            aver_acc_over_time_mul.append(accs)

        y_mean, y_std = calc_mean_std(np.array(aver_acc_over_time_mul))

        print('%s\t%.2f ± %.2f' % (key, y_mean[-1], y_std[-1]))
        # print('%.2f ± %.2f' % (y_mean[-1], y_std[-1]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--order_type', type=str, default='even')
    args = parser.parse_args()

    if args.dataset == 'imagenet':
        dataset = 'imagenet64x64_10x10'
        display_dataset = 'Group ImageNet'
        total_cl, nb_cl = 100, 10
    elif args.dataset == 'inat':
        dataset = 'inat_9x9'
        total_cl, nb_cl = 81, 9
        display_dataset = 'Group iNaturalist'
    else:
        raise Exception('Invalid dataset')

    class_order = args.order_type
    network = 'resnet18'
    stage = '2nd_stage'
    num_epochs = 70
    num_exemplars = 20
    exemplar_budget = 'total'
    num_order = 5
    use_lwf = True
    exemplar_selection = 'random'
    suffix = ''
    base_lr = 0.005

    inc_protocol = 'base_%d_inc_%d_total_%d' % (nb_cl, nb_cl, total_cl)

    result_dir_dict = {
        'iCaRL': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_%d_adam_%s_no_final_relu_aug_wd_0.0001_%s_%d_%s%s_embedding_cosine%s' % (
                dataset, class_order, i + 1, inc_protocol, network, num_epochs, str(base_lr), exemplar_selection,
                num_exemplars, exemplar_budget,
                '_lwf_1.0_temp_2.0' if use_lwf else '', suffix) for i in
            range(num_order)
        ],
        'EEIL': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_%d_adam_%s_aug_wd_0.0001_%s_%d_%s%s_eeil_30%s' % (
                dataset, class_order, i + 1, inc_protocol, network, num_epochs, str(base_lr), exemplar_selection,
                num_exemplars, exemplar_budget,
                '_lwf_1.0_temp_2.0' if use_lwf else '', suffix) for i in
            range(num_order)
        ],
        'LSIL': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_%d_adam_%s_aug_wd_0.0001_%s_%d_%s%s_bic_epochs_2_w_0.1_ratio_0.1_aug%s' % (
                dataset, class_order, i + 1, inc_protocol, network, num_epochs, str(base_lr), exemplar_selection,
                num_exemplars, exemplar_budget,
                '_lwf_1.0_temp_2.0_adj_w' if use_lwf else '', suffix) for i in
            range(num_order)
        ],
        'IL2M': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_%d_adam_%s_aug_wd_0.0001_%s_%d_%s_il2m%s' % (
                dataset, class_order, i + 1, inc_protocol, network, num_epochs, str(base_lr), exemplar_selection,
                num_exemplars, exemplar_budget, suffix) for
            i in
            range(num_order)
        ],
        'WA': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_%d_adam_%s_aug_wd_0.0001_%s_%d_%s%s_weight_aligning%s' % (
                dataset, class_order, i + 1, inc_protocol, network, num_epochs, str(base_lr), exemplar_selection,
                num_exemplars, exemplar_budget,
                '_lwf_1.0_temp_2.0_adj_w' if use_lwf else '', suffix) for i in
            range(num_order)
        ],
        'Post-scaling': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_%d_adam_%s_aug_wd_0.0001_%s_%d_%s%s_post_scaling%s' % (
                dataset, class_order, i + 1, inc_protocol, network, num_epochs, str(base_lr), exemplar_selection,
                num_exemplars, exemplar_budget,
                '_lwf_1.0_temp_2.0' if use_lwf else '', suffix) for i in
            range(num_order)
        ],
    }

    keys = ['iCaRL', 'EEIL', 'LSIL', 'IL2M', 'WA', 'Post-scaling']
    print('[DATASET] %s' % display_dataset)
    print('[ORDER] %s' % class_order)
    vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, MIDDLE_FOLDER='result/%s' % stage)
