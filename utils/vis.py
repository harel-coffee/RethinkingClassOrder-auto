# -*- coding:utf-8 -*-  

""" 
@time: 12/17/19 9:22 PM 
@author: Chen He 
@site:  
@file: vis.py
@description:  
"""

import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
from functools import partial

PROJ_DIR = os.path.dirname(os.path.dirname(__file__))

color_map = {
    'baseline': '#e6194B',
    'baseline+mask': '#ffe119',
    'baseline+mask+ft': '#f58231',
    'EEIL [ECCV18]': '#3cb44b',
    'upperbound': '#4363d8',
    'hiera': '#911eb4',

    'conv1': '#e6194B',
    'conv2': '#3cb44b',
    'fc3': '#ffe119',
}

color_list = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
              '#fabebe', '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
              '#000075', '#a9a9a9', '#ffffff', '#000000']


def vis_loss(data_dict, result_dir, filename):
    keys = sorted(data_dict.keys())
    values = [data_dict[key] for key in keys]

    plt.figure()
    plt.xlabel('Step')
    plt.ylabel(filename)
    plt.plot(keys, values, marker='.')
    plt.savefig(os.path.join(result_dir, '%s.pdf' % filename))
    plt.close()


def vis_conf_mat(test_conf_mat_filename, class_names, all_classes, vmax):
    test_conf_mat = np.load(test_conf_mat_filename)
    plt.figure()
    sns.heatmap(test_conf_mat, vmax=vmax, square=True, xticklabels=[class_names[i] for i in all_classes],
                yticklabels=[class_names[i] for i in all_classes])
    plt.savefig(os.path.splitext(test_conf_mat_filename)[0] + '.pdf')
    plt.close()


def vis_final_reordered_conf_mat(test_conf_mat_filename, class_names, all_classes_idx, vmax, all_classes, order_subset,
                                 dataset='imagenet'):
    if dataset == 'imagenet':
        if order_subset == '10x10':
            reference_order_file = os.path.join(PROJ_DIR, 'datasets/imagenet/order_10x10_group_1_wnid.txt')
        else:
            raise Exception()
    elif dataset == 'inat':
        reference_order_file = os.path.join(PROJ_DIR, 'datasets/inat/order_%s_group_1.txt' % order_subset)
    else:
        print('Do not visualize the confusion matrix')
        return

    reference_order = []
    with open(reference_order_file, 'r') as fin:
        for line in fin.readlines():
            wnid = line.strip()
            reference_order.append(wnid)

    reorder = []
    for wnid in reference_order:
        reorder.append(all_classes.index(wnid))
    reorder = np.array(reorder)

    test_conf_mat = np.load(test_conf_mat_filename)
    plt.figure(figsize=(15, 15))
    reordered_conf_mat = test_conf_mat[reorder][:, reorder]
    reordered_class_names = [class_names[i].split(', ')[0] for i in reorder]
    cmap = 'jet'
    sns.heatmap(reordered_conf_mat, vmax=vmax, square=True,
                xticklabels=[reordered_class_names[i] for i in all_classes_idx],
                yticklabels=[reordered_class_names[i] for i in all_classes_idx],
                cmap=cmap)
    plt.savefig(os.path.splitext(test_conf_mat_filename)[0] + '_reorder_%s.pdf' % cmap)
    plt.close()

    if dataset == 'imagenet' and order_subset == '10x10':
        internal_conf = 0
        external_conf = 0
        for i in range(len(reordered_conf_mat)):
            for j in range(len(reordered_conf_mat)):
                if i == j:
                    continue

                if i // 10 == j // 10:
                    internal_conf += reordered_conf_mat[i][j]
                else:
                    external_conf += reordered_conf_mat[i][j]
        total_conf = internal_conf + external_conf
        with open(os.path.splitext(test_conf_mat_filename)[0] + '_conf_stats.txt', 'w') as fout:
            fout.write('Total samples: %d' % np.sum(reordered_conf_mat) + os.linesep)
            fout.write('Total conf: %d' % total_conf + os.linesep)
            fout.write('Internal conf: %d' % internal_conf + os.linesep)
            fout.write('External conf: %d' % external_conf + os.linesep)
        print('Total conf: %d (int: %d, ext: %d)' % (total_conf, internal_conf, external_conf))


def vis_acc_curve(args, middle_folder, type):
    filename_dict = {
        'top1': 'top1_acc',
        'top5': 'top5_acc',
        'harmonic_mean': 'harmonic_mean',
    }

    name_dict = {
        'top1': 'Top-1 Accuracy',
        'top5': 'Top-5 Accuracy',
        'harmonic_mean': 'Harmonic Mean',
    }

    filename = filename_dict[type]
    name = name_dict[type]

    output_result_folder = os.path.join(args.OUTPUT_FOLDER, 'result_curve', middle_folder)
    if not os.path.exists(output_result_folder):
        os.makedirs(output_result_folder)

    # load
    accs = []
    for group_idx in range((args.total_cl - args.base_cl) // args.nb_cl + 1):
        result_filename = os.path.join(args.OUTPUT_FOLDER, 'group_%d' % (group_idx + 1), 'result', middle_folder,
                                       '%s.txt' % filename)
        if not os.path.exists(result_filename):
            break
        acc = float(open(result_filename, 'r').readline().strip())
        accs.append(acc)
    accs = np.array(accs)

    # txt
    with open(os.path.join(output_result_folder, '%s_curve.txt' % filename), 'w') as fout:
        for acc in accs:
            fout.write('%.2f' % acc + os.linesep)

    # plot
    plt.figure()
    plt.title('%s %s Curve' % (args.dataset, name))
    plt.xlabel('#Classes')
    plt.ylabel('Accuracy')
    plt.plot(range(args.base_cl, args.total_cl + args.nb_cl, args.nb_cl)[:len(accs)], accs, marker='.')
    for a, b in zip(range(args.base_cl, args.total_cl + args.nb_cl, args.nb_cl)[:len(accs)], accs):
        plt.text(a, b + 0.01, '%.2f' % b, ha='center', va='bottom', fontsize=9)
    plt.savefig(os.path.join(output_result_folder, '%s_curve.pdf' % filename))
    plt.close()


vis_top1_acc_curve = partial(vis_acc_curve, type='top1')
vis_top5_acc_curve = partial(vis_acc_curve, type='top5')
vis_harmonic_mean_curve = partial(vis_acc_curve, type='harmonic_mean')


def vis_old_new_acc_curve(result_dir, accs, old_accs, new_accs):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if isinstance(accs, dict):
        epochs = sorted(accs.keys())
    else:
        epochs = range(len(accs))

    assert len(old_accs) == len(new_accs)
    plt.figure()
    plt.xlabel('#Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, [accs[epoch] for epoch in epochs], marker='.', label='avg', color='#3574B2')
    if len(old_accs) > 0:
        plt.plot(epochs, [old_accs[epoch] for epoch in epochs], marker='.', label='old', color='#36A221')
        plt.plot(epochs, [new_accs[epoch] for epoch in epochs], marker='.', label='new', color='#F67D06')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'old_new_acc_curve.pdf'))
    plt.close()


def calc_mean_std(result):
    mean_result = np.mean(result, axis=0)
    std_result = np.std(result, axis=0)
    return mean_result, std_result


def vis_multiple(result_dir_dict, total_cl, nb_cl, keys, output_name, title='Group ImageNet', MIDDLE_FOLDER='', ylim=None):
    fontsize = 14

    x = [i + nb_cl for i in range(0, total_cl, nb_cl)]
    x_names = [str(i) for i in x]
    y = range(0, 110, 10)
    y_names = [str(i) + '%' for i in range(0, 110, 10)]

    plt.figure(figsize=(6, 6), dpi=220)

    plt.gca().set_autoscale_on(False)

    plt.xlim(0, total_cl)
    plt.ylim(0, 100)

    plt.xticks(x, x_names, rotation=45, fontsize=fontsize)
    plt.yticks(y, y_names, fontsize=fontsize)
    plt.margins(0)

    plt.xlabel("Number of classes", fontsize=fontsize)
    plt.ylabel("Accuracy", fontsize=fontsize)
    plt.title(title)

    # Horizontal reference lines
    for i in range(10, 100, 10):
        plt.hlines(i, 0, total_cl, colors="lightgray", linestyles="dashed")

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
        try:
            plt.errorbar(x[:len(y_mean)], y_mean, yerr=y_std, marker='.', label=key, color=color_map[key])
        except:
            plt.errorbar(x[:len(y_mean)], y_mean, yerr=y_std, marker='.', label=key, color=color_list[key_idx])

        print('%s: %.2f' % (key, y_mean[-1]))

    plt.legend(fontsize=fontsize)
    if ylim is not None:
        plt.ylim(ylim)

    plt.savefig('%s.pdf' % output_name)


def get_running_time(result_dir_dict, total_cl, nb_cl, keys):
    for key_idx, key in enumerate(keys):

        result_dirs = result_dir_dict[key]
        aver_acc_over_time_mul = []

        for result_dir in result_dirs:
            time_arr = []
            for group_idx in range((total_cl - nb_cl) // nb_cl + 1):
                time_filename = os.path.join(result_dir, 'stat_time.txt')
                if not os.path.exists(time_filename):
                    break
                times = [float(line.strip()) for line in open(time_filename, 'r').readlines()]
                time = np.mean(times)
                time_arr.append(time)
            time_arr = np.array(time_arr)
            aver_acc_over_time_mul.append(time_arr)

        y_mean, y_std = calc_mean_std(np.array(aver_acc_over_time_mul))
        print('%s: %.2f' % (key, y_mean[-1]))


def vis_multiple_harmonic_mean(result_dir_dict, total_cl, nb_cl, keys, output_name):
    fontsize = 14

    x = [i + nb_cl for i in range(0, total_cl, nb_cl)]
    x_names = [str(i) for i in x]
    y = range(0, 110, 10)
    y_names = [str(i) + '%' for i in range(0, 110, 10)]

    plt.figure(figsize=(8, 8), dpi=220)

    plt.gca().set_autoscale_on(False)

    plt.xlim(0, total_cl)
    plt.ylim(0, 100)

    plt.xticks(x, x_names, rotation=45, fontsize=fontsize)
    plt.yticks(y, y_names, fontsize=fontsize)
    plt.margins(0)

    plt.xlabel("Number of classes", fontsize=fontsize)
    plt.ylabel("Accuracy", fontsize=fontsize)

    # Horizontal reference lines
    for i in range(10, 100, 10):
        plt.hlines(i, 0, total_cl, colors="lightgray", linestyles="dashed")

    for key_idx, key in enumerate(keys):

        result_dirs = result_dir_dict[key]
        aver_acc_over_time_mul = []

        for result_dir in result_dirs:
            avg_accs = []
            for group_idx in range((total_cl - nb_cl) // nb_cl + 1):
                conf_mat_filename = os.path.join(result_dir, 'group_%d' % (group_idx + 1), 'conf_mat.npy')
                if not os.path.exists(conf_mat_filename):
                    break
                conf_mat = np.load(conf_mat_filename)
                accs = np.diag(conf_mat) * 100. / np.sum(conf_mat, axis=1)
                if group_idx > 0:
                    old_avg_acc, new_avg_acc = np.mean(accs[:group_idx * nb_cl]), np.mean(accs[group_idx * nb_cl:])
                    avg_acc = (2 * old_avg_acc * new_avg_acc / (old_avg_acc + new_avg_acc))  # harmonic mean
                else:
                    avg_acc = np.mean(accs)
                avg_accs.append(avg_acc)
            avg_accs = np.array(avg_accs)
            aver_acc_over_time_mul.append(avg_accs)

        y_mean, y_std = calc_mean_std(np.array(aver_acc_over_time_mul))
        try:
            plt.errorbar(x[:len(y_mean)], y_mean, yerr=y_std, marker='.', label=key, color=color_map[key])
        except:
            plt.errorbar(x[:len(y_mean)], y_mean, yerr=y_std, marker='.', label=key, color=color_list[key_idx])

    plt.legend(fontsize=fontsize)

    plt.savefig('%s_harmonic_mean.pdf' % output_name)


def vis_multiple_top5(result_dir_dict, total_cl, nb_cl, keys, output_name):
    fontsize = 14

    x = [i + nb_cl for i in range(0, total_cl, nb_cl)]
    x_names = [str(i) for i in x]
    y = range(0, 110, 10)
    y_names = [str(i) + '%' for i in range(0, 110, 10)]

    plt.figure(figsize=(8, 8), dpi=220)

    plt.gca().set_autoscale_on(False)

    plt.xlim(0, total_cl)
    plt.ylim(0, 100)

    plt.xticks(x, x_names, rotation=45, fontsize=fontsize)
    plt.yticks(y, y_names, fontsize=fontsize)
    plt.margins(0)

    plt.xlabel("Number of classes", fontsize=fontsize)
    plt.ylabel("Top-5 Accuracy", fontsize=fontsize)

    # Horizontal reference lines
    for i in range(10, 100, 10):
        plt.hlines(i, 0, total_cl, colors="lightgray", linestyles="dashed")

    for key_idx, key in enumerate(keys):

        result_dirs = result_dir_dict[key]
        aver_acc_over_time_mul = []

        for result_dir in result_dirs:
            accs = []
            for group_idx in range((total_cl - nb_cl) // nb_cl + 1):
                top5_acc_filename = os.path.join(result_dir, 'group_%d' % (group_idx + 1), 'top5_acc.txt')
                if not os.path.exists(top5_acc_filename):
                    break
                with open(top5_acc_filename, 'r') as fin:
                    acc = float(fin.read())
                    accs.append(acc)
            accs = np.array(accs)
            aver_acc_over_time_mul.append(accs)

        y_mean, y_std = calc_mean_std(np.array(aver_acc_over_time_mul))
        try:
            plt.errorbar(x[:len(y_mean)], y_mean, yerr=y_std, marker='.', label=key, color=color_map[key])
        except:
            plt.errorbar(x[:len(y_mean)], y_mean, yerr=y_std, marker='.', label=key, color=color_list[key_idx])

    plt.legend(fontsize=fontsize)

    plt.savefig('%s_top5.pdf' % output_name)


if __name__ == '__main__':
    result_dir_dict = {
        'baseline': [
            '../result/cifar-100_1/base_10_inc_10/inner_product_cnn/lenet_70_adam_0.001000_random_20_total_lwf_1.0_temp_2.0_aug_wd_0.001/',
        ],
        'baseline+mask': [
            '../result/cifar-100_1/base_10_inc_10/mask_inner_product_cnn/lenet_70_adam_0.001000_random_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_mask_0.75/',
        ],
        'baseline+mask+ft': [
            '../result/cifar-100_1/base_10_inc_10/mask_kb_inner_product_cnn/lenet_70_adam_0.001000_random_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_mask_0.75/',
        ],
        'EEIL [ECCV18]': [
            '../result/cifar-100_1/base_10_inc_10/eeil/lenet_70_adam_0.001000_random_20_total_lwf_1.0_temp_2.0_aug_wd_0.001/',
        ],
        'upperbound': [
            '../result/cifar-100_1/base_10_inc_10/joint_training/lenet_70_adam_0.001000_aug_wd_0.001/',
        ],
    }
    keys = ['baseline', 'EEIL [ECCV18]', 'baseline+mask', 'baseline+mask+ft', 'upperbound']
    total_cl, nb_cl = 100, 10
    output_name = 'cifar-100_from_%d_0216' % nb_cl
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)

    # vis_mask(
    #     '../result/cifar-100_1/base_10_inc_10/mask_inner_product_cnn/lenet_70_adam_0.001000_random_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_mask_0.75/')

    result_dir_dict = {
        'sigmoid (0.1)': [
            '../result/cifar-100_1/base_10_inc_10_total_30/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_0.1_temp_2.0_sigmoid_reweight_aug_wd_0.001_sp_init_0.5/',
        ],
        'softmax (0.1)': [
            '../result/cifar-100_1/base_10_inc_10_total_30/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_0.1_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5/',
        ],
        'sigmoid (0.2)': [
            '../result/cifar-100_1/base_10_inc_10_total_30/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_0.2_temp_2.0_sigmoid_reweight_aug_wd_0.001_sp_init_0.5/',
        ],
        'softmax (0.2)': [
            '../result/cifar-100_1/base_10_inc_10_total_30/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_0.2_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5/',
        ],
        'sigmoid (0.5)': [
            '../result/cifar-100_1/base_10_inc_10_total_30/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_0.5_temp_2.0_sigmoid_reweight_aug_wd_0.001_sp_init_0.5/',
        ],
        'softmax (0.5)': [
            '../result/cifar-100_1/base_10_inc_10_total_30/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_0.5_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5/',
        ],
        'sigmoid (1.0)': [
            '../result/cifar-100_1/base_10_inc_10_total_30/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_sigmoid_reweight_aug_wd_0.001_sp_init_0.5/',
        ],
        'softmax (1.0)': [
            '../result/cifar-100_1/base_10_inc_10_total_30/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5/',
        ],
    }
    keys = ['sigmoid (0.1)', 'sigmoid (0.2)', 'sigmoid (0.5)', 'sigmoid (1.0)',
            'softmax (0.1)', 'softmax (0.2)', 'softmax (0.5)', 'softmax (1.0)']
    total_cl, nb_cl = 30, 10
    output_name = 'cifar-100_0222'
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)

    result_dir_dict = {
        'sigmoid': [
            '../result/cifar-100_1/base_10_inc_10_total_30/eeil/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_sigmoid_reweight_wd_0.001',
        ],
        'softmax': [
            '../result/cifar-100_1/base_10_inc_10_total_30/eeil/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_wd_0.001',
        ],
    }
    keys = ['sigmoid', 'softmax']
    total_cl, nb_cl = 30, 10
    output_name = 'cifar-100_0222_eeil'
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)

    '''
    2020/03/11
    '''
    result_dir_dict = {
        'baseline': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_random_20_total_lwf_1.0_temp_2.0_aug_wd_0.001/',
        ],
        'hiera': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/lenet_70_adam_0.001000_random_20_total_lwf_1.0_temp_2.0_aug_wd_0.001/',
        ],
        'baseline+mask': [
            '../result/cifar-100_1/base_10_inc_10_total_100/mask_inner_product_cnn/lenet_70_adam_0.001000_random_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_mask_0.75/',
        ],
        'baseline+mask+ft': [
            '../result/cifar-100_1/base_10_inc_10_total_100/mask_kb_inner_product_cnn/lenet_70_adam_0.001000_random_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_mask_0.75/',
        ],
        'EEIL [ECCV18]': [
            '../result/cifar-100_1/base_10_inc_10_total_100/eeil/lenet_70_adam_0.001000_random_20_total_lwf_1.0_temp_2.0_aug_wd_0.001/',
        ],
        'upperbound': [
            '../result/cifar-100_1/base_10_inc_10_total_100/joint_training/lenet_70_adam_0.001000_aug_wd_0.001/',
        ],
    }
    keys = ['baseline', 'EEIL [ECCV18]', 'hiera', 'baseline+mask', 'baseline+mask+ft', 'upperbound']
    total_cl, nb_cl = 100, 10
    output_name = 'cifar-100_from_%d_0311' % nb_cl
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)

    '''
    2020/03/12
    '''
    result_dir_dict = {
        'hiera (no sc)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_20_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001/',
        ],
        'hiera (sc 1.0)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_20_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_with_sc_loss_1.0',
        ],
        'hiera (sc 0.5)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_20_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_with_sc_loss_0.5',
        ],
        'hiera (sc 0.2)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_20_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_with_sc_loss_0.2',
        ],
        'hiera (sc 0.1)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_20_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_with_sc_loss_0.1',
        ],
        'plain': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_20_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_no_coarse',
        ],
    }
    keys = ['hiera (no sc)', 'hiera (sc 1.0)', 'hiera (sc 0.5)', 'hiera (sc 0.2)', 'hiera (sc 0.1)', 'plain']
    total_cl, nb_cl = 30, 10
    output_name = 'cifar-100_from_%d_0313' % nb_cl
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)
    # vis_multiple_top5(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)

    '''
    2020/03/12
    '''
    result_dir_dict = {
        'hiera (super 10)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_10_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001/',
        ],
        'hiera (super 10, sc)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_10_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_with_sc_loss_1.0/',
        ],
        # 'hiera (super 10, no coarse)': [
        #     '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_10_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_no_coarse/',
        # ],
        'hiera (super 15)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_15_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001/',
        ],
        'hiera (super 15, sc)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_15_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_with_sc_loss_1.0/',
        ],
        # 'hiera (super 15, no coarse)': [
        #     '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_15_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_no_coarse',
        # ],
        'hiera (super 20)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_20_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001/',
        ],
        'hiera (super 20, sc)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_20_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_with_sc_loss_1.0/',
        ],
        # 'hiera (super 20, no coarse)': [
        #     '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_20_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_no_coarse',
        # ],
        'naive': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5/'
        ],
    }
    # keys = ['hiera (super 10)', 'hiera (super 10, no coarse)', 'hiera (super 15)', 'hiera (super 15, no coarse)',
    #         'hiera (super 20)', 'hiera (super 20, no coarse)']
    keys = ['hiera (super 10)', 'hiera (super 10, sc)', 'hiera (super 15)', 'hiera (super 15, sc)',
            'hiera (super 20)', 'hiera (super 20, sc)', 'naive']
    total_cl, nb_cl = 100, 10
    output_name = 'cifar-100_from_%d_0316_sc' % nb_cl
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)
    # vis_multiple_top5(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)

    '''
    2020/03/12
    '''
    result_dir_dict = {
        'hiera (super 10, sc)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_10_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_with_sc_loss_1.0/',
        ],
        'hiera (super 15, sc)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_15_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_with_sc_loss_1.0/',
        ],
        'hiera (super 20, sc)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera_inner_product_cnn/superclass_20_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_with_sc_loss_1.0/',
        ],
        'hiera2 (super 20)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/hiera2_inner_product_cnn/superclass_20_lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_coarse_T_1.0/',
        ],
        'naive': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5/'
        ],
    }
    keys = ['hiera (super 10, sc)', 'hiera (super 15, sc)', 'hiera (super 20, sc)', 'hiera2 (super 20)', 'naive']
    total_cl, nb_cl = 100, 10
    output_name = 'cifar-100_from_%d_0319_sc' % nb_cl
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)
    # vis_multiple_top5(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)

    # vis_mask(
    #     '/home/hechen/projects/SPCIL_cvpr21/result/cifar-100_1/base_10_inc_10_total_100/mask_inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5_mask_0.75/')

    '''
    2020/03/12
    '''
    result_dir_dict = {
        'wup 0.1': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn_sdm/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5_sdm_wordnet_wup_0.1_2',
        ],
        'wup 0.2': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn_sdm/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5_sdm_wordnet_wup_0.2_2',
        ],
        'wup 0.5': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn_sdm/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5_sdm_wordnet_wup_0.5_2',
        ],
        'wup 1.0': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn_sdm/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5_sdm_wordnet_wup_1.0_2',
        ],
        'naive': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5/'
        ],
    }
    keys = ['wup 0.1', 'wup 0.2', 'wup 0.5', 'wup 1.0', 'naive']
    total_cl, nb_cl = 100, 10
    output_name = 'cifar-100_from_%d_0324_sdm' % nb_cl
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)
    # vis_multiple_top5(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)

    # 2020/03/27
    result_dir_dict = {
        'naive': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5/'
        ],
        'naive (reweight)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5/'
        ],
        'naive (thresholding)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5_post_scale/'
        ],
        'naive (reweight+thresholding)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5_post_scale/'
        ],
        'naive (oversample)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_oversample_aug_wd_0.001_sp_init_0.5/'
        ],
        'naive (mix-up, 0.2)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_mixup_0.2_aug_wd_0.001_sp_init_0.5/'
        ],
    }
    keys = ['naive', 'naive (reweight)', 'naive (thresholding)', 'naive (reweight+thresholding)', 'naive (oversample)',
            'naive (mix-up, 0.2)']
    total_cl, nb_cl = 100, 10
    output_name = 'cifar-100_from_%d_0401_imbalanced' % nb_cl
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)
    # vis_multiple_harmonic_mean(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)
    # vis_multiple_top5(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)

    # 2020/04/23 aug
    result_dir_dict = {
        'naive': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_wd_0.001_sp_init_0.5/'
        ],
        'naive (aug)': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5/'
        ],
    }
    keys = ['naive', 'naive (aug)']
    total_cl, nb_cl = 100, 10
    output_name = 'cifar-100_from_%d_0423_aug' % nb_cl
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)

    result_dir_dict = {
        'naive': [
            '../result/imagenet_10x10_random_1/base_10_inc_10_total_100/inner_product_cnn/resnet_70_0.001000_fixed_20_total_wd_0.0001_lwf_1.0_temp_2.0_post_scale/'
        ],
        'naive (aug)': [
            '../result/imagenet_10x10_random_1/base_10_inc_10_total_100/inner_product_cnn/resnet_70_0.001000_fixed_20_total_aug_wd_0.0001_lwf_1.0_temp_2.0_post_scale/'
        ],
    }
    keys = ['naive', 'naive (aug)']
    total_cl, nb_cl = 100, 10
    output_name = 'imagenet_from_%d_0423_aug' % nb_cl
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name,
    #              title='Group ImageNet')

    # 2020/04/23 exemplar selection
    result_dir_dict = {
        'herding': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_herding_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5/'
        ],
        'highest': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_highest_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5/'
        ],
        'lowest': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_lowest_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5/'
        ],
        'highlow': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_highlow_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5/'
        ],
        'feat_kmedoids': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_feat_kmedoids_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5/'
        ],
        'ori_kmedoids': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_ori_kmedoids_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5/'
        ],
        'ori_kmeans': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_ori_kmeans_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5/'
        ],
        'fixed': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5/'
        ],
        'random': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_random_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5/'
        ],
    }
    keys = ['herding', 'highest', 'lowest', 'highlow', 'feat_kmedoids', 'ori_kmedoids', 'ori_kmeans', 'fixed', 'random']
    total_cl, nb_cl = 100, 10
    output_name = 'cifar-100_from_%d_0423_exemplar_selection' % nb_cl
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)

    # 2020/04/27 exemplar selection
    result_dir_dict = {
        'herding': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_herding_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5_post_scale/'
        ],
        'highest': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_highest_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5_post_scale/'
        ],
        'lowest': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_lowest_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5_post_scale/'
        ],
        'highlow': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_highlow_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5_post_scale/'
        ],
        'feat_kmedoids': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_feat_kmedoids_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5_post_scale/'
        ],
        'ori_kmedoids': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_ori_kmedoids_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5_post_scale/'
        ],
        'ori_kmeans': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_ori_kmeans_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5_post_scale/'
        ],
        'fixed': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5_post_scale/'
        ],
        'random': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_random_20_total_lwf_1.0_temp_2.0_aug_wd_0.001_sp_init_0.5_post_scale/'
        ],
    }
    keys = ['herding', 'highest', 'lowest', 'highlow', 'feat_kmedoids', 'ori_kmedoids', 'ori_kmeans', 'fixed', 'random']
    total_cl, nb_cl = 100, 10
    output_name = 'cifar-100_from_%d_0427_exemplar_selection_post_scaling' % nb_cl
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)

    # 2020/04/23 exemplar num and strategy
    result_dir_dict = {
        '20 total': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5'
        ],
        '5 total': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_fixed_5_total_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5'
        ],
        '20 each': [
            '../result/cifar-100_1/base_10_inc_10_total_100/inner_product_cnn/lenet_70_adam_0.001000_fixed_20_each_lwf_1.0_temp_2.0_reweight_aug_wd_0.001_sp_init_0.5'
        ],
        'Joint Training': [
            '../result/cifar-100_1/base_10_inc_10_total_100/joint_training/lenet_70_adam_0.001000_aug_wd_0.001'
        ]
    }
    keys = ['20 total', '5 total', '20 each', 'Joint Training']
    total_cl, nb_cl = 100, 10
    output_name = 'cifar-100_from_%d_0423_exemplar_num' % nb_cl
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name)

    # Order on Group ImageNet
    result_dir_dict = {
        'Random': [
            '../result/imagenet64x64_10x10_random_1/base_10_inc_10_total_100/inner_product_cnn/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_random_2/base_10_inc_10_total_100/inner_product_cnn/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_random_3/base_10_inc_10_total_100/inner_product_cnn/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_random_4/base_10_inc_10_total_100/inner_product_cnn/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_random_5/base_10_inc_10_total_100/inner_product_cnn/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
        ],
        'Even': [
            '../result/imagenet64x64_10x10_even_1/base_10_inc_10_total_100/inner_product_cnn/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_even_2/base_10_inc_10_total_100/inner_product_cnn/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_even_3/base_10_inc_10_total_100/inner_product_cnn/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_even_4/base_10_inc_10_total_100/inner_product_cnn/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_even_5/base_10_inc_10_total_100/inner_product_cnn/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
        ],
        'Group': [
            '../result/imagenet64x64_10x10_group_1/base_10_inc_10_total_100/inner_product_cnn/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_group_2/base_10_inc_10_total_100/inner_product_cnn/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_group_3/base_10_inc_10_total_100/inner_product_cnn/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_group_4/base_10_inc_10_total_100/inner_product_cnn/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_group_5/base_10_inc_10_total_100/inner_product_cnn/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
        ]
    }
    keys = ['Random', 'Even', 'Group']
    total_cl, nb_cl = 100, 10
    output_name = 'group_imagenet_different_order_clsnet'
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name,
    #              MIDDLE_FOLDER='result_post_scaling', title='Group ImageNet - Classification Network',
    #              ylim=(20, 100))

    result_dir_dict = {
        'Random': [
            '../result/imagenet64x64_10x10_random_1/base_10_inc_10_total_100/icarl_cvpr17/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_random_2/base_10_inc_10_total_100/icarl_cvpr17/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_random_3/base_10_inc_10_total_100/icarl_cvpr17/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_random_4/base_10_inc_10_total_100/icarl_cvpr17/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_random_5/base_10_inc_10_total_100/icarl_cvpr17/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
        ],
        'Even': [
            '../result/imagenet64x64_10x10_even_1/base_10_inc_10_total_100/icarl_cvpr17/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_even_2/base_10_inc_10_total_100/icarl_cvpr17/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_even_3/base_10_inc_10_total_100/icarl_cvpr17/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_even_4/base_10_inc_10_total_100/icarl_cvpr17/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_even_5/base_10_inc_10_total_100/icarl_cvpr17/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
        ],
        'Group': [
            '../result/imagenet64x64_10x10_group_1/base_10_inc_10_total_100/icarl_cvpr17/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_group_2/base_10_inc_10_total_100/icarl_cvpr17/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_group_3/base_10_inc_10_total_100/icarl_cvpr17/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_group_4/base_10_inc_10_total_100/icarl_cvpr17/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
            '../result/imagenet64x64_10x10_group_5/base_10_inc_10_total_100/icarl_cvpr17/skip_first_resnet18_70_0.001000_fixed_20_total_keep_rem_aug_wd_0.0001_lwf_1.0_temp_2.0_py3',
        ]
    }
    keys = ['Random', 'Even', 'Group']
    total_cl, nb_cl = 100, 10
    output_name = 'group_imagenet_different_order_ebdnet'
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name,
    #              MIDDLE_FOLDER='result', title='Group ImageNet - Embedding Network',
    #              ylim=(20, 100))

    # dataset = 'imagenet64x64_10x10'
    # network = 'resnet18'
    # class_order = 'optimal_nb_cl_20'
    class_order = '10x10_group'
    # class_order = 'alphabetical'
    stage = '2nd_stage'
    # inc_protocol = 'base_10_inc_10_total_100'
    total_cl, nb_cl = 100, 10
    if dataset in ['imagenet64x64_10x10', 'inat_9x9']:
        num_order = 5
        if dataset == 'inat_9x9':
            num_order = 3
            total_cl, nb_cl = 81, 9

        title = 'Group ImageNet'
        if network == 'mobilenet':
            base_lr = 0.005
        elif network == 'resnet18':
            base_lr = 0.005

    inc_protocol = 'base_%d_inc_%d_total_%d' % (nb_cl, nb_cl, total_cl)

    if 'optimal' in class_order:
        num_order = 1

    result_dir_dict = {
        'Lowerbound': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001/' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'Joint Training': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_joint_training/' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'LwF': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0/' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'iCaRL': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_no_final_relu_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_embedding_cosine' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'EEIL': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_eeil_30' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'LSIL': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_adj_w_bic_epochs_2_w_0.1_ratio_0.1_aug' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'IL2M': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_il2m' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'MDFCIL': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_adj_w_weight_aligning' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'GDumb': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_undersampling_init_all' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'Re-weighting': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_reweighting' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'Post-scaling': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_post_scaling' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'Under-sampling': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_undersampling' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'Over-sampling': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_oversampling' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'SMOTE': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_smote' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'ADASYN': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_adasyn' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'K-means': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_cluster_centroids' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'K-medoids': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_cluster_centroids_v3' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],

    }
    keys = ['Lowerbound', 'LwF', 'iCaRL', 'EEIL', 'LSIL', 'IL2M', 'MDFCIL', 'Post-scaling', 'Joint Training']
    # keys = ['LwF', 'iCaRL', 'EEIL', 'LSIL', 'IL2M', 'MDFCIL', 'GDumb', 'Re-weighting', 'Post-scaling', 'Under-sampling',
    #         'Over-sampling', 'SMOTE', 'ADASYN', 'K-means', 'K-medoids']
    # keys = ['LwF', 'iCaRL', 'EEIL', 'LSIL', 'IL2M', 'MDFCIL', 'GDumb', 'Re-weighting', 'Under-sampling']
    output_name = '%s_%s_%s' % (dataset, class_order, network)
    print('%s %s: %s (%s)' % (dataset, class_order, network, stage))
    vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name,
                 MIDDLE_FOLDER='result/%s' % stage, title=title, ylim=(10, 100))
    # get_running_time(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys)

    dataset = 'imagenet64x64_10x10'
    network = 'resnet18'
    class_order = 'random'
    stage = '2nd_stage'
    inc_protocol = 'base_10_inc_10_total_100'
    if dataset in ['imagenet64x64_10x10', 'inat_9x9']:
        num_order = 3
        if dataset == 'inat_9x9':
            num_order = 1
            inc_protocol = 'base_9_inc_9_total_81'

        title = 'Group ImageNet'
        if network == 'mobilenet':
            base_lr = 0.005
        elif network == 'resnet18':
            base_lr = 0.005

    result_dir_dict = {
        'MDFCIL (0.2)': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_rsc_0.2_random_20_total_lwf_1.0_temp_2.0_adj_w_weight_aligning' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'MDFCIL (0.5)': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_rsc_0.5_random_20_total_lwf_1.0_temp_2.0_adj_w_weight_aligning' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'MDFCIL (0.8)': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_rsc_0.8_random_20_total_lwf_1.0_temp_2.0_adj_w_weight_aligning' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'MDFCIL': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_adj_w_weight_aligning' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'Post-scaling (0.2)': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_rsc_0.2_random_20_total_lwf_1.0_temp_2.0_post_scaling' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'Post-scaling (0.5)': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_rsc_0.5_random_20_total_lwf_1.0_temp_2.0_post_scaling' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'Post-scaling (0.8)': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_rsc_0.8_random_20_total_lwf_1.0_temp_2.0_post_scaling' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'Post-scaling': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_post_scaling' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
    }
    keys = ['MDFCIL (0.2)', 'MDFCIL (0.5)', 'MDFCIL (0.8)', 'MDFCIL', 'Post-scaling (0.2)', 'Post-scaling (0.5)',
            'Post-scaling (0.8)', 'Post-scaling']
    total_cl, nb_cl = 100, 10
    output_name = '%s_%s_%s' % (dataset, class_order, network)
    # print('%s %s: %s (%s)' % (dataset, class_order, network, stage))
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name,
    #              MIDDLE_FOLDER='result/%s' % stage, title=title, ylim=(10, 100))
    # get_running_time(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys)

    # AwA2
    dataset = 'awa2'
    network = 'resnet18'
    class_order = 'random'
    stage = '2nd_stage'
    total_cl, nb_cl = 50, 10
    base_lr = 0.001
    num_order = 3
    title = 'AwA2'
    inc_protocol = 'base_%d_inc_%d_total_%d' % (nb_cl, nb_cl, total_cl)

    result_dir_dict = {
        'MDFCIL': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_adj_w_weight_aligning' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ],
        'Post-scaling': [
            '../result/%s_%s_%d/%s/seed_1993/skip_first_%s_fc_70_adam_%s_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_post_scaling' % (
                dataset, class_order, i + 1, inc_protocol, network, str(base_lr)) for i in range(num_order)
        ]

    }
    keys = ['MDFCIL', 'Post-scaling']
    output_name = '%s_%s_%s' % (dataset, class_order, network)
    # print('%s %s: %s (%s)' % (dataset, class_order, network, stage))
    # vis_multiple(result_dir_dict, total_cl=total_cl, nb_cl=nb_cl, keys=keys, output_name=output_name,
    #              MIDDLE_FOLDER='result/%s' % stage, title=title, ylim=(10, 50))
