# -*- coding:utf-8 -*-  

""" 
@time: 11/30/20 10:09 PM 
@author: Chen He 
@site:  
@file: main.py.py
@description:  
"""

import argparse
import os
import pickle
import random
import sys
import time
from pprint import pprint

import numpy as np
import scipy
import tensorflow as tf
from nltk.corpus import wordnet as wn
from tqdm import tqdm

from datasets.dataset import CILProtocol
from networks.backbones.imagenet64x64_resnet import resnet18 as resnet18_64x64

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Class Incremental Learning')

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--resolution', '-r', type=str, default='64x64')
    parser.add_argument('--order_subset', type=str, default='10x10')
    parser.add_argument('--order_type', type=str, default='random')
    parser.add_argument('--order_idx', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=-1)

    # incremental protocol
    parser.add_argument('--base_cl', type=int, default=10)
    parser.add_argument('--nb_cl', type=int, default=10)
    parser.add_argument('--total_cl', type=int, default=100)
    parser.add_argument('--to_group_idx', type=int, default=-1)
    parser.add_argument('--random_seed', type=int, default=1993)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--semantic', action='store_true')
    parser.add_argument('--semantic_type', type=str, default='wordnet_wup', choices=['wordnet_wup'])

    parser.add_argument('--dist_type', type=str, default='wdist')
    parser.add_argument('--feat_type', type=str, default='ssl', choices=['ssl', 'supervised'])

    # debug switch
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test_final_only', action='store_true')

    args = parser.parse_args()

    '''
    Check arguments
    '''
    tmp_order_type = args.order_type
    args.order_type = 'random'  # use 'random' order to calculate the distance matrix
    extra_suffix = ''
    if args.dataset == 'imagenet':
        if args.order_subset == '10x10':
            args.base_cl = args.total_cl = 100
        else:
            raise Exception()
    elif args.dataset == 'inat':
        args.resolution = '64x64'
        args.order_subset = '9x9'
        args.base_cl = args.total_cl = 81
    else:
        raise Exception()
    extra_suffix = 'nb_cl_%d' % args.nb_cl
    args.joint_training = False

    # print hyperparameters
    pprint(vars(args), width=1)

    # start time
    program_start_wall_time = time.time()

    # some settings
    args.AUTOTUNE = tf.data.experimental.AUTOTUNE
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.debug:
        tf.config.experimental_run_functions_eagerly(True)

    # random seed
    random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)

    '''
    CIL Training
    '''

    # init dataset
    if args.dataset == 'imagenet':
        if args.resolution == '64x64':
            from datasets.imagenet64x64 import ImageNet64x64

            dataset = ImageNet64x64(args)
        else:
            raise Exception('Invalid resolution')
    elif args.dataset == 'inat':
        from datasets.inaturalist64x64 import INat64x64

        dataset = INat64x64(args)
    else:
        raise Exception('Invalid dataset name')

    # check exists
    output_dir = 'calculated_transferability'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    setting_str = 'semantic_%s' % args.semantic_type if args.semantic else '%s_%s' % (args.feat_type, args.dist_type)
    output_file = os.path.join(output_dir,
                               '%s64x64_%s_%d_%s_%s.txt' % (
                                   args.dataset, tmp_order_type, args.order_idx, extra_suffix, setting_str))

    if os.path.exists(output_file):
        print('File exists: %s' % output_file)
        with open(output_file, 'r') as fin:
            total_trans = float(fin.readline().strip())
            print('Total trans: %f' % total_trans)
        sys.exit(0)

    # init protocol
    tmp_order_idx = args.order_idx
    args.order_idx = 1
    protocol = CILProtocol(args, dataset)

    dist_folder = 'dists'
    if not os.path.exists(dist_folder):
        os.makedirs(dist_folder)

    dist_mat_file = os.path.join(dist_folder, '%s_%s_%s_%d_%s.pkl' % (
        args.dataset, args.order_subset, args.resolution, args.total_cl, setting_str))

    if not os.path.exists(dist_mat_file):

        if not args.semantic:
            feat_extractor = resnet18_64x64()
            if args.dataset == 'imagenet':
                if args.feat_type == 'ssl':
                    feat_extractor.load_weights(
                        'result/imagenet64x64_10x10_random_1/base_100/seed_1993/ssl_resnet18_fc_70_adam_0.005_aug_wd_0.0001/group_1/checkpoints/1st_stage/final')
                elif args.feat_type == 'supervised':
                    feat_extractor.load_weights(
                        'result/imagenet64x64_10x10_random_1/base_100/seed_1993/resnet18_fc_70_adam_0.005_aug_wd_0.0001/group_1/checkpoints/2nd_stage/final')
            elif args.dataset == 'inat':
                if args.feat_type == 'ssl':
                    feat_extractor.load_weights(
                        'result/inat64x64_9x9_random_1/base_81/seed_1993/ssl_resnet18_fc_70_adam_0.005_aug_wd_0.0001/group_1/checkpoints/1st_stage/final')
                elif args.feat_type == 'supervised':
                    feat_extractor.load_weights(
                        'result/inat_9x9_random_1/base_81/seed_1993/resnet18_fc_70_adam_0.005_aug_wd_0.0001/group_1/checkpoints/2nd_stage/final')
                else:
                    raise Exception()
            else:
                raise Exception()

            cur_means = []
            cur_covs = []
            cur_covs_sqrtm = []
            cur_logdet_covs = []
            cur_inv_covs = []
            cur_log_covs = []

            cumul_sum_dist = []
            cumul_min_dist = []

            # start incremental process
            class_inc = protocol.__next__()

            print('Group index: %d' % (class_inc.group_idx + 1))
            inc_start_time = time.time()

            train_dataset = class_inc.train_dataset
            cur_dataset = train_dataset.batch(args.batch_size)

            x = feat_extractor.predict(cur_dataset)
            y = np.stack(class_inc.train_dataset.map(lambda a, b: b))

            print('Features calculated!')

            # means and covs
            for i in tqdm(sorted(set(y))):
                probe_feats_cur = x[y == i]

                mu = np.mean(probe_feats_cur, axis=0, dtype=np.float64)
                cur_means.append(mu)

                cov = np.cov(probe_feats_cur.transpose())
                cov += 1e-6 * np.diag(np.ones(len(cov)))  # prevent det(cov) becomes negative
                cur_covs.append(cov)

                if args.dist_type == 'wdist':
                    cur_covs_sqrtm.append(tf.linalg.sqrtm(cov))

                if args.dist_type == 'mdled':
                    cur_inv_covs.append(tf.linalg.inv(cov))

                    w, v = np.linalg.eigh(cov)
                    log_cov = v.dot(np.diag(np.log(w))).dot(v.T)
                    cur_log_covs.append(log_cov)

            dist_mat = np.zeros([len(cur_means), len(cur_means)])

            if args.dist_type == 'wdist':
                for idx1 in range(len(cur_means)):
                    for idx2 in range(idx1, len(cur_means)):
                        mean_dist = np.linalg.norm(cur_means[idx1] - cur_means[idx2]) ** 2
                        cov_dist = np.linalg.norm(cur_covs_sqrtm[idx1] - cur_covs_sqrtm[idx2], ord='fro') ** 2
                        dist = mean_dist + cov_dist
                        dist_mat[idx1][idx2] = dist_mat[idx2][idx1] = dist

            elif args.dist_type == 'mdled':
                for idx1 in range(len(cur_means)):
                    for idx2 in range(idx1, len(cur_means)):
                        merged_inv_cov = (cur_inv_covs[idx1] + cur_inv_covs[idx2])
                        mean_dist = (cur_means[idx1] - cur_means[idx2]).dot(merged_inv_cov).dot(
                            cur_means[idx1] - cur_means[idx2])
                        cov_dist = scipy.linalg.norm(cur_log_covs[idx1] - cur_log_covs[idx2], ord='fro') ** 2
                        dist = mean_dist + cov_dist
                        dist_mat[idx1][idx2] = dist_mat[idx2][idx1] = dist

            else:
                raise Exception('Invalid distance type')

            print('Covariance calculated')

            pickle.dump(dist_mat, open(dist_mat_file, 'wb'))

        else:
            wnid_order = protocol.wnid_order
            dist_mat = np.zeros([len(wnid_order), len(wnid_order)])
            if args.dataset == 'imagenet':
                if args.semantic_type == 'wordnet_wup':
                    for idx1 in tqdm(range(len(wnid_order)), desc='Calculating distances'):
                        for idx2 in range(idx1 + 1, len(wnid_order)):
                            synset1 = wn.synset_from_pos_and_offset(wn.NOUN, int(wnid_order[idx1][1:]))
                            synset2 = wn.synset_from_pos_and_offset(wn.NOUN, int(wnid_order[idx2][1:]))
                            wup = synset1.wup_similarity(synset2)
                            dist_mat[idx1][idx2] = dist_mat[idx2][idx1] = (1 - wup) * 2.0 / 3.0
            elif args.dataset == 'inat':
                if args.semantic_type == 'wordnet_wup':
                    for idx1 in tqdm(range(len(wnid_order)), desc='Calculating distances'):
                        for idx2 in range(idx1 + 1, len(wnid_order)):
                            set1 = set(wnid_order[idx1].split('_'))
                            set2 = set(wnid_order[idx2].split('_'))
                            dist_mat[idx1][idx2] = dist_mat[idx2][idx1] = 1 - len(set1.intersection(set2)) / 2.0
            else:
                raise Exception()

            pickle.dump(dist_mat, open(dist_mat_file, 'wb'))

    dist_mat = pickle.load(open(dist_mat_file, 'rb'))


    def calc_total_trans(dist_mat, src_indices, dst_indices):
        total_trans = np.sum(np.min(dist_mat[src_indices][:, dst_indices], axis=0))
        return total_trans


    first_iter = True

    old_candidates = []

    # convert dist_mat from 'random' to corresponding order
    random_order = protocol.wnid_order
    args.order_type = tmp_order_type
    args.order_idx = tmp_order_idx
    protocol = CILProtocol(args, dataset)
    current_order_file = protocol.wnid_order
    idx_order = []
    for num_classes in range(args.nb_cl, args.total_cl, args.nb_cl):
        idx_order.append([random_order.index(wnid) for wnid in current_order_file[:num_classes]])
    idx_order.append(list(range(args.total_cl)))

    trans = []
    for i in range(len(idx_order) - 1):
        trans.append(calc_total_trans(dist_mat, idx_order[i], idx_order[i + 1]))

    total_trans = np.sum(trans)
    print('Cumulative min dist: %f' % total_trans)

    with open(output_file, 'w') as fout:
        fout.write('%f' % total_trans + os.linesep)

    print('Run time: %.2f' % (time.time() - program_start_wall_time))
