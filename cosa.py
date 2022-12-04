# -*- coding:utf-8 -*-  

""" 
@time: 11/30/20 10:09 PM 
@author: Chen He 
@site:  
@file: cosa.py
@description:  
"""

import argparse
import os
import pickle
import random
import time
from pprint import pprint

import numpy as np
import scipy
import tensorflow as tf
from nltk.corpus import wordnet as wn
from pyclustering.cluster.kmedoids import kmedoids
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
    parser.add_argument('--feat_type', type=str, default='ssl',
                        choices=['ssl', 'supervised'])

    # debug switch
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test_final_only', action='store_true')

    # search settings
    parser.add_argument('--beam_size', type=int, default=0)
    parser.add_argument('--non_ordered', action='store_true')

    args = parser.parse_args()

    '''
    Check arguments
    '''
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

    # init protocol
    protocol = CILProtocol(args, dataset)

    dist_folder = 'dists'
    if not os.path.exists(dist_folder):
        os.makedirs(dist_folder)

    setting_str = 'semantic_%s' % args.semantic_type if args.semantic else '%s_%s' % (args.feat_type, args.dist_type)

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
                        'result/inat_9x9_random_1/base_81/seed_1993/ssl_resnet18_fc_70_adam_0.005_aug_wd_0.0001/group_1/checkpoints/1st_stage/final')
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

                if args.dist_type == 'bdist':
                    sign, logdet_cov = tf.linalg.slogdet(cov)
                    cur_logdet_covs.append(logdet_cov)
                    assert sign == 1.0

            dist_mat = np.zeros([len(cur_means), len(cur_means)])

            if args.dist_type == 'bdist':
                for idx1 in tqdm(range(len(cur_means))):
                    for idx2 in range(idx1, len(cur_means)):
                        merged_cov = (cur_covs[idx1] + cur_covs[idx2]) / 2
                        mean_dist = (cur_means[idx1] - cur_means[idx2]).dot(tf.linalg.inv(merged_cov)).dot(
                            cur_means[idx1] - cur_means[idx2]) / 8

                        sign, logdet_merged_cov = tf.linalg.slogdet(merged_cov)
                        assert sign == 1.0
                        cov_dist = (logdet_merged_cov - cur_logdet_covs[idx1] / 2 - cur_logdet_covs[idx2] / 2) / 2
                        dist = mean_dist + cov_dist
                        dist_mat[idx1][idx2] = dist_mat[idx2][idx1] = dist

            elif args.dist_type == 'wdist':
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
                            dist_mat[idx1][idx2] = dist_mat[idx2][idx1] = (2 - len(set1.intersection(set2))) / 3.0
            else:
                raise Exception()

            pickle.dump(dist_mat, open(dist_mat_file, 'wb'))

    dist_mat = pickle.load(open(dist_mat_file, 'rb'))


    def calc_total_trans(dist_mat, src_indices, dst_indices):
        total_trans = np.sum(np.min(dist_mat[src_indices][:, dst_indices], axis=0))
        return total_trans


    first_iter = True
    old_candidates = []

    for num_src_classes in range(args.total_cl - args.nb_cl, 0, -args.nb_cl):

        if first_iter:
            dst_indices = list(range(args.total_cl))

            # local minimum
            kmedoids_instance = kmedoids(dist_mat, np.random.choice(len(dist_mat), num_src_classes, replace=False),
                                         data_type='distance_matrix')
            kmedoids_instance.process()
            centers = kmedoids_instance.get_medoids()
            optimal_trans = calc_total_trans(dist_mat, centers, dst_indices)
            old_candidates.append({'idx_order': [centers], 'optimal_trans': optimal_trans})

            # random K (K is beam size)
            for i in range(args.beam_size):
                selected_indices = list(np.random.choice(dst_indices, num_src_classes, replace=False))
                optimal_trans = calc_total_trans(dist_mat, selected_indices, dst_indices)
                old_candidates.append({'idx_order': [selected_indices], 'optimal_trans': optimal_trans})

            first_iter = False
        else:
            candidates = []
            assert old_candidates
            for old_cand in old_candidates:
                dst_indices = old_cand['idx_order'][0]

                # local minimum
                kmedoids_instance = kmedoids(dist_mat[dst_indices][:, dst_indices],
                                             np.random.choice(len(dst_indices), num_src_classes, replace=False),
                                             data_type='distance_matrix')
                kmedoids_instance.process()
                centers = kmedoids_instance.get_medoids()
                real_indices = [dst_indices[center] for center in centers]
                optimal_trans = calc_total_trans(dist_mat, real_indices, dst_indices)
                candidates.append(
                    {'idx_order': [real_indices] + old_cand['idx_order'],
                     'optimal_trans': old_cand['optimal_trans'] + optimal_trans})

                # random K (K is beam size)
                for i in range(args.beam_size):
                    selected_indices = list(np.random.choice(dst_indices, num_src_classes, replace=False))
                    optimal_trans = calc_total_trans(dist_mat, selected_indices, dst_indices)
                    candidates.append(
                        {'idx_order': [selected_indices] + old_cand['idx_order'],
                         'optimal_trans': old_cand['optimal_trans'] + optimal_trans})

            if args.non_ordered:
                selected_indices = np.random.choice(range(len(candidates)), args.beam_size + 1, replace=False)
                old_candidates = [candidates[i] for i in selected_indices]
            else:
                sorted(candidates, key=lambda a: a['optimal_trans'])
                old_candidates = candidates[:(args.beam_size + 1)]

    final_order = old_candidates[0]['idx_order']
    final_optimal_trans = old_candidates[0]['optimal_trans']
    final_order.append(list(range(len(dist_mat))))
    for i in range(len(final_order) - 1, 0, -1):
        final_order[i] = list(set(final_order[i]) - set(final_order[i - 1]))
    final_order = sum(final_order, [])

    dist_wnid_order = protocol.wnid_order

    optimal_wnid_order = [dist_wnid_order[i] for i in final_order]

    beam_search_suffix = '_beam_%d_seed_%d_' % (args.beam_size, args.random_seed)
    beam_search_suffix += '_non_ordered' if args.non_ordered else ''
    new_order_file = protocol.order_file.replace('_%s_' % args.order_type,
                                                 ('_optimal_%s_' % setting_str) + extra_suffix + beam_search_suffix)
    # sorted(candidates, lambda a, b: a['optimal_trans'] < b['optimal_trans'])
    # optimal_idx_order = candidates[0]['']
    with open(new_order_file, 'w') as fout:
        for wnid in optimal_wnid_order:
            fout.write(wnid + os.linesep)

    print('Total trans: %f' % final_optimal_trans)
    print('Run time: %.2f' % (time.time() - program_start_wall_time))
