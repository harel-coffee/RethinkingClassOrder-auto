# -*- coding:utf-8 -*-

"""
@time: 1/18/22 4:59 PM
@author: Chen He
@site:
@file: print_transferability.py
@description:
"""
import os.path

import numpy as np

PARENT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'calculated_transferability')

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='imagenet64x64')
args = parser.parse_args()

if args.dataset == 'imagenet64x64':
    nb_cl = 10
elif args.dataset == 'inat64x64':
    nb_cl = 9
else:
    raise Exception()


def read_results(filename_template):
    measures = []
    for i in range(5):
        with open(os.path.join(PARENT_DIR, filename_template % (i + 1)), 'r') as fin:
            measures.append(float(fin.readline().strip()))

    return np.mean(measures)


dist_mapping = {'wdist': 'WD', 'mdled': 'MD-LED'}
feat_type_mapping = {'supervised': 'Sup.', 'ssl': 'SSL'}

print('\teven\tgroup')
for dist in ['wdist', 'mdled']:
    for feat_type in ['supervised', 'ssl']:
        result_line = []
        for order_type in ['even', 'group']:
            result_line.append(read_results(
                '%s_%s_' % (args.dataset, order_type) + '%d' + '_nb_cl_%d_%s_%s.txt' % (nb_cl, feat_type, dist)))
        print('[%s (%s)]' % (dist_mapping[dist], feat_type_mapping[feat_type]) + '\t' + '\t'.join(
            [str(elem) for elem in result_line]))

result_line = []
for order_type in ['even', 'group']:
    result_line.append(read_results(
        '%s_%s_' % (args.dataset, order_type) + '%d' + '_nb_cl_%d_semantic_wordnet_wup.txt' % nb_cl))
print('[Wu-Palmer]\t' + '\t'.join([str(elem) for elem in result_line]))
