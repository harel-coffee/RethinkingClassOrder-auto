# -*- coding:utf-8 -*-  

""" 
@time: 1/28/22 12:50 AM 
@author: Chen He 
@site:  
@file: draw_class_order.py.py
@description:  
"""

import os
from argparse import ArgumentParser

from PIL import Image
from nltk.corpus import wordnet as wn

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='inat')
parser.add_argument('--order_file', type=str,
                    default='order_9x9_optimal_semantic_wordnet_wup_nb_cl_9_beam_0_seed_1993_1.txt')
parser.add_argument('--no_ordered', action='store_true')

args = parser.parse_args()

if args.dataset == 'imagenet':
    group_file = 'imagenet/order_10x10_group_1_wnid.txt'
    nb_cl = 10
elif args.dataset == 'inat':
    group_file = 'inat/order_9x9_group_1.txt'
    nb_cl = 9
else:
    raise Exception()


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

color_mapping = dict()
with open(group_file, 'r') as fin:
    for line_i, line in enumerate(fin.readlines()):
        color_mapping[line.strip()] = colors[line_i // nb_cl]

order_file = os.path.join(args.dataset, args.order_file)
dst_dir = (os.path.splitext(order_file)[0] + '_ordered') if not args.no_ordered else os.path.splitext(order_file)[0]

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

with open(order_file, 'r') as fin:
    if args.no_ordered:
        for line_i, line in enumerate(fin.readlines()):
            wnid = line.strip()
            if args.dataset == 'imagenet':
                ss = wn.synset_from_pos_and_offset(wn.NOUN, int(wnid[1:]))
                name = ss.lemmas()[0].name()
            elif args.dataset == 'inat':
                name = wnid
            img = Image.new('RGB', size=(32, 32), color=hex_to_rgb(color_mapping[wnid]))
            img.save(os.path.join(dst_dir, '%03d_%s.jpg' % (line_i + 1, name)))
    else:
        objs = []
        wnids = []
        i = 0
        for line_i, line in enumerate(fin.readlines()):

            wnid = line.strip()
            if args.dataset == 'imagenet':
                ss = wn.synset_from_pos_and_offset(wn.NOUN, int(wnid[1:]))
                name = ss.lemmas()[0].name()
            elif args.dataset == 'inat':
                name = wnid

            objs.append({'name': name, 'color': color_mapping[wnid], 'wnid': wnid})
            objs = sorted(objs, key=lambda obj: obj['color'])

            if len(objs) % nb_cl == 0:
                for obj in objs:
                    img = Image.new('RGB', size=(32, 32), color=hex_to_rgb(obj['color']))
                    img.save(os.path.join(dst_dir, '%03d_%s.jpg' % (i + 1, obj['name'])))
                    wnids.append(obj['wnid'])
                    i += 1
                objs = []

        ref_colors = sorted(colors)
        for i in range(len(ref_colors)):
            img = Image.new('RGB', size=(32, 32), color=hex_to_rgb(ref_colors[i]))
            img.save(os.path.join(dst_dir, 'ref_%03d.jpg' % (i + 1)))
            i += 1

        wnids_reordered = sorted(wnids)
        with open(os.path.join(dst_dir, 'wnids.txt'), 'w') as fout:
            for wnid_i, wnid in enumerate(wnids):
                if wnid_i > 0 and wnid_i % nb_cl == 0:
                    fout.write(os.linesep)
                fout.write(str(wnids_reordered.index(wnid) + 1) + '\t')
            # fout.writelines(os.linesep.join(wnids))
