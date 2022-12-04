# -*- coding:utf-8 -*-  

""" 
@time: 12/5/19 10:10 PM 
@author: Chen He 
@site:  
@file: utils_func.py
@description:  
"""

import os
import pickle

import numpy as np
import tensorflow as tf
import wandb
from imageio import imsave

from utils.vis import vis_loss


class MySummary:
    def __init__(self):
        self.log = dict()

    def add(self, name, step, value):
        if name not in self.log:
            self.log[name] = dict()

        self.log[name][step] = value

    def dump(self, filename):
        pickle.dump(self.log, open(filename, 'wb'))

    def vis(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        for key in self.log:
            vis_loss(self.log[key], folder, key)

    def reset(self):
        self.log.clear()


def wandb_log(args, stats):
    if args.wandb_flag:
        wandb.log(stats)


def get_top5_acc(logits, labels):
    top5_hit = 0
    for img_idx, single_logits in enumerate(logits):
        top5_hit += 1 if labels[img_idx] in np.argsort(-single_logits)[:5] else 0
    top5_acc = top5_hit * 100. / len(labels)
    return top5_acc


def get_harmonic_mean(per_class_accs, num_old_classes):
    new_acc = np.mean(per_class_accs[num_old_classes:])
    if num_old_classes == 0:
        hm = new_acc
    else:
        old_acc = np.mean(per_class_accs[:num_old_classes])
        hm = (2 * new_acc * old_acc / (new_acc + old_acc))
    return hm


def post_scaling_il2m(logits, group_idx, nb_cl, old_num_classes, init_classes_means, current_classes_means,
                      models_confidence):
    pred_to_new_class_indices = np.where(np.argmax(logits, axis=1) >= old_num_classes)[0]
    for old_class_idx in range(old_num_classes):
        logits[pred_to_new_class_indices, old_class_idx] *= \
            init_classes_means[old_class_idx] / current_classes_means[old_class_idx] * \
            models_confidence[group_idx] / models_confidence[old_class_idx // nb_cl]
    return logits


def get_num_exemplars(num_total, classes, drop_remainder=False):
    class_ratios = [1. / len(classes)] * len(classes)
    num_exemplars = np.floor(num_total * np.array(class_ratios)).astype(np.int)
    if not drop_remainder:
        remainder = num_total - sum(num_exemplars)
        for i in range(remainder):
            num_exemplars[i] += 1
    return num_exemplars


def save_images(X, save_path):
    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples // rows

    assert X.ndim == 4

    # BCHW -> BHWC
    X = tf.transpose(X, (0, 2, 3, 1))
    h, w = X[0].shape[:2]
    img = np.zeros((h * nh, w * nw, 3), np.uint8)

    for n, x in enumerate(X):
        j = n // nw
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w] = x

    imsave(save_path, img)


def get_folder_size(path):
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if
               os.path.isfile(os.path.join(path, f)))


def lr_scheduler(args, class_inc):
    if args.epochs > 100:
        lr_desc_epochs = [int(0.6 * args.epochs), int(0.8 * args.epochs), int(0.9 * args.epochs)]
    else:
        lr_desc_epochs = [int(0.7 * args.epochs), int(0.9 * args.epochs)]

    # if args.epochs >= 120:
    #     lr_desc_epochs = [70, 100]
    # elif 100 <= args.epochs < 120:
    #     lr_desc_epochs = [70, 90]
    # else:
    #     lr_desc_epochs = [49, 63]

    lrs = []
    base_lr = args.base_lr
    lr = base_lr

    for epoch in range(args.epochs):
        if epoch in lr_desc_epochs:
            lr *= 0.1
        lrs.append(lr)

    return lrs
