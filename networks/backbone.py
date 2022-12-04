# -*- coding:utf-8 -*-  

""" 
@time: 11/30/20 10:41 PM 
@author: Chen He 
@site:  
@file: backbone.py
@description:  
"""

from networks.backbones.imagenet64x64_resnet import resnet18 as resnet18_64x64


def get_backbone(args):
    if args.resolution == '64x64':
        if args.network == 'resnet18':
            backbone = resnet18_64x64(weight_decay=args.weight_decay, final_relu=not args.no_final_relu)
        else:
            raise Exception('Invalid network')
    else:
        raise Exception('Invalid resolution')

    args.feat_layer_idx = len(backbone.layers) - 1

    return backbone
