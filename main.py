import os
# import argparse
from argparse import ArgumentParser
import numpy as np
import cv2

import torch
import torch.nn.functional as F
import torchvision.models as models
from torchinfo import summary

import faiss
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
from scipy.ndimage import gaussian_filter

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from models.spade import Spade
from config import Config
from datasets.mvtec_dataset import MVTecDataset
from utility.visualize import *
from utility.time_measurement import *


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('-k', '--k', type=int, default=5, help='nearest neighbor\'s k')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch-size for feature extraction from ImageNet model')
    parser.add_argument('-pp', '--path_parent', type=str, default='./mvtec_anomaly_detection', help='parent path of data input path')
    parser.add_argument('-pr', '--path_result', type=str, default='./result', help='output path of figure image as the evaluation result')
    parser.add_argument('-c', '--cpu', action='store_true', help='use cpu')
    parser.add_argument('-v', '--verbose', action='store_true', help='save visualization of localization')

    parser.add_argument('--load_size', default=256, type=int, help='画像を読み込むサイズ')
    parser.add_argument('--input_size', default=224, type=int, help='画像をトリミングするサイズ')
    parser.add_argument('--num_cpu_max', default=4, type=int, help='')
    parser.add_argument('--num_workers', default=0, type=int, help='')

    args = parser.parse_args()
    return args


def exec_one_type_data(args, config, type_data):
    tic()
    dataset = MVTecDataset(args, type_data)
    # dataset = MVTecDataset(args, config, type_data)
    toc(f'----> MVTecDataset in {type_data}')

    tic()
    model = Spade(args, config, dataset)
    toc(f'----> Spade() in {type_data}')

    tic()
    model.create_normal_features()
    toc(f'----> model.create_normal_features() in {type_data}')

    tic()
    model.fit(type_data)
    toc(f'----> model.fit in {type_data}')

    toc(f'----> elapsed time for SPADE processing in {type_data}')

    draw_distance_graph(args, type_data, model.image_level_distance, dataset.types_test)
    if args.verbose:
        draw_heatmap_on_image(args, dataset, model, type_data)

    return model.fpr_image, model.tpr_image, model.rocauc_image, model.fpr_pixel, model.tpr_pixel, model.rocauc_pixel


def main(args):
    Config(args)

    os.makedirs(args.path_result, exist_ok=True)
    for type_data in Config.types_data:
        os.makedirs(os.path.join(args.path_result, type_data), exist_ok=True)

    fpr_image = {}
    tpr_image = {}
    rocauc_image = {}
    fpr_pixel = {}
    tpr_pixel = {}
    rocauc_pixel = {}

    # loop for types of data
    for type_data in Config.types_data:
        fpr_img, tpr_img, auc_img, fpr_pix, tpr_pix, auc_pix = exec_one_type_data(args, Config, type_data)

        fpr_image[type_data] = fpr_img
        tpr_image[type_data] = tpr_img
        rocauc_image[type_data] = auc_img

        fpr_pixel[type_data] = fpr_pix
        tpr_pixel[type_data] = tpr_pix
        rocauc_pixel[type_data] = auc_pix

    draw_roc_curve(args, fpr_image, tpr_image, rocauc_image, fpr_pixel, tpr_pixel, rocauc_pixel)

    rocauc_image_ = np.array([rocauc_image[type_data] for type_data in Config.types_data])
    rocauc_pixel_ = np.array([rocauc_pixel[type_data] for type_data in Config.types_data])

    print('np.mean(auc_image) = %.3f' % np.mean(rocauc_image_))
    print('np.mean(auc_pixel) = %.3f' % np.mean(rocauc_pixel_))


if __name__ == '__main__':
    args = arg_parser()
    main(args)
