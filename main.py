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


def tic():
    #require to import time
    global start_time_tictoc
    start_time_tictoc = time.time()


def toc(tag="elapsed time"):
    if "start_time_tictoc" in globals():
        print("{}: {:.1f} [sec]".format(tag, time.time() - start_time_tictoc))
    else:
        print("tic has not been called")


# https://github.com/gsurma/cnn_explainer/blob/main/utils.py
def overlay_heatmap_on_image(img, heatmap, ratio_img=0.5):
    img = img.astype(np.float32)

    heatmap = 1 - np.clip(heatmap, 0, 1)
    heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32)

    overlay = (img * ratio_img) + (heatmap * (1 - ratio_img))
    overlay = np.clip(overlay, 0, 255)
    overlay = overlay.astype(np.uint8)
    return overlay


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

    dataset = MVTecDataset(args, config, type_data)
    model = Spade(args, config, dataset)
    model.create_normal_features()
    model.fit(type_data)

    toc('elapsed time for SPADE processing in %s' % type_data)

    draw_distance_graph(args, type_data, model.image_level_distance, dataset.types_test)

    if args.verbose:
        for type_test in types_test:
            for i, gt in tqdm(enumerate(gts_test[type_test]),
                              desc=('[verbose mode] visualize localization (case:%s)' %
                                    type_test)):
                file = files_test[type_test][i]
                img = imgs_test[type_test][i]
                score_map = score_maps_test[type_test][i]

                plt.figure(figsize=(9, 6), dpi=100, facecolor='white')
                plt.rcParams['font.size'] = 8
                plt.subplot2grid((3, 3), (0, 0), rowspan=1, colspan=1)
                plt.imshow(img)
                plt.title('%s : %s' % (file.split('/')[-2], file.split('/')[-1]))
                plt.subplot2grid((3, 3), (0, 1), rowspan=1, colspan=1)
                plt.imshow(gt)
                plt.subplot2grid((3, 3), (0, 2), rowspan=1, colspan=1)
                plt.imshow(score_map)
                plt.colorbar()
                plt.title('max score : %.2f' % score_max)
                plt.subplot2grid((3, 4), (1, 0), rowspan=2, colspan=2)
                plt.imshow(overlay_heatmap_on_image(img, (score_map / score_max)))
                plt.subplot2grid((3, 4), (1, 2), rowspan=2, colspan=2)
                plt.imshow((img.astype(np.float32) *
                            (score_map / score_max)[..., None]).astype(np.uint8))
                plt.gcf().savefig(os.path.join(args.path_result, type_data,
                                               ('localization_k%02d_%s_%s_%s' %
                                                (args.k, type_data, type_test,
                                                 os.path.basename(file)))))
                plt.clf()
                plt.close()
    print('--------------------------------')


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
        exec_one_type_data(args, Config, type_data)

    plt.figure(figsize=(12, 6), dpi=100, facecolor='white')
    for type_data in Config.types_data:
        plt.subplot(1, 2, 1)
        plt.plot(fpr_image[type_data], tpr_image[type_data], label='%s ROCAUC: %.3f' % (type_data, rocauc_image[type_data]))
        plt.subplot(1, 2, 2)
        plt.plot(fpr_pixel[type_data], tpr_pixel[type_data], label='%s ROCAUC: %.3f' % (type_data, rocauc_pixel[type_data]))

    plt.subplot(1, 2, 1)
    plt.title('Image-level anomaly detection accuracy (ROCAUC %)')
    plt.grid()
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('Pixel-level anomaly detection accuracy (ROCAUC %)')
    plt.grid()
    plt.legend()
    plt.gcf().tight_layout()
    plt.gcf().savefig(os.path.join(args.path_result, ('roc-curve_k%02d.png' % args.k)))
    plt.clf()
    plt.close()

    rocauc_image_ = np.array([rocauc_image[type_data] for type_data in Config.types_data])
    rocauc_pixel_ = np.array([rocauc_pixel[type_data] for type_data in Config.types_data])

    print('np.mean(auc_image) = %.3f' % np.mean(rocauc_image_))
    print('np.mean(auc_pixel) = %.3f' % np.mean(rocauc_pixel_))


if __name__ == '__main__':
    args = arg_parser()
    main(args)
