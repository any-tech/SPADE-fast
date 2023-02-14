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
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='batch-size for feature extraction from ImageNet model')
    parser.add_argument('-pp', '--path_parent', type=str, default='./mvtec_anomaly_detection', help='parent path of data input path')
    parser.add_argument('-pr', '--path_result', type=str, default='./result', help='output path of figure image as the evaluation result')
    parser.add_argument('-c', '--cpu', action='store_true', help='use cpu')
    parser.add_argument('-v', '--verbose', action='store_true', help='save visualization of localization')

    parser.add_argument('--load_size', default=256, type=int, help='画像を読み込むサイズ')
    parser.add_argument('--input_size', default=224, type=int, help='画像をトリミングするサイズ')
    parser.add_argument('--num_cpu_max', default=4, type=int, help='')
    parser.add_argument('--num_workers', default=4, type=int, help='')

    args = parser.parse_args()
    return args


# args = arg_parser()
# config = Config(args)


def read_and_resize_for_train(file):
    img = cv2.imread(file)[..., ::-1]  # BGR2RGB

    # from the paper
    img = cv2.resize(img, (config.SHAPE_MIDDLE[1], config.SHAPE_MIDDLE[0]), interpolation=cv2.INTER_AREA)
    img = img[
          config.pixel_crop[0]:(config.SHAPE_INPUT[0] + config.pixel_crop[0]),
          config.pixel_crop[1]:(config.SHAPE_INPUT[1] + config.pixel_crop[1])
    ]

    # imgs_train[np.where(files_train == file)[0]] = img

    return img


def exec_one_type_data(args, config, type_data, model):
    tic()

    train_dataset = MVTecDataset(args, config, type_data, is_train=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # feature extract for train
    f1_train = None
    f2_train = None
    f3_train = None
    fl_train = None
    for i, img in tqdm(enumerate(train_dataloader), desc='feature extract for train'):
        with torch.no_grad():
            features = model(img.to(config.device))
            f1_train = np.concatenate([f1_train, features[0]], axis=0) if f1_train is not None else features[0]
            f2_train = np.concatenate([f2_train, features[1]], axis=0) if f2_train is not None else features[1]
            f3_train = np.concatenate([f3_train, features[2]], axis=0) if f3_train is not None else features[2]
            fl_train = np.concatenate([fl_train, features[3]], axis=0) if fl_train is not None else features[3]

    fl_train = fl_train.squeeze(-1).squeeze(-1)

    f1_train = np.vstack(outputs[0::4])
    f2_train = np.vstack(outputs[1::4])
    f3_train = np.vstack(outputs[2::4])
    fl_train = np.vstack(outputs[3::4]).squeeze(-1).squeeze(-1)

    # feature extract for test
    f1_test = {}
    f2_test = {}
    f3_test = {}
    fl_test = {}
    for type_test in types_test:

        x_batch = []
        outputs = []
        for i, img in tqdm(enumerate(imgs_test[type_test]), desc='feature extract for test (case:%s)' % type_test):

            x = torch.from_numpy(img.astype(np.float32)).to(device)
            x = x / 255
            x = x - config.MEAN
            x = x / config.STD
            x = x.unsqueeze(0).permute(0, 3, 1, 2)

            x_batch.append(x)

            if (len(x_batch) == args.batch_size) | (i == (len(imgs_test[type_test]) - 1)):
                with torch.no_grad():
                    _ = model(torch.vstack(x_batch))
                x_batch = []

        f1_test[type_test] = np.vstack(outputs[0::4])
        f2_test[type_test] = np.vstack(outputs[1::4])
        f3_test[type_test] = np.vstack(outputs[2::4])
        fl_test[type_test] = np.vstack(outputs[3::4]).squeeze(-1).squeeze(-1)

    # exec knn by final layer feature vector
    d = fl_train.shape[1]
    index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),
                                 d,
                                 faiss.GpuIndexFlatConfig())
    index.add(fl_train)

    D_test = {}
    y_test = {}
    I_test = {}

    type_test = 'good'
    D, I = index.search(fl_test[type_test], args.k)
    D_test[type_test] = np.mean(D, axis=1)
    y_test[type_test] = np.zeros([len(D)], dtype=np.int16)
    I_test[type_test] = I
    for type_test in types_test[types_test != 'good']:
        D, I = index.search(fl_test[type_test], args.k)
        D_test[type_test] = np.mean(D, axis=1)
        y_test[type_test] = np.ones([len(D)], dtype=np.int16)
        I_test[type_test] = I

    D_list = np.concatenate([D_test['good'],
                             np.hstack([D_test[type_test] for type_test in types_test[types_test != 'good']])])
    y_list = np.concatenate([y_test['good'],
                             np.hstack([y_test[type_test] for type_test in types_test[types_test != 'good']])])

    # calculate per-image level ROCAUC
    fpr, tpr, _ = roc_curve(y_list, D_list)
    rocauc = roc_auc_score(y_list, D_list)
    print('%s per-image level ROCAUC: %.3f' % (type_data, rocauc))

    # stock for output result
    fpr_image[type_data] = fpr
    tpr_image[type_data] = tpr
    rocauc_image[type_data] = rocauc

    # prep work variable for measure
    flatten_gt_list = []
    flatten_score_map_list = []
    score_maps_test = {}
    score_max = -9999

    # k nearest features from the gallery (k=1)
    index1 = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),
                                  f1_train.shape[1],
                                  faiss.GpuIndexFlatConfig())
    index2 = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),
                                  f2_train.shape[1],
                                  faiss.GpuIndexFlatConfig())
    index3 = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),
                                  f3_train.shape[1],
                                  faiss.GpuIndexFlatConfig())

    for type_test in types_test:
        score_maps_test[type_test] = []

        for i, gt in tqdm(enumerate(gts_test[type_test]), desc='localization (case:%s)' % type_test):
            score_map_mean = []

            for i_nn in range(3):
                # construct a gallery of features at all pixel locations of the K nearest neighbors
                if i_nn == 0:
                    f_neighbor = f1_train[I_test[type_test][i]]
                    f_query = f1_test[type_test][[i]]
                    index_ = index1
                elif i_nn == 1:
                    f_neighbor = f2_train[I_test[type_test][i]]
                    f_query = f2_test[type_test][[i]]
                    index_ = index2
                elif i_nn == 2:
                    f_neighbor = f3_train[I_test[type_test][i]]
                    f_query = f3_test[type_test][[i]]
                    index_ = index3

                # get shape
                _, C, H, W = f_neighbor.shape

                # adjust dimensions to measure distance in the channel dimension for all combinations
                f_neighbor = f_neighbor.transpose(0, 2, 3, 1)  # (K, C, H, W) -> (K, H, W, C)
                f_neighbor = f_neighbor.reshape(-1, C)         # (K, H, W, C) -> (KHW, C)
                f_query = f_query.transpose(0, 2, 3, 1)  # (K, C, H, W) -> (K, H, W, C)
                f_query = f_query.reshape(-1, C)         # (K, H, W, C) -> (KHW, C)

                # k nearest features from the gallery (k=1)
                index_.reset()
                index_.add(f_neighbor)
                D, _ = index_.search(f_query, 1)

                # transform to scoremap
                score_map = D.reshape(H, W)
                score_map = cv2.resize(score_map, (config.SHAPE_INPUT[0], config.SHAPE_INPUT[1]))
                score_map_mean.append(score_map)

            # average distance between the features
            score_map_mean = np.mean(np.array(score_map_mean), axis=0)
            # apply gaussian smoothing on the score map
            score_map_smooth = gaussian_filter(score_map_mean, sigma=4)

            score_maps_test[type_test].append(score_map_smooth)
            score_max = max(score_max, np.max(score_map_smooth))

            flatten_gt_mask = np.concatenate(gt).ravel()
            flatten_score_map = np.concatenate(score_map_smooth).ravel()
            flatten_gt_list.append(flatten_gt_mask)
            flatten_score_map_list.append(flatten_score_map)

    flatten_gt_list = np.array(flatten_gt_list).reshape(-1)
    flatten_score_map_list = np.array(flatten_score_map_list).reshape(-1)

    # calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(flatten_gt_list, flatten_score_map_list)
    rocauc = roc_auc_score(flatten_gt_list, flatten_score_map_list)
    print('%s per-pixel level ROCAUC: %.3f' % (type_data, rocauc))

    # stock for output result
    fpr_pixel[type_data] = fpr
    tpr_pixel[type_data] = tpr
    rocauc_pixel[type_data] = rocauc

    toc('elapsed time for SPADE processing in %s' % type_data)

    plt.figure(figsize=(10, 8), dpi=100, facecolor='white')
    N_test = 0
    type_test = 'good'
    plt.subplot(2, 1, 1)
    plt.scatter((np.arange(len(D_test[type_test])) + N_test),
                D_test[type_test], alpha=0.5, label=type_test)
    plt.subplot(2, 1, 2)
    plt.hist(D_test[type_test], alpha=0.5, label=type_test, bins=10)
    N_test += len(D_test[type_test])
    for type_test in types_test[types_test != 'good']:
        plt.subplot(2, 1, 1)
        plt.scatter((np.arange(len(D_test[type_test])) + N_test),
                    D_test[type_test], alpha=0.5, label=type_test)
        plt.subplot(2, 1, 2)
        plt.hist(D_test[type_test], alpha=0.5, label=type_test, bins=10)
        N_test += len(D_test[type_test])

    plt.subplot(2, 1, 1)
    plt.grid()
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.grid()
    plt.legend()
    plt.gcf().tight_layout()
    plt.gcf().savefig(os.path.join(args.path_result, type_data, ('pred-dist_k%02d_%s.png' % (args.k, type_data))))
    plt.clf()
    plt.close()

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

    model = Spade(args)

    # loop for types of data
    for type_data in Config.types_data:
        exec_one_type_data(args, Config, type_data, model)

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
