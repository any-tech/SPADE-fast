import os
import numpy as np
from argparse import ArgumentParser

from utils.config import ConfigData, ConfigFeat, ConfigSpade, ConfigDraw
from utils.tictoc import tic, toc
from utils.metrics import calc_imagewise_metrics, calc_pixelwise_metrics
from utils.visualize import draw_roc_curve, draw_distance_graph, draw_heatmap_on_image
from datasets.mvtec_dataset import MVTecDataset
from models.feat_extract import FeatExtract
from models.spade import Spade


def arg_parser():
    parser = ArgumentParser()
    # environment related
    parser.add_argument('-n', '--num_cpu_max', default=4, type=int,
                        help='number of CPUs for parallel reading input images')
    parser.add_argument('-c', '--cpu', action='store_true', help='use cpu')
    # I/O and visualization related
    parser.add_argument('-pp', '--path_parent', type=str, default='./mvtec_anomaly_detection',
                        help='parent path of data input path')
    parser.add_argument('-pr', '--path_result', type=str, default='./result',
                        help='output path of figure image as the evaluation result')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='save visualization of localization')
    # data loader related
    parser.add_argument('-bs', '--batch_size', type=int, default=16,
                        help='batch-size for feature extraction by ImageNet model')
    parser.add_argument('-sr', '--size_resize', type=int, default=256,
                        help='size of resizing input image')
    parser.add_argument('-sc', '--size_crop', type=int, default=224,
                        help='size of cropping after resize')
    # feature extraction related
    parser.add_argument('-b', '--backbone', type=str,
                        default='torchvision.models.wide_resnet50_2',
                        help='specify torchvision model with the full path')
    parser.add_argument('-w', '--weight', type=str,
                        default='torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1',
                        help='specify the trained weights of torchvision model with the full path.')
    parser.add_argument('-lm', '--layer_map', nargs='+', type=str,
                        default=['layer1[-1]', 'layer2[-1]', 'layer3[-1]'],
                        help='specify layers to extract feature map')
    parser.add_argument('-lv', '--layer_vec', type=str, default='avgpool',
                        help='specify layers to extract feature vector')
    # Nearest-Neighbor related
    parser.add_argument('-k', '--k', type=int, default=5,
                        help='nearest neighbor\'s k for coreset searching')
    # post precessing related
    parser.add_argument('-pod', '--pixel_outer_decay', type=int, default=0,
                        help='number of outer pixels to decay anomaly score')

    args = parser.parse_args()
    return args


def apply_spade(type_data, feat_ext, spade, cfg_draw):
    print('\n----> SPADE processing in %s start' % type_data)
    tic()

    # read images
    MVTecDataset(type_data)

    # extract features
    feat_map_train, feat_vec_train = feat_ext.extract(MVTecDataset.imgs_train, is_train=True)
    feat_map_test, feat_vec_test = feat_ext.extract(MVTecDataset.imgs_test, is_train=False)

    # Deep Nearest Neighbor Anomaly Detection
    D_img, I_nn, y_img = spade.search_nearest_neighbor(feat_vec_train, feat_vec_test)

    # Sub-Image Anomaly Detection with Deep Pyramid Correspondences
    D_pix, D_pix_max = spade.localization(feat_map_train, feat_map_test, I_nn)

    # measure per image
    fpr_img, tpr_img, rocauc_img = calc_imagewise_metrics(D_img, y_img)
    print('%s imagewise ROCAUC: %.3f' % (type_data, rocauc_img))
    fpr_pix, tpr_pix, rocauc_pix = calc_pixelwise_metrics(D_pix, MVTecDataset.gts_test)
    print('%s pixelwise ROCAUC: %.3f' % (type_data, rocauc_pix))

    toc(tag=('----> SPADE processing in %s end, elapsed time' % type_data))

    draw_distance_graph(type_data, cfg_draw, D_img)
    if args.verbose:
        draw_heatmap_on_image(type_data, cfg_draw, D_pix, MVTecDataset.gts_test, D_pix_max,
                              MVTecDataset.imgs_test, MVTecDataset.files_test,
                              I_nn, MVTecDataset.imgs_train)

    return [fpr_img, tpr_img, rocauc_img, fpr_pix, tpr_pix, rocauc_pix]


def main(args):
    ConfigData(args)  # static define to speed-up
    cfg_feat = ConfigFeat(args)
    cfg_spade = ConfigSpade(args)
    cfg_draw = ConfigDraw(args)

    feat_ext = FeatExtract(cfg_feat)
    spade = Spade(cfg_spade, feat_ext.survey_depth())

    os.makedirs(args.path_result, exist_ok=True)
    for type_data in ConfigData.types_data:
        os.makedirs(os.path.join(args.path_result, type_data), exist_ok=True)

    fpr_img = {}
    tpr_img = {}
    rocauc_img = {}
    fpr_pix = {}
    tpr_pix = {}
    rocauc_pix = {}

    # loop for types of data
    for type_data in ConfigData.types_data:
        result = apply_spade(type_data, feat_ext, spade, cfg_draw)

        fpr_img[type_data] = result[0]
        tpr_img[type_data] = result[1]
        rocauc_img[type_data] = result[2]

        fpr_pix[type_data] = result[3]
        tpr_pix[type_data] = result[4]
        rocauc_pix[type_data] = result[5]

    draw_roc_curve(cfg_draw, fpr_img, tpr_img, rocauc_img, fpr_pix, tpr_pix, rocauc_pix)

    rocauc_img_ = np.array([rocauc_img[type_data] for type_data in ConfigData.types_data])
    rocauc_pix_ = np.array([rocauc_pix[type_data] for type_data in ConfigData.types_data])
    for type_data in ConfigData.types_data:
        print('rocauc_img[%s] = %.3f' % (type_data, rocauc_img[type_data]))
    print('np.mean(rocauc_img_) = %.3f' % np.mean(rocauc_img_))
    for type_data in ConfigData.types_data:
        print('rocauc_pix[%s] = %.3f' % (type_data, rocauc_pix[type_data]))
    print('np.mean(rocauc_pix_) = %.3f' % np.mean(rocauc_pix_))


if __name__ == '__main__':
    args = arg_parser()
    main(args)
