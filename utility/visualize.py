import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from datasets.mvtec_dataset import MVTecDataset
import cv2
from config import Config


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


def draw_distance_graph(args, type_data, distance, types_test):
    plt.figure(figsize=(10, 8), dpi=100, facecolor='white')
    N_test = 0
    type_test = 'good'
    plt.subplot(2, 1, 1)
    plt.scatter((np.arange(len(distance[type_test])) + N_test), distance[type_test], alpha=0.5, label=type_test)
    plt.subplot(2, 1, 2)
    plt.hist(distance[type_test], alpha=0.5, label=type_test, bins=10)

    N_test += len(distance[type_test])
    for type_test in types_test[types_test != 'good']:
        plt.subplot(2, 1, 1)
        plt.scatter((np.arange(len(distance[type_test])) + N_test), distance[type_test], alpha=0.5, label=type_test)
        plt.subplot(2, 1, 2)
        plt.hist(distance[type_test], alpha=0.5, label=type_test, bins=10)
        N_test += len(distance[type_test])

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


def draw_heatmap_on_image(args, dataset, model, type_data):
    for type_test in dataset.types_test:
        for i, gt in tqdm(
                enumerate(MVTecDataset.gts_test[type_test]),
                desc=f'[verbose mode] visualize localization (case:{type_test})'):
            file = dataset.files_test[type_test][i]
            img = MVTecDataset.image_test[type_test][i]
            score_map = model.score_maps_test[type_test][i]

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
            plt.title('max score : %.2f' % model.score_max)
            plt.subplot2grid((3, 4), (1, 0), rowspan=2, colspan=2)
            plt.imshow(overlay_heatmap_on_image(img, (score_map / model.score_max)))
            plt.subplot2grid((3, 4), (1, 2), rowspan=2, colspan=2)
            plt.imshow((img.astype(np.float32) * (score_map / model.score_max)[..., None]).astype(np.uint8))
            plt.gcf().savefig(os.path.join(args.path_result, type_data,
                                           ('localization_k%02d_%s_%s_%s' %
                                            (args.k, type_data, type_test,
                                             os.path.basename(file)))))
            plt.clf()
            plt.close()


def draw_roc_curve(args, fpr_image, tpr_image, rocauc_image, fpr_pixel, tpr_pixel, rocauc_pixel):
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


