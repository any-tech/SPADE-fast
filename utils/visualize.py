import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.config import ConfigDraw


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


def draw_distance_graph(type_data, D):
    plt.figure(figsize=(10, 8), dpi=100, facecolor='white')

    # 'good' 1st
    N_test = 0
    type_test = 'good'
    plt.subplot(2, 1, 1)
    plt.scatter((np.arange(len(D[type_test])) + N_test), D[type_test],
                alpha=0.5, label=type_test)
    plt.subplot(2, 1, 2)
    plt.hist(D[type_test], alpha=0.5, label=type_test, bins=10)

    # other than 'good'
    N_test += len(D[type_test])
    types_test = np.array([k for k in D.keys() if k != 'good'])
    for type_test in types_test:
        plt.subplot(2, 1, 1)
        plt.scatter((np.arange(len(D[type_test])) + N_test), D[type_test],
                    alpha=0.5, label=type_test)
        plt.subplot(2, 1, 2)
        plt.hist(D[type_test], alpha=0.5, label=type_test, bins=10)
        N_test += len(D[type_test])

    plt.subplot(2, 1, 1)
    plt.grid()
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.grid()
    plt.legend()
    plt.gcf().tight_layout()
    plt.gcf().savefig(os.path.join(ConfigDraw.path_result, type_data,
                                   ('pred-dist_k%02d_%s.png' % (ConfigDraw.k, type_data))))
    plt.clf()
    plt.close()


def draw_heatmap_on_image(type_data, D, y, D_max, imgs, files, I_nn, imgs_nn):
    for type_test in D.keys():
        for i in tqdm(range(len(D[type_test])),
                      desc='[verbose mode] visualize localization (case:%s)' % type_test):
            file = files[type_test][i]
            img = imgs[type_test][i]
            score_map = D[type_test][i]
            score_max = D_max
            gt = y[type_test][i]

            plt.figure(figsize=(8, 11), dpi=100, facecolor='white')
            plt.rcParams['font.size'] = 8
            plt.subplot2grid((5, 3), (0, 0), rowspan=1, colspan=1)
            plt.imshow(img)
            plt.title('%s : %s' % (file.split('/')[-2], file.split('/')[-1]))
            plt.subplot2grid((5, 3), (0, 1), rowspan=1, colspan=1)
            plt.imshow(gt)
            plt.subplot2grid((5, 3), (0, 2), rowspan=1, colspan=1)
            plt.imshow(score_map)
            plt.colorbar()
            plt.title('max score : %.2f' % score_max)
            plt.subplot2grid((5, 2), (1, 0), rowspan=2, colspan=1)
            plt.imshow(overlay_heatmap_on_image(img, (score_map / score_max)))
            plt.subplot2grid((5, 2), (1, 1), rowspan=2, colspan=1)
            plt.imshow((img.astype(np.float32) *
                        (score_map / score_max)[..., None]).astype(np.uint8))
            for j_nn, i_nn in enumerate(I_nn[type_test][i][:min(ConfigDraw.k, 6)]):
                img_nn = imgs_nn[i_nn]
                plt.subplot2grid((5, 3), ((j_nn // 3 + 3), (j_nn % 3)), rowspan=1, colspan=1)
                plt.imshow(img_nn)
                if (j_nn == 0):
                    plt.title('TOP %d NN' % min(ConfigDraw.k, 6))
            plt.gcf().savefig(os.path.join(ConfigDraw.path_result, type_data,
                                           ('localization_k%02d_%s_%s_%s' %
                                            (ConfigDraw.k, type_data, type_test,
                                             os.path.basename(file)))))
            plt.clf()
            plt.close()


def draw_roc_curve(fpr_img, tpr_img, rocauc_img, fpr_pix, tpr_pix, rocauc_pix):
    plt.figure(figsize=(12, 6), dpi=100, facecolor='white')
    for type_data in fpr_img.keys():
        plt.subplot(1, 2, 1)
        plt.plot(fpr_img[type_data], tpr_img[type_data],
                 label='%s ROCAUC: %.3f' % (type_data, rocauc_img[type_data]))
        plt.subplot(1, 2, 2)
        plt.plot(fpr_pix[type_data], tpr_pix[type_data],
                 label='%s ROCAUC: %.3f' % (type_data, rocauc_pix[type_data]))

    plt.subplot(1, 2, 1)
    plt.title('imagewise anomaly detection accuracy (ROCAUC %)')
    plt.grid()
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('pixelwise anomaly detection accuracy (ROCAUC %)')
    plt.grid()
    plt.legend()
    plt.gcf().tight_layout()
    plt.gcf().savefig(os.path.join(ConfigDraw.path_result,
                                   ('roc-curve_k%02d.png' % ConfigDraw.k)))
    plt.clf()
    plt.close()
