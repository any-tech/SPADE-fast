import os
import argparse
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


def tic():
    #require to import time
    global start_time_tictoc
    start_time_tictoc = time.time()


def toc(tag="elapsed time"):
    if "start_time_tictoc" in globals():
        print("{}: {:.1f} [sec]".format(tag, time.time() - start_time_tictoc))
    else:
        print("tic has not been called")


def parse_args():
    parser = argparse.ArgumentParser('SPADE')
    parser.add_argument('-k', '--k', type=int, default=5, help='nearest neighbor\'s k')
    parser.add_argument('-b', '--batch_size', type=int, default=256,
                        help='batch-size for feature extraction from ImageNet model')
    parser.add_argument('-pp', '--path_parent', type=str, default='./mvtec_anomaly_detection',
                        help='parent path of data input path')
    parser.add_argument('-pr', '--path_result', type=str, default='./result',
                        help='output path of figure image as the evaluation result')
    parser.add_argument('-c', '--cpu', action='store_true', help='use cpu')
    return parser.parse_args()


args = parse_args()
print('args =\n', args)

if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')  # default

os.makedirs(args.path_result, exist_ok=True)

NUM_CPU_MAX = 4  # for exec imread and imresize on multiprocess

# https://pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet50_2.html#torchvision.models.Wide_ResNet50_2_Weights
MEAN = torch.FloatTensor([[[0.485, 0.456, 0.406]]]).to(device)
STD = torch.FloatTensor([[[0.229, 0.224, 0.225]]]).to(device)
SHAPE_MIDDLE = (256, 256)  # (H, W)
SHAPE_INPUT = (224, 224)  # (H, W)

pixel_crop = (int(abs(SHAPE_MIDDLE[0] - SHAPE_INPUT[0]) / 2),
              int(abs(SHAPE_MIDDLE[1] - SHAPE_INPUT[1]) / 2))  # (H, W)

model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
model.eval()
model.to(device)
print('model =\n')
summary(model, input_size=(1, 3, SHAPE_INPUT[0], SHAPE_INPUT[1]))

# https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master/blob/main/main.py#L62
# set model's intermediate outputs
outputs = []

def hook(module, input, output):
    outputs.append(output.cpu().numpy())

model.layer1[-1].register_forward_hook(hook)
model.layer2[-1].register_forward_hook(hook)
model.layer3[-1].register_forward_hook(hook)
model.avgpool.register_forward_hook(hook)

types_data = [d for d in os.listdir(args.path_parent)
              if os.path.isdir(os.path.join(args.path_parent, d))]
types_data = np.sort(np.array(types_data))
print('types_data =', types_data)

fpr_image = {}
tpr_image = {}
rocauc_image = {}
fpr_pixel = {}
tpr_pixel = {}
rocauc_pixel = {}

for type_data in types_data:

    tic()

    # collect filename of train
    path_train = os.path.join(args.path_parent, type_data, 'train/good')
    files_train = [os.path.join(path_train, f) for f in os.listdir(path_train)
                   if (os.path.isfile(os.path.join(path_train, f)) &
                       ('.png' in f))]  # only .png files exist in mvtec
    files_train = np.sort(np.array(files_train))

    # collect test-type of test
    types_test = os.listdir(os.path.join(args.path_parent, type_data, 'test'))
    types_test = np.array(sorted(types_test))

    # collect filename of test
    files_test = {}
    for type_test in types_test:
        path_test = os.path.join(args.path_parent, type_data, 'test', type_test)
        files_test[type_test] = [os.path.join(path_test, f)
                                 for f in os.listdir(path_test)
                                 if (os.path.isfile(os.path.join(path_test, f)) &
                                     ('.png' in f))]  # only .png files exist in mvtec
        files_test[type_test] = np.sort(np.array(files_test[type_test]))

    # create memory shared variable
    # https://zenn.dev/ymd_h/articles/4f965f3bfd510d
    shape = (len(files_train), SHAPE_INPUT[0], SHAPE_INPUT[1], 3)
    num_elm = shape[0] * shape[1] * shape[2] * shape[3]
    ctype = np.ctypeslib.as_ctypes_type(np.dtype(np.uint8))
    data = np.ctypeslib.as_array(RawArray(ctype, num_elm))
    data.shape = shape
    imgs_train = data.view(np.uint8)

    # define function for parallel
    def read_and_resize(file):
        img = cv2.imread(file)[..., ::-1]  # BGR2RGB
        img = cv2.resize(img, (SHAPE_MIDDLE[1], SHAPE_MIDDLE[0]),
                         interpolation=cv2.INTER_AREA)  # from the paper
        img = img[pixel_crop[0]:(SHAPE_INPUT[0] + pixel_crop[0]),
                  pixel_crop[1]:(SHAPE_INPUT[1] + pixel_crop[1])]
        imgs_train[np.where(files_train == file)[0]] = img

    # exec imread and imresize on multiprocess
    mp.set_start_method('fork', force=True)
    p = mp.Pool(min(mp.cpu_count(), NUM_CPU_MAX))

    for _ in tqdm(p.imap_unordered(read_and_resize, files_train),
                  total=len(files_train), desc='read image for train'):
        pass
    p.close()

    imgs_test = {}
    for type_test in types_test:
        # create memory shared variable
        shape = (len(files_test[type_test]), SHAPE_INPUT[0], SHAPE_INPUT[1], 3)
        num_elm = shape[0] * shape[1] * shape[2] * shape[3]
        ctype = np.ctypeslib.as_ctypes_type(np.dtype(np.uint8))
        data = np.ctypeslib.as_array(RawArray(ctype, num_elm))
        data.shape = shape
        imgs_test[type_test] = data.view(np.uint8)

        # define function for parallel
        def read_and_resize(file):
            img = cv2.imread(file)[..., ::-1]  # BGR2RGB
            img = cv2.resize(img, (SHAPE_MIDDLE[1], SHAPE_MIDDLE[0]),
                             interpolation=cv2.INTER_AREA)  # from the paper
            img = img[pixel_crop[0]:(SHAPE_INPUT[0] + pixel_crop[0]),
                      pixel_crop[1]:(SHAPE_INPUT[1] + pixel_crop[1])]
            imgs_test[type_test][np.where(files_test[type_test] == file)[0]] = img

        # exec imread and imresize on multiprocess
        mp.set_start_method('fork', force=True)
        p = mp.Pool(min(mp.cpu_count(), NUM_CPU_MAX))

        for _ in tqdm(p.imap_unordered(read_and_resize, files_test[type_test]),
                      total=len(files_test[type_test]),
                      desc='read image for test (case:%s)' % type_test):
            pass
        p.close()

    gts_test = {}
    for type_test in types_test:
        # create memory shared variable
        shape = (len(files_test[type_test]), SHAPE_INPUT[0], SHAPE_INPUT[1])
        if (type_test == 'good'):
            gts_test[type_test] = np.zeros(shape, dtype=np.uint8)
        else:
            num_elm = shape[0] * shape[1] * shape[2]
            ctype = np.ctypeslib.as_ctypes_type(np.dtype(np.uint8))
            data = np.ctypeslib.as_array(RawArray(ctype, num_elm))
            data.shape = shape
            gts_test[type_test] = data.view(np.uint8)

            # define function for parallel
            def read_and_resize(file):
                file_gt = file.replace('/test/', '/ground_truth/')
                file_gt = file_gt.replace('.png', '_mask.png')
                gt = cv2.imread(file_gt, cv2.IMREAD_GRAYSCALE)
                gt = cv2.resize(gt, (SHAPE_MIDDLE[1], SHAPE_MIDDLE[0]),
                                interpolation=cv2.INTER_NEAREST)
                gt = gt[pixel_crop[0]:(SHAPE_INPUT[0] + pixel_crop[0]),
                        pixel_crop[1]:(SHAPE_INPUT[1] + pixel_crop[1])]
                if (np.max(gt) != 0):
                    gt = (gt / np.max(gt)).astype(np.uint8)
                gts_test[type_test][np.where(files_test[type_test] == file)[0]] = gt

            # exec imread and imresize on multiprocess
            mp.set_start_method('fork', force=True)
            p = mp.Pool(min(mp.cpu_count(), NUM_CPU_MAX))

            for _ in tqdm(p.imap_unordered(read_and_resize, files_test[type_test]),
                          total=len(files_test[type_test]),
                          desc='read ground-truth for test (case:%s)' % type_test):
                pass
            p.close()

    # feature extract for train
    x_batch = []
    outputs = []
    for i, img in tqdm(enumerate(imgs_train), desc='feature extract for train'):

        x = torch.from_numpy(img.astype(np.float32)).to(device)
        x = x / 255
        x = x - MEAN
        x = x / STD
        x = x.unsqueeze(0).permute(0, 3, 1, 2)

        x_batch.append(x)

        if (len(x_batch) == args.batch_size) | (i == (len(imgs_train) - 1)):
            with torch.no_grad():
                _ = model(torch.vstack(x_batch))
            x_batch = []

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
        for i, img in tqdm(enumerate(imgs_test[type_test]),
                           desc='feature extract for test (case:%s)' % type_test):

            x = torch.from_numpy(img.astype(np.float32)).to(device)
            x = x / 255
            x = x - MEAN
            x = x / STD
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
                             np.hstack([D_test[type_test] for type_test
                                        in types_test[types_test != 'good']])])
    y_list = np.concatenate([y_test['good'],
                             np.hstack([y_test[type_test] for type_test
                                        in types_test[types_test != 'good']])])

    # calculate per-image level ROCAUC
    fpr, tpr, _ = roc_curve(y_list, D_list)
    rocauc = roc_auc_score(y_list, D_list)
    print('%s per-image level ROCAUC: %.3f' % (type_data, rocauc))

    # stock for output result
    fpr_image[type_data] = fpr
    tpr_image[type_data] = tpr
    rocauc_image[type_data] = rocauc

    # prep work variable for measure
    flatten_gt_mask_list = []
    flatten_score_map_list = []

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
        for i, img in tqdm(enumerate(imgs_test[type_test]),
                           desc='localization (case:%s)' % type_test):
            gt = gts_test[type_test][i]

            score_map_mean = []
            for i_nn in range(3):
                # construct a gallery of features at all pixel locations of the K nearest neighbors
                if (i_nn == 0):
                    f_neighbor = f1_train[I_test[type_test][i]]
                    f_query = f1_test[type_test][[i]]
                    index_ = index1
                elif (i_nn == 1):
                    f_neighbor = f2_train[I_test[type_test][i]]
                    f_query = f2_test[type_test][[i]]
                    index_ = index2
                elif (i_nn == 2):
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
                score_map = cv2.resize(score_map, (224, 224))
                score_map_mean.append(score_map)

            # average distance between the features
            score_map_mean = np.mean(np.array(score_map_mean), axis=0)

            # apply gaussian smoothing on the score map
            score_map_smooth = gaussian_filter(score_map_mean, sigma=4)

            flatten_gt_mask = np.concatenate(gt).ravel()
            flatten_score_map = np.concatenate(score_map_smooth).ravel()
            flatten_gt_mask_list.append(flatten_gt_mask)
            flatten_score_map_list.append(flatten_score_map)

    flatten_gt_mask_list = np.array(flatten_gt_mask_list).reshape(-1)
    flatten_score_map_list = np.array(flatten_score_map_list).reshape(-1)

    # calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(flatten_gt_mask_list, flatten_score_map_list)
    rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)
    print('%s per-pixel level ROCAUC: %.3f' % (type_data, rocauc))

    # stock for output result
    fpr_pixel[type_data] = fpr
    tpr_pixel[type_data] = tpr
    rocauc_pixel[type_data] = rocauc

    toc('elapsed time for SPADE processing in %s' % type_data)
    print('--------------------------------')

    plt.figure(figsize=(10, 8), dpi=100, facecolor='white')
    N_test = 0
    type_test = 'good'
    plt.subplot(2, 1, 1)
    plt.scatter((np.arange(len(D_test[type_test])) + N_test), 
                D_test[type_test], alpha=0.5, label=type_test)
    plt.subplot(2, 1, 2)
    plt.hist(D_test[type_test], alpha=0.5, label=type_test,
             bins=int(np.max(D_test[type_test])//2))
    N_test += len(D_test[type_test])
    for type_test in types_test[types_test != 'good']:
        plt.subplot(2, 1, 1)
        plt.scatter((np.arange(len(D_test[type_test])) + N_test), 
                    D_test[type_test], alpha=0.5, label=type_test)
        plt.subplot(2, 1, 2)
        plt.hist(D_test[type_test], alpha=0.5, label=type_test,
                 bins=int(np.max(D_test[type_test])//2))
        N_test += len(D_test[type_test])

    plt.subplot(2, 1, 1)
    plt.grid()
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.grid()
    plt.legend()
    plt.gcf().tight_layout()
    plt.gcf().savefig(os.path.join(args.path_result,
                                   ('pred_dist_k%02d_%s.png' % (args.k, type_data))))

plt.figure(figsize=(12, 6), dpi=100)
for type_data in types_data:

    plt.subplot(1, 2, 1)
    plt.plot(fpr_image[type_data], tpr_image[type_data],
             label='%s ROCAUC: %.3f' % (type_data, rocauc_image[type_data]))
    plt.subplot(1, 2, 2)
    plt.plot(fpr_pixel[type_data], tpr_pixel[type_data],
             label='%s ROCAUC: %.3f' % (type_data, rocauc_pixel[type_data]))

plt.subplot(1, 2, 1)
plt.title('Image-level anomaly detection accuracy (ROCAUC %)')
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.title('Pixel-level anomaly detection accuracy (ROCAUC %)')
plt.grid()
plt.legend()
plt.gcf().tight_layout()
plt.gcf().savefig(os.path.join(args.path_result, ('roc_curve_k%02d.png' % args.k)))

rocauc_image_ = np.array([rocauc_image[type_data] for type_data in types_data])
rocauc_pixel_ = np.array([rocauc_pixel[type_data] for type_data in types_data])

print('np.mean(auc_image) = %.3f' % np.mean(rocauc_image_))
print('np.mean(auc_pixel) = %.3f' % np.mean(rocauc_pixel_))
