import os
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

flg_gpu = True
batch_size = 300
k = 3
path_parent = './mvtec_anomaly_detection/'

if flg_gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

MEAN = torch.FloatTensor([[[0.485, 0.456, 0.406]]]).to(device)
STD = torch.FloatTensor([[[0.229, 0.224, 0.225]]]).to(device)
SHAPE_INPUT = (224, 224)
NUM_CPU_MAX = 4


def tic():
    #require to import time
    global start_time_tictoc
    start_time_tictoc = time.time()

    
def toc(tag="elapsed time"):
    if "start_time_tictoc" in globals():
        print("{}: {:.9f} [sec]".format(tag, time.time() - start_time_tictoc))
    else:
        print("tic has not been called")


# https://pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet50_2.html
model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
model.eval()
model.to(device)

# https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master/blob/main/main.py#L62
# set model's intermediate outputs
outputs = []

def hook(module, input, output):
    outputs.append(output.cpu().numpy())

model.layer1[-1].register_forward_hook(hook) 
model.layer2[-1].register_forward_hook(hook) 
model.layer3[-1].register_forward_hook(hook) 
model.avgpool.register_forward_hook(hook) 
summary(model, input_size=(1, 3, 224, 224))

types_data = [d for d in os.listdir(path_parent)
              if os.path.isdir(os.path.join(path_parent, d))]
types_data = np.sort(np.array(types_data))
print('types_data =', types_data)

auc_image = []
auc_pixel = []

for type_data in types_data:

    tic()
    path_train = os.path.join(path_parent, type_data, 'train/good')
    files_train = [os.path.join(path_train, f) for f in os.listdir(path_train)
                   if (os.path.isfile(os.path.join(path_train, f)) &
                       ('.png' in f))]
    files_train = np.sort(np.array(files_train))

    types_test = os.listdir(os.path.join(path_parent, type_data, 'test'))
    types_test = np.array(sorted(types_test))

    files_test = {}

    for type_test in types_test:
        path_test = os.path.join(path_parent, type_data, 'test', type_test)
        files_test[type_test] = [os.path.join(path_test, f)
                                 for f in os.listdir(path_test)
                                 if (os.path.isfile(os.path.join(path_test, f)) &
                                     ('.png' in f))]
        files_test[type_test] = np.sort(np.array(files_test[type_test]))

    # create output variable
    # https://zenn.dev/ymd_h/articles/4f965f3bfd510d
    # 載せたい型とサイズの指定
    shape = (len(files_train), SHAPE_INPUT[0], SHAPE_INPUT[1], 3)
    num_elm = shape[0] * shape[1] * shape[2] * shape[3]
    ctype = np.ctypeslib.as_ctypes_type(np.dtype(np.uint8))
    data = np.ctypeslib.as_array(RawArray(ctype, num_elm))
    data.shape = shape
    imgs_train = data.view(np.uint8)

    def read_and_resize(file):
        img = cv2.imread(file)[..., ::-1]  # BGR2RGB
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = img[16:(256-16), 16:(256-16)]
        imgs_train[np.where(files_train == file)[0]] = img

    # read and resize
    mp.set_start_method('fork', force=True)
    p = mp.Pool(min(mp.cpu_count(), NUM_CPU_MAX))

    for _ in tqdm(p.imap_unordered(read_and_resize, files_train),
                  total=len(files_train), desc='read image for train'):
        pass
    p.close()

    imgs_test = {}
    for type_test in types_test:
        # create output variable    
        shape = (len(files_test[type_test]), SHAPE_INPUT[0], SHAPE_INPUT[1], 3)
        num_elm = shape[0] * shape[1] * shape[2] * shape[3]
        ctype = np.ctypeslib.as_ctypes_type(np.dtype(np.uint8))
        data = np.ctypeslib.as_array(RawArray(ctype, num_elm))
        data.shape = shape
        imgs_test[type_test] = data.view(np.uint8)

        def read_and_resize(file):
            img = cv2.imread(file)[..., ::-1]  # BGR2RGB
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img = img[16:(256-16), 16:(256-16)]
            imgs_test[type_test][np.where(files_test[type_test] == file)[0]] = img

        # read and resize
        mp.set_start_method('fork', force=True)
        p = mp.Pool(min(mp.cpu_count(), NUM_CPU_MAX))

        for _ in tqdm(p.imap_unordered(read_and_resize, files_test[type_test]),
                      total=len(files_test[type_test]),
                      desc='read image for test (case:%s)' % type_test):
            pass
        p.close()

    gts_test = {}
    for type_test in types_test:
        # create output variable    
        shape = (len(files_test[type_test]), SHAPE_INPUT[0], SHAPE_INPUT[1])
        if (type_test == 'good'):
            gts_test[type_test] = np.zeros(shape, dtype=np.uint8)
        else:
            num_elm = shape[0] * shape[1] * shape[2]
            ctype = np.ctypeslib.as_ctypes_type(np.dtype(np.uint8))
            data = np.ctypeslib.as_array(RawArray(ctype, num_elm))
            data.shape = shape
            gts_test[type_test] = data.view(np.uint8)

            def read_and_resize(file):
                file_gt = file.replace('/test/', '/ground_truth/')
                file_gt = file_gt.replace('.png', '_mask.png')
                gt = cv2.imread(file_gt, cv2.IMREAD_GRAYSCALE)
                gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)
                gt = gt[16:(256-16), 16:(256-16)]
                if (np.max(gt) != 0):
                    gt = (gt / np.max(gt)).astype(np.uint8)
                gts_test[type_test][np.where(files_test[type_test] == file)[0]] = gt

            # read and resize
            mp.set_start_method('fork', force=True)
            p = mp.Pool(min(mp.cpu_count(), NUM_CPU_MAX))

            for _ in tqdm(p.imap_unordered(read_and_resize, files_test[type_test]),
                          total=len(files_test[type_test]),
                          desc='read ground-truth for test (case:%s)' % type_test):
                pass
            p.close()

    tic()
    x_batch = []
    outputs = []
    for i, img in tqdm(enumerate(imgs_train), desc='feature extract for train'):

        x = torch.from_numpy(img.astype(np.float32)).to(device)
        x = x / 255
        x = x - MEAN
        x = x / STD
        x = x.unsqueeze(0).permute(0, 3, 1, 2)

        x_batch.append(x)

        if (len(x_batch) == batch_size) | (i == (len(imgs_train) - 1)):
            with torch.no_grad():
                _ = model(torch.vstack(x_batch))
            x_batch = []

    f1_train = np.vstack(outputs[0::4])
    f2_train = np.vstack(outputs[1::4])
    f3_train = np.vstack(outputs[2::4])
    fl_train = np.vstack(outputs[3::4]).squeeze(-1).squeeze(-1)

    tic()
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

            if (len(x_batch) == batch_size) | (i == (len(imgs_test[type_test]) - 1)):
                with torch.no_grad():
                    _ = model(torch.vstack(x_batch))
                x_batch = []

        f1_test[type_test] = np.vstack(outputs[0::4])
        f2_test[type_test] = np.vstack(outputs[1::4])
        f3_test[type_test] = np.vstack(outputs[2::4])
        fl_test[type_test] = np.vstack(outputs[3::4]).squeeze(-1).squeeze(-1)

    d = fl_train.shape[1]
    index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 
                                 d, 
                                 faiss.GpuIndexFlatConfig())
    index.add(fl_train)

    I_test = {}
    D_test = {}

    type_test = 'good'
    D, I = index.search(fl_test[type_test], k)
    D_test[type_test] = np.mean(D, axis=1)
    I_test[type_test] = I
    for type_test in types_test[types_test != 'good']:
        D, I = index.search(fl_test[type_test], k)
        D_test[type_test] = np.mean(D, axis=1)
        I_test[type_test] = I

    D_list = []
    y_list = []
    type_test = 'good'
    D_list.append(D_test[type_test])
    y_list.append(np.zeros([len(D_test[type_test])], dtype=np.int16))
    for type_test in types_test[types_test != 'good']:
        D_list.append(D_test[type_test])
        y_list.append(np.ones([len(D_test[type_test])], dtype=np.int16))
    D_list = np.hstack(D_list)
    y_list = np.hstack(y_list)

    # calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(y_list, D_list)
    per_image_rocauc = roc_auc_score(y_list, D_list)
    auc_image.append(per_image_rocauc)

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
    plt.show()

    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (type_data, per_image_rocauc))
    plt.grid()
    plt.legend()
    plt.show()

    print('%s ROCAUC: %.3f' % (type_data, per_image_rocauc))

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
    per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)
    auc_pixel.append(per_pixel_rocauc)
    toc('elapsed time for SPADE processing in %s' % type_data)

    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (type_data, per_pixel_rocauc))
    plt.grid()
    plt.legend()
    plt.show()

    print('%s ROCAUC: %.3f' % (type_data, per_pixel_rocauc))

auc_image = np.array(auc_image)
auc_pixel = np.array(auc_pixel)

print('np.mean(auc_image) =', np.mean(auc_image))
print('np.mean(auc_pixel) =', np.mean(auc_pixel))
