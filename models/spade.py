import numpy as np
import cv2
from tqdm import tqdm
import faiss
from scipy.ndimage import gaussian_filter
from utils.config import ConfigSpade


class Spade:
    def __init__(self, depth):
        self.index_feat_vec = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),
                                                   depth[3],
                                                   faiss.GpuIndexFlatConfig())
        self.index_feat_map_l1 = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),
                                                      depth[0],
                                                      faiss.GpuIndexFlatConfig())
        self.index_feat_map_l2 = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),
                                                      depth[1],
                                                      faiss.GpuIndexFlatConfig())
        self.index_feat_map_l3 = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),
                                                      depth[2],
                                                      faiss.GpuIndexFlatConfig())
        self.index_feat_map = [self.index_feat_map_l1, self.index_feat_map_l2,
                               self.index_feat_map_l3]

    def search_nearest_neighbor(self, feat_vec_train, feat_vec_test):
        # make feature gallery for imagewise knn
        self.index_feat_vec.reset()
        self.index_feat_vec.add(feat_vec_train)
        k = ConfigSpade.k

        D = {}
        I = {}
        y = {}

        # knn with final layer feature vector
        for type_test in feat_vec_test.keys():
            D_tmp, I_tmp = self.index_feat_vec.search(feat_vec_test[type_test], k)
            D[type_test] = np.mean(D_tmp, axis=1)
            I[type_test] = I_tmp
            if (type_test == 'good'):
                y[type_test] = np.zeros([len(D_tmp)], dtype=np.int16)
            else:
                y[type_test] = np.ones([len(D_tmp)], dtype=np.int16)

        return D, I, y

    def localization(self, feat_map_train, feat_map_test, I_nn):
        D_pix = {}
        score_max = -9999
        # loop for test cases
        for type_test in feat_map_test[0].keys():
            D_pix[type_test] = []

            # loop for test data
            for i, gt in tqdm(enumerate(feat_map_test[0][type_test]),
                              desc='localization (case:%s)' % type_test):
                # pickup test data
                feat_map_test_ = [feat_map_test[0][type_test][i],
                                  feat_map_test[1][type_test][i],
                                  feat_map_test[2][type_test][i]]
                # measure distance pixelwise
                score_map = self.measure_dist_pixelwise(feat_map_train=feat_map_train,
                                                        feat_map_test=feat_map_test_,
                                                        I_nn=I_nn[type_test][i])
                D_pix[type_test].append(score_map)
                score_max = max(score_max, np.max(score_map))
        return D_pix, score_max

    def measure_dist_pixelwise(self, feat_map_train, feat_map_test, I_nn):
        score_map_mean = []
        # loop for layers
        for i_layer in range(len(feat_map_train)):
            # construct a gallery of features at all pixel locations of the K nearest neighbors
            feat_map_neighbor = feat_map_train[i_layer][I_nn]
            feat_map_query = feat_map_test[i_layer][None]
            index = self.index_feat_map[i_layer]

            # get shape
            _, C, H, W = feat_map_neighbor.shape

            # adjust dimensions to measure distance in the channel dimension for all combinations
            feat_map_neighbor = feat_map_neighbor.transpose(0, 2, 3, 1)  # (K, C, H, W) -> (K, H, W, C)
            feat_map_neighbor = feat_map_neighbor.reshape(-1, C)         # (K, H, W, C) -> (KHW, C)
            feat_map_query = feat_map_query.transpose(0, 2, 3, 1)        # (K, C, H, W) -> (K, H, W, C)
            feat_map_query = feat_map_query.reshape(-1, C)               # (K, H, W, C) -> (KHW, C)

            # k nearest features from the gallery (k=1)
            index.reset()
            index.add(feat_map_neighbor)
            D, _ = index.search(feat_map_query, 1)

            # transform to scoremap
            score_map = D.reshape(H, W)
            score_map = cv2.resize(score_map, (ConfigSpade.shape_stretch[0],
                                               ConfigSpade.shape_stretch[1]))
            score_map_mean.append(score_map)

        # average distance between the features
        score_map_mean = np.mean(np.array(score_map_mean), axis=0)
        # apply gaussian smoothing on the score map
        score_map_smooth = gaussian_filter(score_map_mean, sigma=4)

        return score_map_smooth
