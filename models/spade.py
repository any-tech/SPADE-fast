import numpy as np
import cv2
from tqdm import tqdm
import faiss
from scipy.ndimage import gaussian_filter


class Spade:
    def __init__(self, cfg_spade, depth):
        self.k = cfg_spade.k
        self.shape_stretch = cfg_spade.shape_stretch
        self.pixel_outer_decay = cfg_spade.pixel_outer_decay

        # prep knn index
        self.index_feat_map = []
        for i_depth in range(len(depth) - 1):
            self.index_feat_map.append(faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),
                                                            depth[i_depth],
                                                            faiss.GpuIndexFlatConfig()))
        self.index_feat_vec = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),
                                                   depth[3],
                                                   faiss.GpuIndexFlatConfig())

    def search_nearest_neighbor(self, feat_vec_train, feat_vec_test):
        # make feature gallery for imagewise knn
        self.index_feat_vec.reset()
        self.index_feat_vec.add(feat_vec_train)
        k = self.k

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
        for type_test in feat_map_test.keys():
            D_pix[type_test] = []

            # loop for test data
            num_data = len(feat_map_test[type_test][0])
            for i in tqdm(range(num_data),
                          desc='localization (case:%s)' % type_test):
                # pickup test data
                feat_map_test_ = [feat_map_test[type_test][0][i],
                                  feat_map_test[type_test][1][i],
                                  feat_map_test[type_test][2][i]]
                # measure distance pixelwise
                score_map = self.measure_dist_pixelwise(feat_map_train=feat_map_train,
                                                        feat_map_test=feat_map_test_,
                                                        I_nn=I_nn[type_test][i])
                # adjust score of outer-pixel (provisional heuristic algorithm)
                if (self.pixel_outer_decay > 0):
                    score_map[:self.pixel_outer_decay, :] *= 0.6
                    score_map[-self.pixel_outer_decay:, :] *= 0.6
                    score_map[:, :self.pixel_outer_decay] *= 0.6
                    score_map[:, -self.pixel_outer_decay:] *= 0.6
                # stock score map
                D_pix[type_test].append(score_map)
                score_max = max(score_max, np.max(score_map))

            # cast list to numpy array
            D_pix[type_test] = np.array(D_pix[type_test])

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
            score_map = cv2.resize(score_map, (self.shape_stretch[0],
                                               self.shape_stretch[1]))
            score_map_mean.append(score_map)

        # average distance between the features
        score_map_mean = np.mean(np.array(score_map_mean), axis=0)
        # apply gaussian smoothing on the score map
        score_map_smooth = gaussian_filter(score_map_mean, sigma=4)

        return score_map_smooth
