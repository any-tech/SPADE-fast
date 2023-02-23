import matplotlib.pyplot as plt
import numpy as np
import os


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
