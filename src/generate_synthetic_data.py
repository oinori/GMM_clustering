# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import sys
import csv
import argparse
import time

def parse_args():
    # Parse argments.
    parser = argparse.ArgumentParser(description="Generate synthetic data for Gaussian mixture clustering. ")

    parser.add_argument('--N', type= int , default = 100, help = 'Number of data points. ')

    parser.add_argument('--D', type= int , default = 2, help = 'Dimension of the space.')

    parser.add_argument('--K', type= int , default = 3, help = 'Number of clusters.')

    return parser.parse_args()

def generate_means_and_covs(D, K):
    # give parameters explicitly
    means = np.array([[0, 0], [2, 2], [-5, -10]])
    covs = np.array([[[1, 0], [0, 10]], [[4, 1], [1, 1]], [[5, -2], [-2, 1]]])
    return means, covs

def generate_data_points(N, D, K, means, covs):
    data_points = []
    cluster_ids = []

    for n in range(N):
        cluster_id = np.random.randint(K)
        data_point = np.random.multivariate_normal(means[cluster_id], covs[cluster_id]).T
        cluster_ids.append(cluster_id)
        data_points.append(data_point)

    data_points = np.vstack(data_points)
    cluster_ids = np.array(cluster_ids)

    return data_points, cluster_ids

def visualize_data_points(N, D, K, data_points, cluster_ids):
    if D != 2:
        return

    # make true cluster figure
    plt.clf()
    for k in range(K):
        cluster_points = data_points[cluster_ids==k, :]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color = cm.seismic(k/(1.0*K)))
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.savefig('fig/true_cluster.png', dpi=300)

    # make input figure
    plt.clf()
    plt.scatter(data_points[:, 0], data_points[:, 1])
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.savefig('fig/input_data.png', dpi=300)

    return

def output(N, D, K, data_points, cluster_ids):
    np.savetxt('input/data_points.txt', data_points, delimiter=',')
    np.savetxt('input/cluster_ids.txt', cluster_ids, delimiter=',', fmt='%d')

def main(args):

    means, covs = generate_means_and_covs(args.D, args.K)
    data_points, cluster_ids = generate_data_points(args.N, args.D, args.K, means, covs)

    visualize_data_points(args.N, args.D, args.K, data_points, cluster_ids)

    output(args.N, args.D, args.K, data_points, cluster_ids)

if __name__ == '__main__':
    args = parse_args()
    main(args)