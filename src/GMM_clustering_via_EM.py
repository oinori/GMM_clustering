import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import sys
import csv
import os
import shutil
import argparse
import time
from scipy.stats import multivariate_normal

def parse_args():
    # Parse argments.
    parser = argparse.ArgumentParser(description="MLE via EM algorithm for Gaussian mixture clustering. ")

    parser.add_argument('--K', type = int , default = 3, help = 'Number of clusters.')

    parser.add_argument('--eps', type = float , default = 5, help = 'threshold for stopping estimation.')

    return parser.parse_args()

def load_data():
    data_points = np.loadtxt(fname='input/data_points.txt', delimiter=',')
    cluster_ids_true = np.loadtxt(fname='input/cluster_ids.txt', delimiter=',')
    return data_points, cluster_ids_true

class GMM:
    def __init__(self, K, data_points, eps):
        self.N, self.D = data_points.shape
        self.K = K
        self.data_points = data_points
        self.EPS = 10**(-eps)
        self.mus = np.random.randint(-3, 3, (self.K, self.D))

        diagonal = np.random.randint(1, 4, (self.K, self.D))
        self.sigmas = np.array([np.diag(diagonal[k, :]) for k in range(self.K)])

        self.p = np.random.random(3)
        self.p /= np.sum(self.p)

        self.gammas = np.zeros((self.N, self.K))
        for k in range(self.K):
            rv = multivariate_normal(self.mus[k], self.sigmas[k])
            gamma = rv.pdf(self.data_points)
            self.gammas[:, k] = gamma

        self.LL_list = []
        self.itr = 0

    def update_gammas(self):
        terms = np.zeros((self.N, self.K))
        for k in range(self.K):
            rv = multivariate_normal(self.mus[k], self.sigmas[k])
            term = rv.pdf(self.data_points)
            terms[:, k] = term

        self.gammas = np.diag(1.0/np.sum(terms, axis=1)).dot(terms)

    def update_mus(self):
        Nk = np.sum(self.gammas, axis=0)
        self.mus = np.diag(1.0/Nk).dot((self.gammas.T).dot(self.data_points))

    def update_sigmas(self):
        Nk = np.sum(self.gammas, axis=0)
        new_sigmas = np.zeros((self.K, self.D, self.D))
        for k in range(self.K):
            sigma = np.zeros((self.D, self.D))
            for n in range(self.N):
                sigma += (1.0/Nk[k]) * self.gammas[n][k] * np.tensordot(self.data_points[n] - self.mus[k], self.data_points[n] - self.mus[k], axes=0)
            new_sigmas[k, :, :] = sigma
        self.sigmas = new_sigmas

    def update_p(self):
        self.p = np.sum(self.gammas, axis=0)/self.N

    def calc_LL(self):
        terms = np.zeros((self.N, self.K))
        for k in range(self.K):
            rv = multivariate_normal(self.mus[k], self.sigmas[k])
            term = rv.pdf(self.data_points)
            terms[:, k] = term
        return np.sum(np.log(np.sum(terms, axis=1)))

    def visualize_cluster(self, output_path, dpi):
        plt.clf()
        aloc = np.argmax(self.gammas, axis=1)
        for k in range(self.K):
            col = cm.seismic(k/(1.0*self.K))
            plot_points = self.data_points[aloc==k]
            plt.plot(self.mus[k, 0], self.mus[k, 1], color=col, marker='D', markersize=10)
            plt.scatter(plot_points[:, 0], plot_points[:, 1], color=col)
        plt.title('iteration: {}, log likelihood: {}'.format(self.itr, self.calc_LL()))
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.savefig(output_path, dpi=dpi)

    def fit(self):
        LL = self.calc_LL()
        itr = 0

        fig_dir = 'fig/progress'
        shutil.rmtree(fig_dir)
        os.makedirs(fig_dir)

        while 1:
            print>>sys.stderr, LL
            self.LL_list.append(LL)
            fig_path = fig_dir + '/itr_{}.png'.format(self.itr)
            self.visualize_cluster(fig_path, None)
            self.update_gammas()
            self.update_mus()
            self.update_sigmas()
            self.update_p()

            self.itr += 1
            new_LL = self.calc_LL()
            if np.abs(LL - new_LL) < self.EPS:
                break
            LL = new_LL

def main(args):
    data_points, cluster_ids_true = load_data()
    gmm = GMM(args.K, data_points, args.eps)
    gmm.fit()
    gmm.visualize_cluster('fig/clustered.png', 300)

if __name__ == '__main__':
    args = parse_args()
    main(args)