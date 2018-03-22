#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans

from src.config import *


def pca(X, n_components):
    pca = PCA(n_components)
    X_proj = pca.fit_transform(X)
    return pca.inverse_transform(X_proj)


def get_distance_matrix(features):
    diff = features[:, None] - features
    return np.linalg.norm(diff, axis=-1)


def plot_pca_accuracy(X):
    # Matplotlib settings
    plt.axes().set_xlim(0, 209)
    plt.axes().set_ylim(0, 1)
    plt.grid(True, alpha=0.2)

    # Compute initial Frobenius norm
    norm_X = np.linalg.norm(X, axis=None)

    # Compute accuracy for differend numbers of components
    n_comp_values = np.arange(4, 210, 5)
    accuracies = []
    for n_comp in n_comp_values:
        X_reconstructed = pca(X, n_comp)

        err = np.linalg.norm(X - X_reconstructed, ord='fro', axis=None)
        accuracies.append(1 - err / norm_X)

    # Plot the results
    plt.plot(n_comp_values, accuracies)
    plt.show()


def plot_geo_to_lingu_distance(xy, A, apply_pca=False, n_components=30, fit_intercept=True, ax=None):
    n_munis = len(xy)
    off_diag = ~np.eye(n_munis).astype(bool)

    if apply_pca:
        A = pca(A, n_components)

    # Compute geographic and linguistic distance matrices (then flatten)
    xy_dist = get_distance_matrix(xy)[off_diag]
    A_dist = get_distance_matrix(A)[off_diag]

    print('Pearson correlation coefficient: %.3f' %
          np.corrcoef(xy_dist, A_dist)[0, 1])

    # Fit linear regression (OLS)
    lin_reg = LinearRegression(fit_intercept=fit_intercept)
    lin_reg.fit(xy_dist[:, None], A_dist[:, None])

    # Plot results
    if ax:
        t = np.linspace(min(xy_dist), max(xy_dist), 2000)
        ax.scatter(xy_dist, A_dist, s=3, alpha=0.2, lw=0)
        ax.plot(t, lin_reg.predict(t[:, None]))

        ax.set_ylim(0.)
        ax.set_xlim(0.)


def cummulated_logistic_regression(x, ys):
    n = len(x)
    log_regs = []

    x2 = np.concatenate((x, x))
    y01 = np.zeros(2*n)
    y01[n:] = 1.
    y = np.mean(ys, axis=-1)
    w = np.concatenate((1 - y, y))

    lr = LogisticRegression(C=1E10, penalty='l1')
    lr.fit(x2[:, None], y01, sample_weight=w)

    return lambda x_: lr.predict_proba(x_)[:, 1]
    # for y in ys.T:
    #     lr = LogisticRegression(C=1E10, penalty='l1')
    #
    #     w = np.concatenate((1-y, y))
    #     lr.fit(x2[:, None], y01, sample_weight=w)
    #
    #     log_regs.append(lr)
    #
    # print(len(log_regs))
    #
    # return lambda x_new: sum([lr.predict_proba(x_new)[:, 1] for lr in log_regs])

if __name__ == '__main__':
    answers = pd.read_csv(ANSWERS_PATH, index_col=0)
    municipals = pd.read_csv(MUNICAPALS_PATH, index_col=0).set_index('bfs')

    # Aggregate the municipalities by taking the in-group mean
    muni_answers = answers.groupby('bfs').mean()
    # muni_answers = answers.set_index('bfs')

    # Define column names for linguistic features and geographical features
    answer_colums = muni_answers.columns.values
    xy_columns = ['x', 'y']

    # Join (align) linguistic and geographical data
    muni_answers = muni_answers.join(municipals, how='inner')

    # Extract separate matrices for geo. and lingu. features
    xy = muni_answers[xy_columns].values
    A = muni_answers[answer_colums].values

    n_munis = A.shape[0]

    off_diag = ~np.eye(n_munis).astype(bool)
    xy_dist = get_distance_matrix(xy)[off_diag]
    A_diff = np.abs(A[:, None] - A)[off_diag]
    print(A_diff.shape)
    A_dist = np.mean(A_diff, axis=-1)

    # f = cummulated_logistic_regression(xy_dist, A_diff)
    from sklearn.svm import SVR
    from sklearn.kernel_ridge import KernelRidge
    samples = np.random.random(len(xy_dist)) < 0.01
    f = SVR(kernel='rbf', gamma=1e-9).fit(xy_dist[samples, None], A_dist[samples]).predict

    var_total = np.var(A_dist)
    var_residuals = np.var(f(xy_dist[:, None]) - A_dist)

    print('R²: %.2f' % (1 - var_residuals / var_total))

    # A = pca(A, 50)
    #
    # n_clusters = 4
    #
    # fig, [ax1, ax2] = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
    #
    # ax1.set_title('Clustered Geolocations')
    # ax2.set_title('Euclidean -> Morphosyntactic Distance')
    #
    # k_means = KMeans(n_clusters)
    # k_means.fit(A)
    # A_cluster_idx = k_means.predict(A)
    # for i in range(n_clusters):
    #     ax1.scatter(*xy[A_cluster_idx == i].T)
    #     in_cluster = (A_cluster_idx == i)
    #     plot_geo_to_lingu_distance(xy[in_cluster], A[in_cluster], ax=ax2)
    #
    # ax1.set_aspect('equal')
    # ax1.get_xaxis().set_ticks([])
    # ax1.get_yaxis().set_ticks([])
    #
    # fig.set_size_inches((5, 8.2))

    t = np.linspace(min(xy_dist), max(xy_dist), 2000)
    plt.scatter(xy_dist, A_dist, s=1, alpha=0.2, lw=0, c='grey')
    plt.plot(t, f(t[:, None]))

    plt.axes().set_ylim(0.)
    plt.axes().set_xlim(0.)

    plt.tight_layout()
    plt.show()
