#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
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


def plot_geo_to_lingu_distance(xy, A, apply_pca=True, n_components=30, fit_intercept=True, ax=None):
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



if __name__ == '__main__':
    answers = pd.read_csv(ANSWERS_PATH, index_col=0)
    municipals = pd.read_csv(MUNICAPALS_PATH, index_col=0).set_index('bfs')

    # Aggregate the municipalities by taking the in-group mean
    muni_answers = answers.groupby('bfs').mean()

    # Define column names for linguistic features and geographical features
    answer_colums = muni_answers.columns.values
    xy_columns = ['x', 'y']

    # Join (align) linguistic and geographical data
    muni_answers = muni_answers.join(municipals, how='inner')

    # Extract separate matrices for geo. and lingu. features
    xy = muni_answers[xy_columns].values
    A = muni_answers[answer_colums].values

    A = pca(A, 50)

    n_clusters = 4

    fig, [ax1, ax2] = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

    ax1.set_title('Clustered Geolocations')
    ax2.set_title('Euclidean -> Morphosyntactic Distance')

    k_means = KMeans(n_clusters)
    k_means.fit(A)
    A_cluster_idx = k_means.predict(A)
    for i in range(n_clusters):
        ax1.scatter(*xy[A_cluster_idx == i].T)
        in_cluster = (A_cluster_idx == i)
        plot_geo_to_lingu_distance(xy[in_cluster], A[in_cluster], ax=ax2)

    ax1.set_aspect('equal')
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])

    fig.set_size_inches((5, 8.2))
    plt.tight_layout()

    plt.show()
