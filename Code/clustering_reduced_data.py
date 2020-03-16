import Code.process_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Code import principle_component_analysis
from Code import independant_component_analysis
from Code import svd_projection
from Code import random_projections
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, cluster, plot_confusion_matrix
import seaborn
from yellowbrick.cluster import KElbowVisualizer
from sklearn.mixture import GaussianMixture
from Code import process_data
from Code import k_means
from Code import expectation_maximization


def main():
    np.random.seed(311)
    # import the datasets
    x_census_data, y_census_data = process_data.process_census_data('./../Datasets/Census_Income')
    x_stock_data, y_stock_data = process_data.process_stock_data('./../Datasets/Stocks', ["2016", "2017"])
    stock_params = {
        'n_components': 100,
        'filename': None,
        'projection_loss_graph': None,
        'num_retry': 10,
        'ica_title': None,
        'components': 100
    }

    census_params = {
        'n_components': 4,
        'filename': None,
        'projection_loss_graph': None,
        'num_retry': 10,
        'ica_title': None,
        'components': 4
    }

    x_stock_data_reduced = []
    x_census_data_reduced = []

    # import all the reduced dimensionality datasets
    # PCA
    x_stock_data_reduced.append(principle_component_analysis.run_pca(stock_params, x_stock_data))
    x_census_data_reduced.append(principle_component_analysis.run_pca(census_params, x_census_data))

    # ICA
    x_stock_data_reduced.append(independant_component_analysis.run_ica(stock_params, x_stock_data))
    x_census_data_reduced.append(independant_component_analysis.run_ica(census_params, x_census_data))

    # Random Projections
    x_stock_data_reduced.append(random_projections.create_random_guassian_projections(stock_params, x_stock_data))
    x_census_data_reduced.append(random_projections.create_random_guassian_projections(census_params, x_census_data))

    # SVD
    x_stock_data_reduced.append(svd_projection.run_svd(stock_params, x_stock_data))
    x_census_data_reduced.append(svd_projection.run_svd(census_params, x_census_data))

    # Make some rough elbow graphs to figure out how many clusters to do
    km_stock_elbow_dict = {'elbow_graph': 'Stock Dataset K Means Elbow Graph Reduced Dimension',
                           'path': './Reduced_Data_Clustering/'}
    k_means.run_k_means_elbow(km_stock_elbow_dict, x_stock_data_reduced[0])
    km_census_elbow_dict = {'elbow_graph': 'Census Dataset K Means Elbow Graph Reduced Dimension',
                           'path': './Reduced_Data_Clustering/'}
    k_means.run_k_means_elbow(km_census_elbow_dict, x_census_data_reduced[0])


    # Run K means on all these guys
    k_means_stock_params_km = []
    k_means_stock_params_km.append({'n_clusters': [23], 'con_mat': "Stock Dataset Confusion Matrix - K-Means PCA Reduction",
                                 'path': './Reduced_Data_Clustering/', 'fig_size': (20, 5)})
    k_means_stock_params_km.append({'n_clusters': [23], 'con_mat': "Stock Dataset Confusion Matrix - K-Means ICA Reduction",
                                 'path': './Reduced_Data_Clustering/', 'fig_size': (20, 5)})
    k_means_stock_params_km.append({'n_clusters': [23], 'con_mat': "Stock Dataset Confusion Matrix - K-Means Random Projections",
                                 'path': './Reduced_Data_Clustering/', 'fig_size': (20, 5)})
    k_means_stock_params_km.append({'n_clusters': [23], 'con_mat': "Stock Dataset Confusion Matrix - K-Means SVD Reduction",
                                 'path': './Reduced_Data_Clustering/', 'fig_size': (20, 5)})

    k_means_census_params_km = []
    k_means_census_params_km.append({'n_clusters': [8], 'con_mat': "Census Dataset Confusion Matrix - K-Means PCA Reduction",
                                 'path': './Reduced_Data_Clustering/', 'fig_size': (20, 5)})
    k_means_census_params_km.append({'n_clusters': [8], 'con_mat': "Census Dataset Confusion Matrix - K-Means ICA Reduction",
                                 'path': './Reduced_Data_Clustering/', 'fig_size': (20, 5)})
    k_means_census_params_km.append({'n_clusters': [8], 'con_mat': "Census Dataset Confusion Matrix - K-Means Random Projections",
                                 'path': './Reduced_Data_Clustering/', 'fig_size': (20, 5)})
    k_means_census_params_km.append({'n_clusters': [8], 'con_mat': "Census Dataset Confusion Matrix - K-Means SVD Reduction",
                                 'path': './Reduced_Data_Clustering/', 'fig_size': (20, 5)})

    em_stock_params = []
    em_stock_params.append({'n_clusters': [8], 'con_mat': "Stock Dataset Confusion Matrix - EM PCA Reduction",
                            'path': './Reduced_Data_Clustering/', 'fig_size': (20, 5)})
    em_stock_params.append({'n_clusters': [8], 'con_mat': "Stock Dataset Confusion Matrix - EM ICA Reduction",
                            'path': './Reduced_Data_Clustering/', 'fig_size': (20, 5)})
    em_stock_params.append({'n_clusters': [8], 'con_mat': "Stock Dataset Confusion Matrix - EM Random Projections",
                            'path': './Reduced_Data_Clustering/', 'fig_size': (20, 5)})
    em_stock_params.append({'n_clusters': [8], 'con_mat': "Stock Dataset Confusion Matrix - EM SVD Reduction",
                            'path': './Reduced_Data_Clustering/', 'fig_size': (20, 5)})

    em_census_params = []
    em_census_params.append({'n_clusters': [12], 'con_mat': "Census Dataset Confusion Matrix - EM PCA Reduction",
                             'path': './Reduced_Data_Clustering/', 'fig_size': (20, 5)})
    em_census_params.append({'n_clusters': [12], 'con_mat': "Census Dataset Confusion Matrix - EM ICA Reduction",
                             'path': './Reduced_Data_Clustering/', 'fig_size': (20, 5)})
    em_census_params.append({'n_clusters': [12], 'con_mat': "Census Dataset Confusion Matrix - EM Random Projections",
                             'path': './Reduced_Data_Clustering/', 'fig_size': (20, 5)})
    em_census_params.append({'n_clusters': [12], 'con_mat': "Census Dataset Confusion Matrix - EM SVD Reduction",
                             'path': './Reduced_Data_Clustering/', 'fig_size': (20, 5)})

    # loop through all the parameters and datasets to do the analysis
    # 4 sets of 4 analyses = 16 in total
    for i in range(4):
        k_means.calculate_k_means_clusters(k_means_stock_params_km[i], x_stock_data_reduced[i], y_stock_data)
        k_means.calculate_k_means_clusters(k_means_census_params_km[i], x_census_data_reduced[i], y_census_data)
        expectation_maximization.calculate_em_clusters(em_stock_params[i], x_stock_data_reduced[i], y_stock_data)
        expectation_maximization.calculate_em_clusters(em_census_params[i], x_census_data_reduced[i], y_census_data)



if __name__ == '__main__':
    main()