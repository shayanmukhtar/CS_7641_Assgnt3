import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score, cluster, plot_confusion_matrix
from Code import process_data
import gower
import seaborn
from yellowbrick.cluster import KElbowVisualizer


def return_em_clusters(params, x_data):
    cluster = params['clusters']
    cluster_predictor = GaussianMixture(n_components=cluster)
    cluster_labels = cluster_predictor.fit_predict(x_data)
    return cluster_labels


# code adapted from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
def calculate_em_clusters(params, x_data, y_data):
    cluster = params['n_clusters'][0]
    cluster_predictor = GaussianMixture(n_components=cluster)
    cluster_labels = cluster_predictor.fit_predict(x_data)

    # form the confusion matrix
    confusion_matrix = sklearn.metrics.cluster.contingency_matrix(y_data, cluster_labels)
    print(confusion_matrix)

    columns = ["C" + str(d) for d in range(cluster)]
    index = ['T0', 'T1']
    df_cm = pd.DataFrame(data=confusion_matrix, index=index, columns=columns)

    fig, ax = plt.subplots(figsize=params['fig_size'])
    c_plt = seaborn.heatmap(df_cm, annot=True, fmt="d", linewidths=0.5, ax=ax)
    ax.set_ylabel("Ground Truth Labels")
    ax.set_xlabel("Cluster Labels")

    # print the cluster purities based on the ground truth
    purities = np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)
    print("Cluster Purities:")
    print(str(purities))

    c_plt.title.set_text(params['con_mat'] + " Cluster Purities: " + str(purities))
    try:
        path = params['path'] + params['con_mat'] + ".png"
    except:
        path = params['con_mat'] + ".png"
    plt.savefig(path)

    print("Homegenity Score:")
    print(str(sklearn.metrics.homogeneity_score(y_data, cluster_labels)))

    try:
        file_path = params['cluster_center_file']
    except:
        file_path = None

    if file_path is not None:
        with open(file_path, 'w+') as file_out:
            i = 0
            for cluster_center in cluster_predictor.means_:
                file_out.write(str(i) + "\t" + str(cluster_center) + "\n")
                i += 1


# adapted from https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
def run_em_elbow(params, x_data, y_data):
    n_components = range(2, 30)

    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(x_data)
              for n in n_components]

    cluster_labels = [model.fit_predict(x_data) for model in models]
    con_mats = [sklearn.metrics.cluster.contingency_matrix(y_data, cluster_label) for cluster_label in cluster_labels]

    purities = [np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix) for confusion_matrix in con_mats]

    plt.figure()
    plt.plot(n_components, purities)
    plt.xlabel('n_components')
    plt.ylabel("Cluster Purity")

    plt.title(params['elbow_graph'])
    plt.savefig(params['elbow_graph'] + ".png")


def main():
    np.random.seed(311)
    # import the datasets
    x_census_data, y_census_data = process_data.process_census_data('./../Datasets/Census_Income')
    x_stock_data, y_stock_data = process_data.process_stock_data('./../Datasets/Stocks', ["2016", "2017"])
    stock_params = {
        'n_clusters': [8],
        'elbow_graph': 'Stock Dataset Elbow Graph - EM',
        'con_mat': 'Stock Dataset Confusion Matrix - EM',
        'fig_size': (20, 5),
        'cluster_center_file': "stock_em_cluster_centers.txt"
    }

    census_params = {
        'n_clusters': [12],
        'elbow_graph': 'Census Dataset Elbow Graph - EM',
        'con_mat': 'Census Dataset Confusion Matrix - EM',
        'fig_size': (20, 5),
        'cluster_center_file': "census_em_cluster_centers.txt"
    }

    # Run the elbow technique on the stock dataset and then calculate the clusters using the ideal
    # cluster size, and see what those cluster
    run_em_elbow(stock_params, x_stock_data, y_stock_data)
    calculate_em_clusters(stock_params, x_stock_data, y_stock_data)

    plt.figure()

    run_em_elbow(census_params, x_census_data, y_census_data)
    calculate_em_clusters(census_params, x_census_data, y_census_data)


if __name__ == '__main__':
    main()