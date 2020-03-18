import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, cluster, plot_confusion_matrix
from Code import process_data
import gower
import seaborn
from yellowbrick.cluster import KElbowVisualizer


def return_k_means_clusters(params, x_data):
    cluster = params['clusters']
    cluster_predictor = KMeans(n_clusters=cluster, random_state=786)
    cluster_labels = cluster_predictor.fit_predict(x_data)
    return cluster_labels


# code adapted from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
def calculate_k_means_clusters(params, x_data, y_data):
    cluster = params['n_clusters'][0]
    cluster_predictor = KMeans(n_clusters=cluster, random_state=786)
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


def run_k_means_elbow(params, x_data):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2, 40), metric='distortion')
    plt.figure()
    visualizer.fit(x_data)
    visualizer.set_title(params['elbow_graph'])

    try:
        path = params['path'] + params['elbow_graph'] + '.png'
    except:
        path = params['elbow_graph'] + '.png'

    visualizer.show(outpath=path)


def main():
    np.random.seed(311)
    # import the datasets
    x_census_data, y_census_data = process_data.process_census_data('./../Datasets/Census_Income')
    x_stock_data, y_stock_data = process_data.process_stock_data('./../Datasets/Stocks', ["2016", "2017"])
    stock_params = {
        'n_clusters': [21],
        'elbow_graph': 'Stock Dataset Elbow Graph - K Means',
        'con_mat': 'Stock Dataset Confusion Matrix - K Means',
        'fig_size': (20, 5)
    }

    census_params = {
        'n_clusters': [8],
        'elbow_graph': 'Census Dataset Elbow Graph - K Means',
        'con_mat': 'Census Dataset Confusion Matrix - K Means',
        'fig_size': (20, 5)
    }

    # Run the elbow technique on the stock dataset and then calculate the clusters using the ideal
    # cluster size, and see what those cluster
    run_k_means_elbow(stock_params, x_stock_data)
    calculate_k_means_clusters(stock_params, x_stock_data, y_stock_data)

    plt.figure()

    run_k_means_elbow(census_params, x_census_data)
    calculate_k_means_clusters(census_params, x_census_data, y_census_data)


if __name__ == '__main__':
    main()