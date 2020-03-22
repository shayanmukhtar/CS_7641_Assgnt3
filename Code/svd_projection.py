from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import process_data


def run_svd_reconstruction(params, x_data):
    variances = params['variance']
    clfs = [TruncatedSVD(variance) for variance in variances]
    x_data_news = [clf.fit_transform(x_data) for clf in clfs]
    x_projection_losses = []

    # evaluate loss for the PCA
    # https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn

    for i in range(len(variances)):
        print(str(i))
        x_projection = clfs[i].inverse_transform(x_data_news[i])
        x_projection_loss = ((x_data - x_projection)**2).mean()
        x_projection_losses.append(x_projection_loss)

    plt.figure()
    plt.plot(variances, x_projection_losses)
    plt.xlabel('Components Kept')
    plt.ylabel("Reconstruction Error")
    plt.title(params['svd_graph_recon'])
    plt.savefig(params['svd_graph_recon'] + ".png")


def run_svd(params, x_data, y_data=None):
    clf = TruncatedSVD(params['n_components'])
    x_data_new = clf.fit_transform(x_data)

    if params['filename'] is not None:
        with open(params['filename'], 'w+') as fileOut:
            fileOut.write("Eigenvectors: " + "\n")
            for i in range(0, params['n_components']):
                fileOut.write(str(clf.components_[i, :]) + "\n")

            fileOut.write("" + "\n")
            fileOut.write("" + "\n")
            fileOut.write("Eigenvalues and explained variance percentage:" + "\n")
            for i in range(0, params['n_components']):
                fileOut.write("Eigenvalue: " + str(clf.explained_variance_[i]) +
                              "\t\tExplained Variance Percentage: " + str(clf.explained_variance_ratio_[i]) + "\n")

    try:
        graph_name = params['svd_graph']
    except:
        graph_name = None

    if graph_name is not None and y_data is not None:
        # graph the data against its principal components
        plt.figure()
        plt.scatter(x_data_new[:, 0], x_data_new[:, 1], c=y_data, cmap=plt.cm.get_cmap('RdYlBu', 10))
        plt.xlabel("Component One")
        plt.ylabel("Componenet Two")
        plt.title(params['svd_graph'])
        plt.savefig(params['svd_graph'] + ".png")

    return x_data_new


def main():
    np.random.seed(311)
    # import the datasets
    x_census_data, y_census_data = process_data.process_census_data('./../Datasets/Census_Income')
    x_stock_data, y_stock_data = process_data.process_stock_data('./../Datasets/Stocks', ["2016", "2017"])

    stock_params = {
        'variance': range(1, 221),
        'name': "Stock Data",
        'svd_graph_recon': "Stock Data Reconstruction Graph of SVD",
        'n_components': 100,
        'filename': "stock_data_svd.txt",
        'svd_graph': "Stock Data - Data Plotted on Principal Axes using SVD"
    }

    census_params = {
        'variance': range(1, 13),
        'name': "Census Data",
        'svd_graph_recon': "Census Data Reconstruction Graph of SVD",
        'n_components': 4,
        'filename': "census_data_svd.txt",
        'svd_graph': "Census Data - Data Plotted on Principal Axes using SVD"
    }

    run_svd_reconstruction(stock_params, x_stock_data)
    run_svd_reconstruction(census_params, x_census_data)
    run_svd(stock_params, x_stock_data, y_stock_data)
    run_svd(census_params, x_census_data, y_census_data)


if __name__ == '__main__':
    main()