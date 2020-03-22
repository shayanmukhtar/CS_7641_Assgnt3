from sklearn.decomposition import FastICA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import process_data


# taken from https://github.com/akcarsten/Independent_Component_Analysis
def kurt(x):
    n = np.shape(x)[0]
    mean = np.sum((x**1)/n) # Calculate the mean
    var = np.sum((x-mean)**2)/n # Calculate the variance
    skew = np.sum((x-mean)**3)/n # Calculate the skewness
    kurt = np.sum((x-mean)**4)/n # Calculate the kurtosis
    kurt = kurt/(var**2)-3

    return kurt, skew, var, mean


def run_ica_reconstruction(params, x_data):
    variances = params['variance']
    clfs = [FastICA(variance) for variance in variances]
    x_data_news = [clf.fit_transform(x_data) for clf in clfs]
    x_projection_losses = []

    # evaluate loss for the PCA
    # https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn

    for i in range(len(variances)):
        x_projection = clfs[i].inverse_transform(x_data_news[i])
        x_projection_loss = ((x_data - x_projection)**2).mean()
        x_projection_losses.append(x_projection_loss)

    plt.figure()
    plt.plot(variances, x_projection_losses)
    plt.xlabel('Independent Components Kept')
    plt.ylabel("Reconstruction Error")
    plt.title(params['ica_graph_recon'])
    plt.savefig(params['ica_graph_recon'] + ".png")


def run_ica(params, x_data, y_data=None):
    clf = FastICA(params['n_components'])
    x_data_new = clf.fit_transform(x_data)

    # just debugging to make sure that X = SAtranspose is true
    x_data_recon = np.dot(x_data_new, clf.mixing_.T) + clf.mean_
    kurtosis_old, _, _, _ = kurt(x_data[:, 0])
    kurtosis_new, _, _, _ = kurt(x_data_new[:, 0])

    if params['ica_title'] is not None:
        fig, ax1 = plt.subplots()
        ax1.set_ylabel('Original', color="tab:red")
        ax1.plot(x_data[:, 0], color="tab:red")

        ax2 = ax1.twinx()
        ax2.set_ylabel("New", color="tab:blue")
        ax2.plot(x_data_new[:, 0], color="tab:blue")
        plt.title(params['ica_title'])
        fig.tight_layout()
        plt.savefig(params['ica_title'] + ".png")

    if params['filename'] is not None:
        with open(params['filename'], 'w+') as fileOut:
            fileOut.write("Kurtosis of 1st Component Original Dataset: " + str(kurtosis_old) + "\n")
            fileOut.write("Kurtosis of 1st Component Transformed Data: " + str(kurtosis_new) + "\n")
            fileOut.write("\n\n")
            fileOut.write("Mixing Matrix\n")

            for i in range(len(clf.mixing_.T)):
                fileOut.write(str(clf.mixing_.T[i]) + "\n")

    try:
        graph_name = params['ica_graph']
    except:
        graph_name = None

    if graph_name is not None and y_data is not None:
        # graph the data against its principal components
        plt.figure()
        plt.scatter(x_data_new[:, 0], x_data_new[:, 1], c=y_data, cmap=plt.cm.get_cmap('RdYlBu', 10))
        plt.xlabel("Component One")
        plt.ylabel("Componenet Two")
        plt.title(params['ica_graph'])
        plt.savefig(params['ica_graph'] + ".png")

    return x_data_new


def main():
    np.random.seed(311)
    # import the datasets
    x_census_data, y_census_data = process_data.process_census_data('./../Datasets/Census_Income')
    x_stock_data, y_stock_data = process_data.process_stock_data('./../Datasets/Stocks', ["2016", "2017"])

    stock_params = {
        'variance': range(1, 221),
        'name': "Stock Data",
        'ica_graph_recon': "Stock Data Reconstruction Graph of ICA",
        'n_components': 100,
        'filename': "stock_data_ica.txt",
        'ica_title': 'Stock Data Kurtosis of Calculated Components',
        'ica_graph': "Stock Data on Principle Components of ICA"
    }

    census_params = {
        'variance': range(1, 14),
        'name': "Census Data",
        'ica_graph_recon': "Census Data Reconstruction Graph of ICA",
        'n_components': 4,
        'filename': "census_data_ica.txt",
        'ica_title': 'Census Data Kurtosis of Calculated Components',
        'ica_graph': "Census Data on Principle Components of ICA"
    }

    # run_ica_reconstruction(stock_params, x_stock_data)
    # run_ica_reconstruction(census_params, x_census_data)
    run_ica(stock_params, x_stock_data, y_stock_data)
    run_ica(census_params, x_census_data, y_census_data)


if __name__ == '__main__':
    main()