import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import gaussian_random_matrix, GaussianRandomProjection
from sklearn import random_projection
from Code import process_data


def determine_min_dim(params, x_data):
    eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    min_dim = johnson_lindenstrauss_min_dim(x_data.shape[0], eps)

    plt.figure()
    plt.plot(eps, min_dim)
    plt.ylabel("Minimum Number of Dimensions")
    plt.xlabel("Distortion EPS")
    plt.title(params['min_dim_graph'])
    plt.savefig(params['min_dim_graph'] + ".png")


def create_random_guassian_projections(params, x_data):
    components = params['components']
    grps = [GaussianRandomProjection(n_components=components) for _ in range(params['num_retry'])]
    x_data_news = []
    x_data_recons = []
    x_data_projection_losses = []
    for i in range(0, params['num_retry']):
        print(str(i))
        # project data from high dim to low dim
        x_data_news.append(grps[i].fit_transform(x_data))

        # now reconstruct the data by projecting it back into higher dimensions
        x_data_recons.append(np.dot(x_data_news[i], grps[i].components_))

        # calculate projection errors
        x_projection_loss = ((x_data - x_data_recons) ** 2).mean()
        x_data_projection_losses.append(x_projection_loss)

    if params['projection_loss_graph'] is not None:
        plt.figure()
        plt.plot(x_data_projection_losses)
        plt.ylabel("Mean Squared Error")
        plt.xlabel("Random Model")
        plt.title(params['projection_loss_graph'])
        plt.savefig(params['projection_loss_graph'] + '.png')

    # TODO return the best three datasets
    return x_data_news[0]


def main():
    np.random.seed(311)
    # import the datasets
    x_census_data, y_census_data = process_data.process_census_data('./../Datasets/Census_Income')
    x_stock_data, y_stock_data = process_data.process_stock_data('./../Datasets/Stocks', ["2016", "2017"])

    stock_params = {
        'min_dim_graph': "Stock Dataset - Minimum Dimension vs EPS",
        'projection_loss_graph': "Stock Dataset - Projection Losses for Data Random Projections",
        'components': 100,
        'num_retry': 10
    }

    census_params = {
        'min_dim_graph': "Census Dataset - Minimum Dimension vs EPS",
        'projection_loss_graph': "Census Dataset - Projection Losses for Data Random Projections",
        'components': 4,
        'num_retry': 10
    }

    # determine_min_dim(stock_params, x_stock_data)
    # determine_min_dim(census_params, x_census_data)
    create_random_guassian_projections(stock_params, x_stock_data)
    create_random_guassian_projections(census_params, x_census_data)


if __name__ == '__main__':
    main()