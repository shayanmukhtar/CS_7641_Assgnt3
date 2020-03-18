import numpy as np
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from Code import process_data
from Code import principle_component_analysis
from Code import independant_component_analysis
from Code import svd_projection
from Code import random_projections
from Code import k_means
from Code import expectation_maximization
import matplotlib.pyplot as plt
from Code import evaluate_model_learning_complexity
import sys
np.set_printoptions(threshold=sys.maxsize)


def neural_network_learning(x_data, y_data, params):
    # split into training set and validation set
    train_size = 0.6
    random_state = 86
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=train_size, random_state=random_state)

    # create the grid search parameter dictionary
    paramaters = [
        {'solver': ['sgd'], 'hidden_layer_sizes': [(32, 32)]},
    ]

    grid_searcher = GridSearchCV(MLPClassifier(), paramaters)

    grid_searcher.fit(x_train, y_train)

    # form a 2d list of your data
    report = [["Parameters", "Mean Fit Time", "Std Dev Fit Time", "Split 0 Score", "Split 1 Score", "Split 2 Score", "Split 3 Score", "Split 4 Score"]]
    for row in range(0, len(grid_searcher.cv_results_['params'])):
        row_data = [str(grid_searcher.cv_results_['params'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['mean_fit_time'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['std_fit_time'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split0_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split1_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split2_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split3_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split4_test_score'][row]),
                    ]
        report.append(row_data)

    # print dictionary of scores
    print("Grid Search Report")
    print()

    col_width = max(len(word) for row in report for word in row) + 2  # padding

    with open(params['path'] + params['file'], 'w+') as fileOut:
        for row in report:
            fileOut.write("".join(word.ljust(col_width) + "\n" for word in row))

    # plot the learning curve
    title = params['title'] + " ANN - " + str(grid_searcher.best_params_)
    alpha_range = np.logspace(-6, -1, 5)
    figure = evaluate_model_learning_complexity.plot_learning_curve(grid_searcher.best_estimator_, title,
                                                                    x_train, y_train, parameter="alpha",
                                                                    param_grid=alpha_range, param_string="Alpha",
                                                                    log_range=True)

    figure.savefig(params['path'] + params['title'] + " ANN Learning Plots")

    # Plot the confusion matrix
    plt.figure()
    cm = plot_confusion_matrix(grid_searcher.best_estimator_, x_test, y_test, display_labels=['<=$50K', '>$50K'])
    cm.plot()
    plt.savefig(params['path'] + params['cm_mat'])

    with open(params['path'] + params['file'], 'a') as fileOut:
        fileOut.write("Scoring ANN with parameters: " + str(grid_searcher.best_params_) +
                      "\tOn Data: " + params['title'] + "\n")
        fileOut.write(str(grid_searcher.best_estimator_.score(x_test, y_test)))


def dimensionality_reduction_ann():
    x_census_data, y_census_data = process_data.process_census_data('./../Datasets/Census_Income')
    train_size = 0.6
    random_state = 86
    x_census_data_reduced = []

    census_params = {
        'n_components': 4,
        'filename': None,
        'projection_loss_graph': None,
        'num_retry': 10,
        'ica_title': None,
        'components': 4
    }
    # import all the reduced dimensionality datasets
    # PCA
    x_census_data_reduced.append(principle_component_analysis.run_pca(census_params, x_census_data))

    # ICA
    x_census_data_reduced.append(independant_component_analysis.run_ica(census_params, x_census_data))

    # Random Projections
    x_census_data_reduced.append(random_projections.create_random_guassian_projections(census_params, x_census_data))

    # SVD
    x_census_data_reduced.append(svd_projection.run_svd(census_params, x_census_data))

    census_params = []
    census_params.append({"title": "Census Dataset - PCA Dimension Reduction", "path": "./ANN_Reduced_Data/",
                         'file': 'pca_reduced_ann.txt',
                          'cm_mat': "Census Dataset - PCA Dimension Reduction Confusion Matrix"})
    census_params.append({"title": "Census Dataset - ICA Dimension Reduction", "path": "./ANN_Reduced_Data/",
                         'file': 'ica_reduced_ann.txt',
                          'cm_mat': "Census Dataset - ICA Dimension Reduction Confusion Matrix"})
    census_params.append({"title": "Census Dataset - Random Projection Reduction", "path": "./ANN_Reduced_Data/",
                         'file': 'random_reduced_ann.txt',
                          'cm_mat': "Census Dataset - Random Projection Reduction Confusion Matrix"})
    census_params.append({"title": "Census Dataset - SVD Dimension Reduction", "path": "./ANN_Reduced_Data/",
                         'file': 'svd_reduced_ann.txt',
                          'cm_mat': "Census Dataset - SVD Dimension Reduction Confusion Matrix"})

    # Run 4 neural nets - one for each instance of the reduced data
    for i in range(4):
        neural_network_learning(x_census_data_reduced[i], y_census_data, census_params[i])


def clusters_added_ann():
    x_census_data, y_census_data = process_data.process_census_data('./../Datasets/Census_Income')
    # 2 ANN's here, one for each clustering algorithm
    census_params_km = {'clusters': 8, "title": "Census Dataset - K Means Clusters Added as Feature",
                        "path": "./ANN_Clustered_Data/", "file": "k_means_ann.txt",
                        "cm_mat": "Census Dataset - K Means Clusters added Confusion Matrix"
    }
    census_params_em = {'clusters': 12, "title": "Census Dataset - EM Clusters Added as Feature",
                        "path": "./ANN_Clustered_Data/", "file": "em_ann.txt",
                        "cm_mat": "Census Dataset - EM Clusters added Confusion Matrix"
    }
    census_params_orig = {"title": "Census Dataset - Original",
                          "path": "./ANN_Clustered_Data/", "file": "ann_orig.txt",
                          "cm_mat": "Census Dataset - Original Confusion Matrix"
                        }

    km_clusters = k_means.return_k_means_clusters(census_params_km, x_census_data).reshape((len(x_census_data), 1))
    em_clusters = k_means.return_k_means_clusters(census_params_em, x_census_data).reshape((len(x_census_data), 1))

    x_census_data_km = np.hstack([x_census_data, km_clusters])
    x_census_data_em = np.hstack([x_census_data, em_clusters])

    neural_network_learning(x_census_data_km, y_census_data, census_params_km)
    neural_network_learning(x_census_data_em, y_census_data, census_params_em)
    neural_network_learning(x_census_data, y_census_data, census_params_orig)


def main():
    # dimensionality_reduction_ann()
    clusters_added_ann()


if __name__ == '__main__':
    main()