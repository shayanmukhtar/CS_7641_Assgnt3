from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import numpy as np
import matplotlib.pyplot as plt


# below code sourced from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, x_train, y_train, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), parameter=None, param_grid=None,
                        param_string="", log_range=False):
    # form a learning curve using the best estimator we have so far from the grid search - this is to evaluate
    # learning complexity of the model
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, x_train, y_train, train_sizes=train_sizes, return_times=True, n_jobs=n_jobs)

    # if there is another parameter, make 4 graphs, one with score against that param
    _, axes = plt.subplots(1, 3, figsize=(20, 5))

    # if the parameter is not none, generate a validation curve based on that parameter
    if parameter is not None:
        valid_scores_train, valid_scores_test = validation_curve(estimator, x_train, y_train,
                                                                 param_name=parameter, param_range=param_grid,
                                                                 n_jobs=-1)

        valid_scores_test_mean = np.mean(valid_scores_test, axis=1)
        valid_scores_test_std = np.std(valid_scores_test, axis=1)
        valid_scores_train_mean = np.mean(valid_scores_train, axis=1)
        valid_scores_train_std = np.std(valid_scores_train, axis=1)

    axes[0].set_title(title)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1,
                            color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training Score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                    label="Cross-Validation Score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                            fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training Examples")
    axes[1].set_ylabel("Fit Times")
    axes[1].set_title("Scalability of the Model")

    # Plot fit_time vs score
    # axes[1][0].grid()
    # axes[1][0].plot(fit_times_mean, test_scores_mean, 'o-')
    # axes[1][0].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                            # test_scores_mean + test_scores_std, alpha=0.1)
    # axes[1][0].set_xlabel("Fit Times")
    # axes[1][0].set_ylabel("Score")
    # axes[1][0].set_title("Performance of the Model")

    if parameter is not None:
        axes[2].grid()
        axes[2].set_xlabel(param_string)
        axes[2].set_ylabel("Score")
        axes[2].fill_between(len(param_grid), valid_scores_train_mean - valid_scores_train_std,
                                valid_scores_train_mean + valid_scores_train_std, alpha=0.1,
                                color="r")
        axes[2].fill_between(len(param_grid), valid_scores_test_mean - valid_scores_test_std,
                                valid_scores_test_mean + valid_scores_test_std, alpha=0.1,
                                color="g")

        if log_range is False:
            axes[2].plot(param_grid, valid_scores_train_mean, 'o-', color="r",
                            label="Training Score")
            axes[2].plot(param_grid, valid_scores_test_mean, 'o-', color="g",
                            label="Cross-Validation Score")
        else:
            axes[2].semilogx(param_grid, valid_scores_train_mean, 'o-', color="r",
                                label="Training Score")
            axes[2].semilogx(param_grid, valid_scores_test_mean, 'o-', color="g",
                                label="Cross-Validation Score")

        axes[2].legend(loc="best")
        axes[2].set_title("Model Score versus " + param_string)

    return plt


