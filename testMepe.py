import GPy
import numpy as np
from matplotlib import pyplot as plt

from mepe import mepe_sampling, noiseless_gp


def simple_1d_func(x):
    return 3.0 * (1.0 - x) ** 2 * np.exp(-x ** 2 - 1) - 10 * (x / 5.0 - x ** 3) * np.exp(-x ** 2)


def plot_gp_1d(ax, gp: GPy.models.GPRegression, all_points, tru_func=None):
    if tru_func is not None:
        ax.plot(all_points, tru_func(all_points), color='r', label=r'Ground Truth')

    ax.plot(gp.X, gp.Y, 'kx', markersize=5, label='Observations')
    y_pred, y_vars = gp.predict_noiseless(all_points)
    y_sig = np.sqrt(y_vars).reshape(-1)
    ax.plot(all_points, y_pred, 'b-', label='Prediction')

    ax.fill_between(all_points.reshape(-1), y_pred.reshape(-1) - 1.96 * y_sig, y_pred.reshape(-1) + 1.96 * y_sig,
                    alpha=0.2, fc='b', ec='None', label='95% confidence interval')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.legend(loc='best')


def run():
    # Generate original points by a space filling algorithm
    domain = (-4, 1)
    initial_points = np.atleast_2d(np.linspace(domain[0], domain[1], num=3)).T
    candidate_points = np.atleast_2d(np.linspace(domain[0], domain[1], 100)).T
    budget = 7
    gp, X = mepe_sampling(budget, initial_points, candidate_points, simple_1d_func, False)

    fig, (ax_0, ax_1) = plt.subplots(nrows=1, ncols=2)

    # Testing stage: Take 1000 test points in the domain, compute their true error, compute their predictions, then do average RMSE
    test_X = np.atleast_2d(np.linspace(domain[0], domain[1], 1000)).T
    test_ys_tru = np.array([simple_1d_func(x) for x in test_X])
    test_ys_mepe_pred, test_ys_mepe_vars = gp.predict_noiseless(test_X)

    mepe_rmse = np.sqrt(np.average((test_ys_tru - test_ys_mepe_pred) ** 2))

    test_ys_mepe_std = np.sqrt(test_ys_mepe_vars)

    plot_gp_1d(ax_0, gp, test_X, simple_1d_func)
    ax_0.set_title("MEPE Final")

    # Sanity check: Test against random selection?
    rand_picks = np.random.choice(len(candidate_points), budget, replace=False)
    random_X = np.concatenate([initial_points, candidate_points[rand_picks]])
    random_ys_tru = np.array([simple_1d_func(x) for x in random_X])
    rand_gp = noiseless_gp(random_X, random_ys_tru)
    rand_gp.optimize_restarts(verbose=False, parallel=True)
    test_ys_random_pred, test_ys_random_var = rand_gp.predict_noiseless(test_X)
    test_ys_random_std = np.sqrt(test_ys_random_var)

    # rand_gp.plot()
    plot_gp_1d(ax_1, rand_gp, test_X, simple_1d_func)
    ax_1.set_title("Random")
    plt.show()

    random_rmse = np.sqrt(np.average((test_ys_tru - test_ys_random_pred) ** 2))

    print("MEPE Error: ", mepe_rmse)
    print("Random Error: ", random_rmse)


if __name__ == "__main__":
    run()
