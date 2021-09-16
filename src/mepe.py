import time
from typing import List, Callable

import GPy
import numpy as np
from matplotlib import pyplot as plt


def index_of_closest(ref_point, points: List) -> int:
    return np.argmin([np.linalg.norm(p - ref_point) for p in points])


def cross_validation_error(i: int, X: np.ndarray, y_trus: np.ndarray) -> float:
    assert len(X) == len(y_trus)
    X_sub_i = X[np.arange(len(X)) != i]
    y_trus_sub_i = y_trus[np.arange(len(y_trus)) != i]
    gp_sub_i = noiseless_gp(X_sub_i, y_trus_sub_i)
    gp_sub_i.optimize()
    y_pred, y_vars = gp_sub_i.predict_noiseless(np.array([X[i]]))
    err_cv = (y_trus[i].reshape(-1) - y_pred.reshape(-1)) ** 2
    assert len(err_cv) == 1
    return err_cv[0]


def cross_validation_error_fast(gp: GPy.models.GPRegression) -> List[float]:
    R = gp.kern.K(gp.X) / gp.kern.variance
    R_inv = np.linalg.inv(R)
    one_vec = np.ones((len(gp.X), 1))

    inv_11 = np.linalg.inv(one_vec.T @ one_vec)
    H = one_vec @ inv_11 @ one_vec.T

    R_inv_y = R_inv @ gp.Y
    inv_1r1 = np.linalg.inv(one_vec.T @ R_inv @ one_vec)
    mu_est = inv_1r1 @ one_vec.T @ R_inv_y
    d = gp.Y - one_vec @ mu_est

    rhs_mat = d.reshape(-1) + one_vec.reshape(-1) + H * d / (1.0 - H.diagonal())
    cv_err_fast = (np.sum(R_inv * rhs_mat, axis=1) / R_inv.diagonal()) ** 2.0

    return cv_err_fast


def noiseless_gp(xs, ys) -> GPy.models.GPRegression:
    kernel = GPy.kern.Matern52(input_dim=xs.shape[-1])

    gp = GPy.models.GPRegression(xs, ys, kernel, normalizer=True)
    gp.Gaussian_noise.variance = 0.0
    gp.Gaussian_noise.variance.fix()
    return gp


def noisy_gp(xs, ys) -> GPy.models.GPRegression:
    kernel = GPy.kern.Matern52(1)
    gp = GPy.models.GPRegression(xs, ys, kernel)
    return gp


def mepe_sampling(budget: int, initial_points: np.ndarray, candidate_points: np.ndarray, sim_func: Callable) -> GPy.models.GPRegression:
    assert initial_points.ndim >= 2
    assert candidate_points.ndim == initial_points.ndim
    assert budget > 0

    X = initial_points
    y_trus = np.array([sim_func(x) for x in X]).reshape(-1, 1)

    # Instantiate a Gaussian Process model
    gp = noiseless_gp(X, y_trus)
    gp.optimize_restarts(verbose=False, parallel=True)

    # While the stopping criterion is not me
    balance_factors = []
    balance_factor = 0.5
    for q in range(budget):
        print("\tIteration: ", q, end='\t')
        start_time = time.time()
        y_preds, cp_vars = gp.predict_noiseless(candidate_points)

        # X_err_cvs = [cross_validation_error(i, X, y_trus) for i in range(len(X))]
        X_err_cvs = cross_validation_error_fast(gp)

        # Calculate the CV error at each observed point using Eq (17)
        cp_cv_errs = np.array([X_err_cvs[index_of_closest(cp, X)] for cp in candidate_points])

        # Form EPE criterion in (23)
        expected_prediction_errs = balance_factor * cp_cv_errs + (1.0 - balance_factor) * cp_vars.reshape(-1)

        found_count = 0
        for i, cp in enumerate(candidate_points):
            if np.any(np.all(X == cp, axis=1)):
                expected_prediction_errs[i] = 0.0
                found_count += 1
        assert found_count == len(X)

        # Obtain new point by solving (25)
        best_points = sorted(zip(expected_prediction_errs, range(len(candidate_points))), reverse=True)
        best_point_i = best_points[0][1]
        new_point = candidate_points[best_point_i]
        if np.any((np.all(X == new_point, axis=1))):
            raise ValueError("Should not choose previously observed point as best candidate!")

        # Update information
        X = np.append(X, [new_point], axis=0)
        new_y_tru = sim_func(new_point)
        y_trus = np.append(y_trus, np.array([new_y_tru]).reshape(-1, 1), axis=0)

        new_y_predicted = gp.predict_noiseless(np.array([new_point]))[0][0][0]
        new_err_tru = (new_y_tru[0] - new_y_predicted) ** 2
        new_err_cv = cp_cv_errs[best_point_i]

        # Refit GP
        gp = noiseless_gp(X, y_trus)
        gp.optimize_restarts(verbose=False, parallel=True)

        sanity_epe = balance_factor * new_err_cv + (1.0 - balance_factor) * cp_vars[best_point_i][0]
        prev_epe = expected_prediction_errs[best_point_i]
        assert sanity_epe == prev_epe
        assert 0.0 < balance_factor < 1.0

        balance_factor = 0.99 * np.minimum(1.0, 0.5 * new_err_tru / new_err_cv)
        balance_factors.append(balance_factor)

        print("Time: ", time.time() - start_time)

    print(balance_factors)

    gp = noiseless_gp(X, y_trus)
    gp.optimize_restarts(verbose=False, parallel=True)

    return gp