import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def bayes_est(X_tilda, X, t, y, kernel_func_theta, kernel_func_f, s0, s1):
    (n, d) = X.shape
    (m, d) = X_tilda.shape

    # z is the Gram matrix for f
    Phi_z = kernel_func_f(X)
    # w is the Gram matrix for theta
    Phi_w_nn = kernel_func_theta(X)
    Phi_w_mn = kernel_func_theta(X_tilda, X)
    Phi_w_mm = kernel_func_theta(X_tilda)
    T = np.diag(t)
    # Precision of the noise (inverse of variance)
    S = s0 * T + s1 * (np.eye(n) - T)

    # Combine Gram matrices for theta
    Phi_w_all = np.zeros((m + n, m + n))
    Phi_w_all[0:n, 0:n] = Phi_w_nn
    Phi_w_all[0:n, n:(m + n)] = Phi_w_mn.T
    Phi_w_all[n:(m + n), 0:n] = Phi_w_mn
    Phi_w_all[n:(m + n), n:(m + n)] = Phi_w_mm

    # Combine Gram matrices for theta and f
    E_inv = np.zeros((m + 2 * n, m + 2 * n))
    E_inv[0:(m + n), 0:(m + n)] = Phi_w_all
    E_inv[(m + n):(m + 2 * n), (m + n):(m + 2 * n)] = Phi_z

    F = np.zeros((m + 2 * n, m + 2 * n))
    F[0:n, 0:n] = S.dot(T.dot(T))
    F[0:n, (m + n):(m + 2 * n)] = S.dot(T)
    F[(m + n):(m + 2 * n), 0:n] = S.dot(T)
    F[(m + n):(m + 2 * n), (m + n):(m + 2 * n)] = S

    # Use Schur complement to compute the inverse matrix
    A_inv = E_inv - E_inv.dot(F.dot(np.linalg.inv(np.eye(m + 2 * n) + E_inv.dot(F))).dot(E_inv))
    B = np.zeros((m + 2 * n, n))
    B[0:n, 0:n] = -S.dot(T)
    B[(m + n):(m + 2 * n), 0:n] = -S
    C = B.T
    D = S

    Sigma = D - C.dot(A_inv.dot(B))
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except:
        Sigma_inv = np.linalg.pinv(Sigma)

    Sigma_theta_theta = A_inv + A_inv.dot(B.dot(Sigma_inv.dot(C.dot(A_inv))))
    Sigma_theta_y = -A_inv.dot(B.dot(Sigma_inv))
    Sigma_y_theta = Sigma_theta_y.T
    Sigma_y_y = Sigma_inv

    Sigma_y_y_inv = Sigma

    M = Sigma_theta_y.dot(Sigma_y_y_inv)
    mu_theta_y = M.dot(y)

    post_Sigma_theta = Sigma_theta_theta - Sigma_theta_y.dot(Sigma_y_y_inv.dot(Sigma_y_theta))

    return mu_theta_y[n:(n + m)], post_Sigma_theta[n:(n + m), n:(n + m)]

def create_kernel(kernel_class, params, prefix):
    # Instantiate the kernel with the given parameters
    if kernel_class == RBF:
        length_scale = params[f'{prefix}_length_scale']
        return RBF(length_scale=length_scale)
    elif kernel_class == Matern:
        length_scale = params[f'{prefix}_length_scale']
        nu = params[f'{prefix}_nu']
        return Matern(length_scale=length_scale, nu=nu)
    elif kernel_class == RationalQuadratic:
        length_scale = params[f'{prefix}_length_scale']
        alpha = params[f'{prefix}_alpha']
        return RationalQuadratic(length_scale=length_scale, alpha=alpha)
    else:
        raise ValueError(f"Unsupported kernel class: {kernel_class}")

def get_kernel_hyperparameters(kernel_class, prefix):
    # Return the hyperparameter names and bounds for the given kernel
    hyperparameters = []
    if kernel_class == RBF:
        hyperparameters.append({'name': f'{prefix}_length_scale', 'type': 'continuous', 'domain': (1e-2, 100.0)})
    elif kernel_class == Matern:
        hyperparameters.append({'name': f'{prefix}_length_scale', 'type': 'continuous', 'domain': (1e-2, 100.0)})
        # For nu, we can fix it to common values due to its discrete nature
        hyperparameters.append({'name': f'{prefix}_nu', 'type': 'categorical', 'domain': (1.5, 2.5)})
    elif kernel_class == RationalQuadratic:
        hyperparameters.append({'name': f'{prefix}_length_scale', 'type': 'continuous', 'domain': (1e-2, 100.0)})
        hyperparameters.append({'name': f'{prefix}_alpha', 'type': 'continuous', 'domain': (1e-5, 10.0)})
    else:
        raise ValueError(f"Unsupported kernel class: {kernel_class}")
    return hyperparameters

def marginal_likelihood_wrapper(params_array, param_names, X, t, y, kernel_theta_class, kernel_f_class, fixed_params):
    # Convert params_array to a dictionary
    params = dict(zip(param_names, params_array))
    params.update(fixed_params)  # Add fixed parameters like 'nu' if any

    s0 = params['s0']
    s1 = params['s1']

    # Create kernels
    try:
        kernel_theta = create_kernel(kernel_theta_class, params, prefix='theta')
        kernel_f = create_kernel(kernel_f_class, params, prefix='f')
    except Exception as e:
        # If kernel creation fails (e.g., due to invalid hyperparameters), return a high cost
        return np.inf

    # Compute Gram matrices
    try:
        Phi_w = kernel_theta(X)
        Phi_z = kernel_f(X)
        T = np.diag(t)
        S = np.diag(np.where(t == 1, 1.0 / s1, 1.0 / s0))

        cov = S + T.dot(Phi_w.dot(T)) + Phi_z
        nll = -mvn.logpdf(y, cov=cov)  # Negative log-likelihood for minimization
        if np.isinf(nll) or np.isnan(nll):
            return np.inf
        return nll
    except Exception as e:
        return np.inf

def optimize_hyperparameters_scipy(X, t, y, kernel_options, plot_progress=True):
    '''
    Optimize s0, s1, and kernel hyperparameters using scipy.optimize.minimize
    over different kernel choices.
    '''
    best_neg_log_likelihood = np.inf
    best_params = None
    best_kernel_theta = None
    best_kernel_f = None

    # For Matern kernel, possible nu values
    matern_nu_values = [0.5, 1.5, 2.5]

    for kernel_theta_class in kernel_options:
        for kernel_f_class in kernel_options:
            # Prepare parameter names and bounds
            param_list = [
                {'name': 's0', 'type': 'continuous', 'domain': (0.01, 500)},
                {'name': 's1', 'type': 'continuous', 'domain': (0.01, 500)},
            ]
            param_names = ['s0', 's1']
            bounds = [(0.1, 500), (0.1, 500)]
            x0 = [1.0/np.var(y), 1.0/np.var(y)]  # Initial guesses

            # Get hyperparameters for theta kernel
            theta_hyperparams = get_kernel_hyperparameters(kernel_theta_class, prefix='theta')
            for param in theta_hyperparams:
                if param['type'] == 'continuous':
                    param_list.append(param)
                    param_names.append(param['name'])
                    bounds.append(param['domain'])
                    x0.append(np.mean(param['domain']))
            # Handle discrete parameters separately (fixed parameters)
            fixed_params_theta = {}
            for param in theta_hyperparams:
                if param['type'] == 'categorical':
                    fixed_params_theta[param['name']] = param['domain']

            # Get hyperparameters for f kernel
            f_hyperparams = get_kernel_hyperparameters(kernel_f_class, prefix='f')
            for param in f_hyperparams:
                if param['type'] == 'continuous':
                    param_list.append(param)
                    param_names.append(param['name'])
                    bounds.append(param['domain'])
                    x0.append(np.mean(param['domain']))
            # Handle discrete parameters separately (fixed parameters)
            fixed_params_f = {}
            for param in f_hyperparams:
                if param['type'] == 'categorical':
                    fixed_params_f[param['name']] = param['domain']

            # Combine fixed parameters
            fixed_params_options = []
            # Generate combinations of fixed parameters
            from itertools import product
            theta_nu_values = fixed_params_theta.get('theta_nu', [None])
            f_nu_values = fixed_params_f.get('f_nu', [None])
            theta_nu_values = theta_nu_values if isinstance(theta_nu_values, (list, tuple)) else [theta_nu_values]
            f_nu_values = f_nu_values if isinstance(f_nu_values, (list, tuple)) else [f_nu_values]
            fixed_params_combinations = product(theta_nu_values, f_nu_values)

            for theta_nu, f_nu in fixed_params_combinations:
                fixed_params = {}
                if theta_nu is not None:
                    fixed_params['theta_nu'] = theta_nu
                if f_nu is not None:
                    fixed_params['f_nu'] = f_nu

                # Define the objective function
                def objective(params_array):
                    return marginal_likelihood_wrapper(params_array, param_names, X, t, y, kernel_theta_class, kernel_f_class, fixed_params)

                # Run optimization
                res = minimize(
                    objective,
                    x0=x0,
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'disp': False, 'maxiter': 100}
                )

                current_neg_log_likelihood = res.fun
                if current_neg_log_likelihood < best_neg_log_likelihood:
                    best_neg_log_likelihood = current_neg_log_likelihood
                    best_params = res.x
                    best_param_names = param_names.copy()
                    best_fixed_params = fixed_params.copy()
                    best_kernel_theta = kernel_theta_class
                    best_kernel_f = kernel_f_class

    # Convert best_params to a dictionary
    best_params_dict = dict(zip(best_param_names, best_params))
    best_params_dict.update(best_fixed_params)

    s0_opt = best_params_dict['s0']
    s1_opt = best_params_dict['s1']

    print(f"\nBest Kernel Theta: {best_kernel_theta.__name__}, Best Kernel f: {best_kernel_f.__name__}")
    print(f"Optimized Hyperparameters:")
    print(f" s0={s0_opt:.5f}, s1={s1_opt:.5f}")

    # Print optimized hyperparameters for theta kernel
    print(" Theta Kernel Hyperparameters:")
    for name in best_param_names:
        if name.startswith('theta_') and name not in ['s0', 's1']:
            print(f"  {name} = {best_params_dict[name]}")
    if 'theta_nu' in best_params_dict:
        print(f"  theta_nu = {best_params_dict['theta_nu']}")

    # Print optimized hyperparameters for f kernel
    print(" f Kernel Hyperparameters:")
    for name in best_param_names:
        if name.startswith('f_') and name not in ['s0', 's1']:
            print(f"  {name} = {best_params_dict[name]}")
    if 'f_nu' in best_params_dict:
        print(f"  f_nu = {best_params_dict['f_nu']}")

    print(f"\nMarginal Likelihood Value: {-best_neg_log_likelihood:.5f}")

    return s0_opt, s1_opt, best_params_dict, best_kernel_theta, best_kernel_f, -best_neg_log_likelihood

def cate_estimater(X, y, t, X_new):
    kernel_options = [RBF, Matern]

    s0, s1, best_params_dict, kernel_theta_class, kernel_f_class, marginal_likelihood = optimize_hyperparameters_scipy(
        X, t, y, kernel_options
    )

    # Create the best kernels with optimized hyperparameters
    kernel_func_theta = create_kernel(kernel_theta_class, best_params_dict, prefix='theta')
    kernel_func_f = create_kernel(kernel_f_class, best_params_dict, prefix='f')

    post, predictive_post = bayes_est(
        X_new, X, t, y, kernel_func_theta, kernel_func_f, s0, s1
    )
    return post, predictive_post, marginal_likelihood
