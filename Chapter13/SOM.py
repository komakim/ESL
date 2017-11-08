#https://qiita.com/shuhei_f/items/5c4ff6ed278eb1747a6f
# -*- coding: utf-8 -*-

import numpy as np


def linear_kernel(x1, x2):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    return np.dot(x1, x2.T)


# k(x, y) = exp(- gamma || x1 - x2 ||^2)
def get_rbf_kernel(gamma):
    def rbf_kernel(x1, x2):
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        m1, _ = x1.shape
        m2, _ = x2.shape
        norm1 = np.dot(np.ones([m2, 1]), np.atleast_2d(np.sum(x1 ** 2, axis=1))).T
        norm2 = np.dot(np.ones([m1, 1]), np.atleast_2d(np.sum(x2 ** 2, axis=1)))
        return np.exp(- gamma * (norm1 + norm2 - 2 * np.dot(x1, x2.T)))
    return rbf_kernel


# k(x1, x2) = (<x1, x2> + coef0)^degree
def get_polynomial_kernel(degree, coef0):
    def polynomial_kernel(x1, x2):
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        return (np.dot(x1, x2.T) + coef0)**degree
    return polynomial_kernel


class CSvc:

    def __init__(self, C=1e0, kernel=linear_kernel, tol=1e-3, max_iter=1000,
                 gamma=1e0, degree=3, coef0=0):
        self._EPS = 1e-5
        self._TAU = 1e-12
        self._cache = {}
        self.tol = tol
        self.max_iter = max_iter
        self.C = C
        self.gamma = gamma
        self.degree, self.coef0 = degree, coef0
        self.kernel = None
        self._alpha = None
        self.intercept_ = None
        self._grad = None
        self.itr = None
        self._ind_y_p, self._ind_y_n = None, None
        self._i_low, self._i_up = None, None
        self.set_kernel_function(kernel)

    def _init_solution(self, y):
        num = len(y)
        self._alpha = np.zeros(num)
        self._i_low = y < 0
        self._i_up = y > 0

    def set_kernel_function(self, kernel):
        if callable(kernel):
            self.kernel = kernel
        elif kernel == 'linear':
            self.kernel = linear_kernel
        elif kernel == 'rbf' or kernel == 'gaussian':
            self.kernel = get_rbf_kernel(self.gamma)
        elif kernel == 'polynomial' or kernel == 'poly':
            self.kernel = get_polynomial_kernel(self.degree, self.coef0)
        else:
            raise ValueError('{} is undefined name as kernel function'.format(kernel))

    def _select_working_set1(self, y):
        minus_y_times_grad = - y * self._grad
        # Convert boolean mask to index
        i_up = self._i_up.nonzero()[0]
        i_low = self._i_low.nonzero()[0]
        ind_ws1 = i_up[np.argmax(minus_y_times_grad[i_up])]
        ind_ws2 = i_low[np.argmin(minus_y_times_grad[i_low])]
        return ind_ws1, ind_ws2

    def fit(self, x, y):
        self._init_solution(y)
        self._cache = {}
        num, _ = x.shape
        # Initialize the dual coefficients and gradient
        self._grad = - np.ones(num)
        # Start the iterations of SMO algorithm
        for itr in xrange(self.max_iter):
            # Select two indices of variables as working set
            ind_ws1, ind_ws2 = self._select_working_set1(y)
            # Check stopping criteria: m(a_k) <= M(a_k) + tolerance
            m_lb = - y[ind_ws1] * self._grad[ind_ws1]
            m_ub = - y[ind_ws2] * self._grad[ind_ws2]
            kkt_violation = m_lb - m_ub
            # print 'KKT Violation:', kkt_violation
            if kkt_violation <= self.tol:
                print 'Converged!', 'Iter:', itr, 'KKT Violation:', kkt_violation
                break
            # Compute (or get from cache) two columns of gram matrix
            if ind_ws1 in self._cache:
                qi = self._cache[ind_ws1]
            else:
                qi = self.kernel(x, x[ind_ws1]).ravel() * y * y[ind_ws1]
                self._cache[ind_ws1] = qi
            if ind_ws2 in self._cache:
                qj = self._cache[ind_ws2]
            else:
                qj = self.kernel(x, x[ind_ws2]).ravel() * y * y[ind_ws2]
                self._cache[ind_ws2] = qj
            # Construct sub-problem
            qii, qjj, qij = qi[ind_ws1], qj[ind_ws2], qi[ind_ws2]
            # Solve sub-problem
            if y[ind_ws1] * y[ind_ws2] > 0:  # The case where y_i equals y_j
                v1, v2 = 1., -1.
                d_max = min(self.C - self._alpha[ind_ws1], self._alpha[ind_ws2])
                d_min = max(-self._alpha[ind_ws1], self._alpha[ind_ws2] - self.C)
            else:  # The case where y_i equals y_j
                v1, v2 = 1., 1.
                d_max = min(self.C - self._alpha[ind_ws1], self.C - self._alpha[ind_ws2])
                d_min = max(-self._alpha[ind_ws1], -self._alpha[ind_ws2])
            quad_coef = v1**2 * qii + v2**2 * qjj + 2 * v1 * v2 * qij
            quad_coef = max(quad_coef, self._TAU)
            d = - (self._grad[ind_ws1] * v1 + self._grad[ind_ws2] * v2) / quad_coef
            d = max(min(d, d_max), d_min)
            # Update dual coefficients
            self._alpha[ind_ws1] += d * v1
            self._alpha[ind_ws2] += d * v2
            # Update the gradient
            self._grad += d * v1 * qi + d * v2 * qj
            # Update I_up with respect to ind_ws1 and ind_ws2
            self._update_iup_and_ilow(y, ind_ws1)
            self._update_iup_and_ilow(y, ind_ws2)
        else:
            print 'Exceed maximum iteration'
            print 'KKT Violation:', kkt_violation
        # Set results after optimization procedure
        self._set_result(x, y)
        self.intercept_ = (m_lb + m_ub) / 2.
        self.itr = itr + 1

    def _update_iup_and_ilow(self, y, ind):
        # Update I_up with respect to ind
        if (y[ind] > 0) and (self._alpha[ind] / self.C <= 1 - self._EPS):
            self._i_up[ind] = True
        elif (y[ind] < 0) and (self._EPS <= self._alpha[ind] / self.C):
            self._i_up[ind] = True
        else:
            self._i_up[ind] = False
        # Update I_low with respect to ind
        if (y[ind] > 0) and (self._EPS <= self._alpha[ind] / self.C):
            self._i_low[ind] = True
        elif (y[ind] < 0) and (self._alpha[ind] / self.C <= 1 - self._EPS):
            self._i_low[ind] = True
        else:
            self._i_low[ind] = False

    def _set_result(self, x, y):
        self.support_ = np.where(self._EPS < (self._alpha / self.C))[0]
        self.support_vectors_ = x[self.support_]
        self.dual_coef_ = self._alpha[self.support_] * y[self.support_]
        # Compute w when using linear kernel
        if self.kernel == linear_kernel:
            self.coef_ = np.sum(self.dual_coef_ * x[self.support_].T, axis=1)

    def decision_function(self, x):
        return np.sum(self.kernel(x, self.support_vectors_) * self.dual_coef_, axis=1) + self.intercept_

    def predict(self, x):
        return np.sign(self.decision_function(x))

    def score(self, x, y):
        return sum(self.decision_function(x) * y > 0) / float(len(y))


if __name__ == '__main__':
    # Create toy problem
    np.random.seed(0)
    num_p = 15
    num_n = 15
    dim = 2
    x_p = np.random.multivariate_normal(np.ones(dim) * 1, np.eye(dim), num_p)
    x_n = np.random.multivariate_normal(np.ones(dim) * 2, np.eye(dim), num_n)
    x = np.vstack([x_p, x_n])
    y = np.array([1.] * num_p + [-1.] * num_n)

    # Set parameters
    max_iter = 500000
    C = 1e0
    gamma = 0.005
    tol = 1e-3

    # Set kernel function
    # kernel = get_rbf_kernel(gamma)
    # kernel = linear_kernel
    # kernel = 'rbf'
    # kernel = 'polynomial'
    kernel = 'linear'

    # Create object
    csvc = CSvc(C=C, kernel=kernel, max_iter=max_iter, tol=tol)

    # Run SMO algorithm
    csvc.fit(x, y)
