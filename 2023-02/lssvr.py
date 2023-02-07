import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.utils import check_X_y, check_array
from sklearn.exceptions import NotFittedError
from scipy.sparse import linalg
from sklearn.metrics import r2_score


class LSSVR(BaseEstimator, RegressorMixin):
    def __init__(self, C=1.0, kernel='rbf', gamma=None):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel

    # 拟合求解出参数
    def fit(self, X, y, support=None):
        X, y = check_X_y(X, y, multi_output=True, dtype='float')  ## 强制X为2D矩阵
        if not support:
            self.support_ = np.ones(X.shape[0], dtype=bool)
        else:
            self.support_ = check_array(support, ensure_2d=False, dtype="bool")

        self.support_vectors_ = X[self.support_, :]
        support_labels = y[self.support_]

        self.K_ = self.kernel_func(X, self.support_vectors_)
        w = self.K_.copy()  # 拷贝一下

        # 创建矩阵
        np.fill_diagonal(w, w.diagonal() + self.support_ / self.C)  # 对角阵上加上一项
        D = np.empty(np.array(w.shape) + 1)
        D[1:, 1:] = w
        D[0, 0] = 0
        D[0, 1:] = 1
        D[1:, 0] = 1

        # 创建y的矩阵
        shape = np.array(support_labels.shape)
        shape[0] = shape[0] + 1
        p = np.empty(shape)
        p[0] = 0  # 因为所有系数之和为0
        p[1:] = support_labels  # 其余都为y

        # 解方程(加上异常处理)(D * z = p)=>(z=p*D^-1)
        try:
            z = linalg.lsmr(D.T, p)[0]
        except:
            z = np.linalg.pinv(D).T @ p

        self.bias_ = z[0]
        self.alpha_ = z[1:]
        self.alpha_ = self.alpha_[self.support_]
        return self

    """
    f(x)=\sum_{i=1}^{N}\alpha_i*K(x_i,x)+b
    """

    def predict(self, X):
        if not hasattr(self, 'support_vectors_'):
            raise NotFittedError

        X = check_array(X, ensure_2d=False)
        K = self.kernel_func(X, self.support_vectors_)
        return K @ self.alpha_ + self.bias_

    def kernel_func(self, a, b):  # 默认“rbf”还有linear poly lap 3个核
        if self.kernel == 'linear':
            return np.dot(a, b.T)
        elif self.kernel == 'rbf':
            return rbf_kernel(a, b, gamma=self.gamma)
        elif self.kernel == "poly":
            return polynomial_kernel(a, b)
        elif self.kernel == "lap":
            return laplacian_kernel(a, b)
        elif callable(self.kernel):
            if hasattr(self.kernel, 'gamma'):
                return self.kernel(a, b, gamma=self.gamma)
            else:
                return self.kernel(a, b)
        else:
            return rbf_kernel(a, b, gamma=self.gamma)

    # 按照sklearn里面的模式,评价标准为r方
    def score(self, X, y):
        score = r2_score(y, self.predict(X))
        return score

    def __repr__(self):
        return "LSSVR()"