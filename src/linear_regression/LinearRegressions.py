import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t, f
from typing import Union
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import percentileofscore

class LinearRegressionSM:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self._model = None

    def fit(self):
        alfa = sm.add_constant(self.right_hand_side)
        beta_0 = self.left_hand_side
        self._model = sm.OLS(beta_0, alfa).fit()

    def get_params(self):
        return self._model.params.rename('Beta coefficients')

    def get_pvalues(self):
        return self._model.pvalues.rename('P-values for the corresponding coefficients')

    def get_wald_test_result(self, restrictions):
        wald_test = self._model.wald_test(restrictions)
        fvalue = float(wald_test.statistic)
        pvalue = float(wald_test.pvalue)
        return f'F-value: {fvalue:.3}, p-value: {pvalue:.3}'

    def get_model_goodness_values(self):
        ars = self._model.rsquared_adj
        ak = self._model.aic
        by = self._model.bic
        return f'Adjusted R-squared: {ars:.3}, Akaike IC: {ak:.3}, Bayes IC: {by:.3}'


class LinearRegressionNP:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side

    def fit(self):
        self.right_hand_side = pd.concat([pd.Series(1, index=self.right_hand_side.index, name='Constant'),
                                         self.right_hand_side], axis=1)
        self.X = self.right_hand_side.to_numpy()
        self.y = self.left_hand_side.to_numpy()
        self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        self.beta_coefficients = pd.Series(self.beta, index=self.right_hand_side.columns, name='Beta coefficients')

    def get_params(self):
        return self.beta_coefficients

    def get_pvalues(self):
        n, K = self.right_hand_side.shape[0], self.right_hand_side.shape[1]
        SSE = ((self.y - self.X @ self.beta) ** 2).sum() / (n - K)
        XTX_inv = np.linalg.inv(self.X.T @ self.X)
        t_stats = self.beta / np.sqrt(SSE * np.diag(XTX_inv))
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - K))
        return pd.Series(p_values, index=self.right_hand_side.columns,
                         name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, R):
        RES = self.y - self.X @ self.beta
        R_M = np.array(R)
        R = R_M @ self.beta
        n = len(self.left_hand_side)
        M, K = R_M.shape
        Sigma2 = np.sum(RES ** 2) / (n-K)
        H = R_M @ np.linalg.inv(self.X.T @ self.X) @ R_M.T
        wald_value = (R.T @ np.linalg.inv(H) @ R) / (M*Sigma2)
        p_value = 1 - stats.f.cdf(wald_value, dfn=M, dfd=n-K)
        return f"Wald: {wald_value:.3f}, p-value: {p_value:.3f}"

    def get_model_goodness_values(self) -> str:
        n, K = self.right_hand_side.shape[0], self.right_hand_side.shape[1]
        SSE = ((self.y - self.X @ self.beta) ** 2).sum()
        SST = ((self.y - self.y.mean()) ** 2).sum()
        SSR = SST - SSE
        crs = SSR / SST
        ars = 1 - (1 - crs) * ((n - 1) / (n - K))
        return f"Centered R-squared: {crs:.3f}, Adjusted R-squared: {ars:.3f}"

    def get_paired_se_and_percentile_ci(self, number_of_bootstrap_samples, alpha, random_seed):
        np.random.seed(random_seed)
        bootstrap_samples = []
        n = len(self.left_hand_side)
        for _ in range(number_of_bootstrap_samples):
            indices = np.random.choice(n, n, replace=True)
            X_bootstrap = self.X[indices,:]
            y_bootstrap = self.y[indices]
            beta_bootstrap = np.linalg.inv(X_bootstrap.T @ X_bootstrap) @ X_bootstrap.T @ y_bootstrap
            bootstrap_samples.append(beta_bootstrap[1])
        se_bootstrap = np.std(bootstrap_samples)
        ci_lower, ci_upper = np.percentile(bootstrap_samples, [alpha / 2 * 100, (1 - alpha / 2) * 100])
        return f"Paired Bootstraped SE: {se_bootstrap:.3f}, CI: [{ci_lower:.3f}, {ci_upper:.3f}]"

    def get_wild_se_and_normal_ci(self, number_of_bootstrap_samples, alpha, random_seed):
        np.random.seed(random_seed)
        bootstrap_samples = []
        n, k = self.X.shape
        residuals = self.y - self.X @ self.beta
        for _ in range(number_of_bootstrap_samples):
            v = np.random.normal(0,1,size=n)
            indices = np.random.choice (n,size=n, replace=True)
            X_bootstrap = self.X[indices, :]
            residuals_bootstrap = (self.y[indices] - X_bootstrap @ self.beta) * np.random.normal(size=n)
            y_bootstrap = X_bootstrap @ self.beta + v*residuals_bootstrap
            beta_bootstrap = np.linalg.inv(X_bootstrap.T @ X_bootstrap) @ X_bootstrap.T @ y_bootstrap
            bootstrap_samples.append(beta_bootstrap[1])
            se_bootstrap = np.std(bootstrap_samples)
            z = norm.ppf(1 - alpha / 2)
            ci_lower =self.beta[1] - z*se_bootstrap
            ci_upper =self.beta[1] + z*se_bootstrap
        return f"Wild Bootstraped SE: {se_bootstrap:.3f}, CI: [{ci_lower:.3f}, {ci_upper:.3f}]"

class LinearRegressionGLS:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.right_hand_side = pd.concat([pd.Series(1, index=self.right_hand_side.index, name='Constant'),self.right_hand_side], axis=1)
        self.X = self.right_hand_side.to_numpy()
        self.y = self.left_hand_side.to_numpy()

    def fit(self):
        ols_beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        residuals = self.y - self.X @ ols_beta
        squared_residuals = np.square(residuals)
        new_lhs = np.log(squared_residuals)
        new_X = self.right_hand_side.to_numpy()
        new_betas = np.linalg.inv(new_X.T @ new_X) @ new_X.T @ new_lhs
        predicted_values = np.exp(new_X @ new_betas)
        self.v_inverse = np.diag(1 / np.sqrt(predicted_values))
        self.cov_matrix = np.linalg.inv(new_X.T @ self.v_inverse @ new_X)
        self.beta = np.linalg.inv(self.X.T @ self.v_inverse @ self.X) @ self.X.T @ self.v_inverse @ self.y
        self.beta_coefficients = pd.Series(self.beta, index=self.right_hand_side.columns, name='Beta coefficients')
        self.residuals_gls = self.y - self.X @ self.beta

    def get_params(self):
        return pd.Series(self.beta_coefficients, name='Beta coefficients')

    def get_pvalues(self):
        self.df = len(self.y) - self.X.shape[1]
        self.residuals_var = (self.residuals_gls @ self.residuals_gls)/ self.df
        t_statistics = self.beta / np.sqrt(np.diag(self.residuals_var*np.linalg.inv(self.X.T @ self.v_inverse @ self.X)))
        p_values = [min(value, 1 - value) * 2 for value in t.cdf(-np.abs(t_statistics), df=self.df)]
        p_values = pd.Series(p_values, name='P-values for the corresponding coefficients')
        return p_values

    def get_wald_test_result(self, R):
        r_matrix= np.array(R)
        r = r_matrix @ self.beta
        m, k = r_matrix.shape
        self.n = len(self.y)
        wald_value = (r.T @ np.linalg.inv(r_matrix @ np.linalg.inv(self.X.T @ self.v_inverse @ self.X) @ r_matrix.T) @ r) / (m*self.residuals_var)
        p_value = 1 - f.cdf(wald_value,dfn=m,dfd=self.n-k)
        return f'Wald: {wald_value:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        n1=self.n-1
        tss = self.y.T @ self.v_inverse @ self.y
        rss = self.y.T @ self.v_inverse @ self.X @ np.linalg.inv(self.X.T @ self.v_inverse @ self.X) @ self.X.T @ self.v_inverse @ self.y
        crs = 1 - (rss / tss)
        ars = 1 - (rss / self.df * n1)/ tss
        return f'Centered R-squared: {crs:.3f}, Adjusted R-squared: {ars:.3f}'


class LinearRegressionML:
    def __init__(self,left_hand_side:pd.DataFrame,right_hand_side:pd.DataFrame):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side

    def llh(self,params):
        self.X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        self.y = self.left_hand_side
        beta0,beta1,beta2,beta3,self.sigma = params
        beta_params = np.array([beta0,beta1,beta2,beta3])
        beta_pred = self.X @ beta_params
        llh = -np.sum(norm.logpdf(self.y,beta_pred,self.sigma))
        return llh

    def fit(self):
        initial_params = np.array([0.1,0.1,0.1,0.1,0.1])
        result = minimize(self.llh,initial_params,method='L-BFGS-B')
        beta0,beta1,beta2,beta3,sig = result.x
        self.beta_ML = np.array([beta0,beta1,beta2,beta3])

    def get_params(self):
        return pd.Series(self.beta_ML, name='Beta coefficients')

    def get_pvalues(self):
        self.n, self.k = self.X.shape
        residuals = self.y - self.X @ self.beta_ML
        sigma2 = (np.square(self.sigma)*self.n) * (self.n-self.k)
        var_cov_matrix = np.linalg.inv(self.X.T @ self.X) * sigma2
        se = np.sqrt(np.diag(var_cov_matrix))
        t_stat = self.beta_ML / se
        p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=self.n-self.k))
        return pd.Series(p_value, name='P-values for the corresponding coefficients')

    def get_model_goodness_values(self):
        y_pred = self.X @ self.beta_ML
        y_mean = np.mean(self.y)
        y_hat = np.dot(self.X, self.beta_ML)
        tss = np.sum((self.y - y_mean)**2)
        rss = np.sum((self.y - y_hat)**2)
        crs = 1 - (rss / tss)
        ars = 1 - (1 - crs) * (self.n - 1) / (self.n - self.k)
        return f"Centered R-squared: {crs:.3f}, Adjusted R-squared: {ars:.3f}"
