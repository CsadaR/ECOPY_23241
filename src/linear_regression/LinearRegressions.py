import statsmodels.api as sm
from typing import List
import pandas as pd
import numpy as np
from scipy import stats

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
        self.right_hand_side = pd.concat(
            [pd.Series(1, index=self.right_hand_side.index, name='Constant'), self.right_hand_side], axis=1)
        self.X = self.right_hand_side.to_numpy()
        self.y = self.left_hand_side.to_numpy()
        self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        self.beta_coefficients = pd.Series(self.beta, index=self.right_hand_side.columns, name='Beta coefficients')

    def get_params(self):
        return self.beta_coefficients

    def get_pvalues(self):
        n, k = self.right_hand_side.shape[0], self.right_hand_side.shape[1]
        SSE = ((self.y - self.X @ self.beta) ** 2).sum() / (n - k)
        XTX_inv = np.linalg.inv(self.X.T @ self.X)
        t_stats = self.beta / np.sqrt(SSE * np.diag(XTX_inv))
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))
        return pd.Series(p_values, index=self.right_hand_side.columns,
                         name='P-values for the corresponding coefficients')


    def get_wald_test_result(self, R: List[List[int]]) -> str:
        R = np.array(R)
        n, k = self.right_hand_side.shape[0], self.right_hand_side.shape[1]
        SSE = ((self.y - self.X @ self.beta) ** 2).sum() / (n - k)
        XTX_inv = np.linalg.inv(self.X.T @ self.X)
        t_stats = self.beta / np.sqrt(SSE * np.diag(XTX_inv))
        wald_value = (R @ self.beta) @ np.linalg. inv(R @ XTX_inv @ R.T) @ (R @ self.beta)
        p_value = 1 - stats.f.cdf(wald_value, len(R), n - k)
        return f"Wald: {wald_value:.3f}, p-value: {p_value:.3f}"

    def get_model_goodness_values(self) -> str:
        n, k = self.right_hand_side.shape[0], self.right_hand_side.shape[1]
        SSE = ((self.y - self.X @ self.beta) ** 2).sum()
        SST = ((self.y - self.y.mean()) ** 2).sum()
        SSR = SST - SSE
        crs = SSR / SST
        ars = 1 - (1 - crs) * ((n - 1) / (n - k))
        return f"Centered R-squared: {crs:.3f}, Adjusted R-squared: {ars:.3f}"





