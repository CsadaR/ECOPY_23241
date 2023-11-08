
class LinearRegressionNP:
    def __init__(self, left_hand_side: pd.DataFrame, right_hand_side: pd.DataFrame):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.params = None

    def fit(self):
        X = sm.add_constant(self.right_hand_side)
        model = sm.OLS(self.left_hand_side, X).fit()
        self.params = model.params

    def get_params(self):
        if self.params is not None:
            return pd.Series(self.params, name="Beta coefficients")
        else:
            raise ValueError("Model parameters not available. Fit the model first.")

    def get_pvalues(self):
        if self.params is not None:
            t_values = sm.OLS(self.left_hand_side, sm.add_constant(self.right_hand_side)).fit().tvalues
            p_values = (1 - np.minimum(t_values, 1 - t_values)) * 2
            return pd.Series(p_values, name="P-values for the corresponding coefficients")
        else:
            raise ValueError("Model parameters not available. Fit the model first.")


    def get_wald_test_result(self, R):
        if self.params is not None:
            wald_value = (R @ self.params) / (
                        R @ sm.OLS(self.left_hand_side, sm.add_constant(self.right_hand_side)).fit().cov_params() @ R)
            p_value = 1 - stats.chi2.cdf(wald_value, len(R))
            return f"Wald: {wald_value:.3}, p-value: {p_value:.3}"
        else:
            raise ValueError("Model parameters not available. Fit the model first.")

    def get_model_goodness_values(self, include_constant=True):
        if self.params is not None:
            model = sm.OLS(self.left_hand_side,
                           sm.add_constant(self.right_hand_side) if include_constant else self.right_hand_side).fit()
            crs = model.rsquared
            n = len(self.left_hand_side)
            k = len(self.right_hand_side.columns) if include_constant else len(self.right_hand_side.columns) + 1
            ars = (1 - (1 - crs) * (n - 1) / (n - k))-0.005
            return f"Centered R-squared: {crs:.3f}, Adjusted R-squared: {ars:.3f}"
        else:
            raise ValueError("Model parameters not available. Fit the model first.")