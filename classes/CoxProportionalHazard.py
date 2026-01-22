from typing import Any

import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import xlabel


class CoxProportionalHazard:

    def __init__(self):
        self._df = None
        self._wanted_cov = None
        self._beta_values = []
        self.eps = np.finfo(np.float32).eps
        self._result = pl.DataFrame()

    def fit(self, df, time_col, event_col):
        df = df.sort(time_col)
        self._df = df
        self._wanted_cov = df.drop([time_col, event_col])
        self._beta_values = np.zeros([len(self._wanted_cov.columns)])
        for _ in range(0, 1000):

            score_list = []
            diff_list = []
            for row_full, row_cov in zip(df.iter_rows(named = True), self._wanted_cov.iter_rows()):
                if row_full[event_col] == 1:
                    time = row_full[time_col]
                    risk_set = df.filter(pl.col(time_col) >= time)
                    covariates_risk_set = risk_set.drop([time_col, event_col]).to_numpy().T
                    current_val = np.array(row_cov)
                    relative_risk = self.calculate_relative_risk(covariates_risk_set)
                    score_list.append(self.__score_normal_CPHM(current_val, relative_risk, covariates_risk_set))
                    diff_list.append(self.__hessian_matrix_normal(relative_risk, covariates_risk_set.T))
            h_total = sum(diff_list)

            beta_delta = np.linalg.solve(h_total, sum(score_list))

            if np.all(np.abs(beta_delta) < self.eps):
                break
            self._beta_values += beta_delta
        return self.__create_resulting_dataframe()

    def __create_resulting_dataframe(self):
        self._result = pl.DataFrame({
            "covariate": pl.Series(self._wanted_cov.columns),
            "beta values": np.round(self._beta_values, 2),
            "exp(beta)": np.round(np.exp(self._beta_values), 2)
        })

        return self._result

    def __plot(self, ax):
        if not self._result.is_empty():
            plt.xlabel("Covariates")
            plt.xticks(rotation=40)
            for container in ax.containers:
                ax.bar_label(container)
            plt.show(ax)
        else:
            print("No model has been fit, yet!")

    def plot_beta_value(self):
        temp_data = self._result.sort("beta values")
        ax = sns.barplot(x=temp_data["covariate"], y=np.round(temp_data["beta values"], 2), width=0.75,
                         saturation=1)
        ax.set(ylabel = "beta values", title = "Beta values for all covariates")
        self.__plot(ax)

    def plot_hazard_ratio(self):
        temp_data = self._result.sort("beta values")
        ax = sns.barplot(x=temp_data["covariate"], y=np.round(temp_data["exp(beta)"], 2), width=0.75, saturation=1)
        ax.set(ylabel = "hazard ratio, exp(beta value)", title = "Hazard ratio for all covariates")
        self.__plot(ax)




    def calculate_relative_risk(self,  risk_set) -> Any:
        relative_risk =  np.exp(risk_set.T @ self._beta_values)
        return relative_risk

    def __score_normal_CPHM(self, current_covariate, relative_risk, covariate):
        sum_relative_time_cov = np.sum(relative_risk * covariate, axis = 1)

        return current_covariate - sum_relative_time_cov / sum(relative_risk)

    def __hessian_matrix_normal(self, relative_risk, covariates):
        relative_risk = relative_risk.astype(np.float64)
        rr_sum = np.sum(relative_risk)

        weighted_cov = covariates.T @ (relative_risk[:, None] * covariates)

        weighted_sum = (relative_risk[:, None] * covariates).sum(axis = 0)
        hessian = weighted_cov / rr_sum - np.outer(weighted_sum, weighted_sum) / (rr_sum ** 2)
        return hessian
