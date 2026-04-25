import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import math
import os


class EvaluationMetrics:
    def __init__(self, path, name):
        self.path = path
        self.name = name

    def save_to_excel(self, eval_results):
        file_name = self.path + 'evaluation_results.xlsx'
        if os.path.exists(file_name):
            existing_df = pd.read_excel(file_name)
            combined_df = pd.concat([existing_df, eval_results], ignore_index=True)
        else:
            combined_df = eval_results
        combined_df.to_excel(file_name, index=False)

    def smape(self, y_true, y_pred):
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2
        denom = np.where(denom == 0, 1, denom)
        return np.mean(np.abs(y_pred - y_true) / denom)

    def deter_metrices(self, y_test, y_pred, run_time):
        y_test, y_pred = np.array(y_test).ravel(), np.array(y_pred).ravel()
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        smape = self.smape(y_test, y_pred)
        nrmse = rmse / (np.max(y_test) - np.min(y_test))
        ia = self.calculate_ia(y_test, y_pred)
        r2, rmse, nrmse, mae, mape, smape, ia = [round(value, 3) for value in [r2, rmse, nrmse, mae, mape, smape, ia]]

        print(f'---- {self.name} Deterministic Evaluation ----')
        print('MAE\tRMSE\tSMAPE\tIA')
        print(mae, '\t', rmse, '\t', smape, '\t', ia)

        eval_results = pd.DataFrame({
            'Model': [self.name],
            'MAE': [mae],
            'RMSE': [rmse],
            'NRMSE': [nrmse],
            'MAPE': [mape],
            'IA': [ia],
            'R2': [r2],
            'Time': [run_time],
        })

        print('-' * 80)
        return eval_results

    def inter_metrices(self, y_test, y_max, y_min):
        picp = self._PICP(y_test, y_max, y_min)
        pinaw = self._PINAW(y_test, y_max, y_min)
        cwc = self._CWC(picp, pinaw)
        ais = self._AIS(y_test, y_max, y_min)
        cpia = self._calculate_cpia(y_test, y_max, y_min, pinaw)
        picp, pinaw, cwc, ais, cpia = [round(value, 4) for value in [picp, pinaw, cwc, ais, cpia]]

        print(f'---- {self.name} Interval Evaluation ----')
        print(f'{"PICP":^10}\t{"PINAW":^10}\t{"CWC":^10}\t{"AIS":^10}\t{"CPIA":^10}')
        print(f'{picp:^10.3f}\t{pinaw:^10.3f}\t{cwc:^10.3f}\t{ais:^10.3f}\t{cpia:^10.3f}')

        eval_results = pd.DataFrame({
            'Model': [self.name],
            'PICP': [picp],
            'PINAW': [pinaw],
            'CWC': [cwc],
            'AIS': [ais],
            'CPIA': [cpia],
        })

        print('-' * 80)
        return eval_results

    def calculate_ia(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        y_mean = np.mean(y_true)
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((np.abs(y_pred - y_mean) + np.abs(y_true - y_mean)) ** 2)
        return 1 - (numerator / denominator)

    def calculate_ct(self, y_hat, U, L, d):
        T = len(y_hat)
        ct = np.zeros(T)
        for t in range(T):
            if U[t] <= y_hat[t] <= U[t] + d[t] / 2:
                ct[t] = 1 - (y_hat[t] - U[t]) / d[t]
            elif L[t] <= y_hat[t] <= U[t]:
                ct[t] = 1
            elif L[t] - d[t] / 2 <= y_hat[t] <= L[t]:
                ct[t] = 1 - (L[t] - y_hat[t]) / d[t]
            else:
                ct[t] = 0
        return ct

    def _calculate_cpia(self, y_hat, U, L, PINAW):
        d = U - L
        ct = self.calculate_ct(y_hat, U, L, d)
        return np.mean((1 - PINAW) * ct)

    def _PICP(self, y_true, y_pred_max, y_pred_min):
        y_true, y_pred_max, y_pred_min = np.array(y_true), np.array(y_pred_max), np.array(y_pred_min)
        count = sum(1 for i in range(len(y_true)) if y_pred_min[i] <= y_true[i] <= y_pred_max[i])
        return count / len(y_true)

    def _PINAW(self, y_true, y_pred_max, y_pred_min):
        y_true, y_pred_max, y_pred_min = np.array(y_true), np.array(y_pred_max), np.array(y_pred_min)
        R = np.max(y_true) - np.min(y_true)
        R1 = [y_pred_max[i] - y_pred_min[i] for i in range(len(y_true))]
        return np.mean(R1) / R

    def _CWC(self, PICP, PINAW, v=0.9, n=10):
        if PICP <= v:
            return PINAW * (1 + math.exp(n * (v - PICP)))
        return PINAW

    def _AIS(self, y_true, y_pred_max, y_pred_min, alpha=0.1):
        y_true, y_pred_max, y_pred_min = np.array(y_true), np.array(y_pred_max), np.array(y_pred_min)
        S1 = []
        for i in range(len(y_true)):
            if y_pred_min[i] <= y_true[i] <= y_pred_max[i]:
                S1.append(-2 * alpha * (y_pred_max[i] - y_pred_min[i]))
            elif y_true[i] < y_pred_min[i]:
                S1.append(-2 * alpha * (y_pred_max[i] - y_pred_min[i]) - 4 * (y_pred_min[i] - y_true[i]))
            else:
                S1.append(-2 * alpha * (y_pred_max[i] - y_pred_min[i]) - 4 * (y_true[i] - y_pred_max[i]))
        return np.mean(S1)
