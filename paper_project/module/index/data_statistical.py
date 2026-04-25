import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
from statsmodels.stats.stattools import jarque_bera as jb_test
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf
from minepy import MINE


class StatisticalTests:
    def __init__(self, args):
        self.args = args

    def run_tests(self, series, name):
        print('#' * 100, '\n统计测试')
        print(series.describe())
        lags = self.adf_test(series)
        self.lbox_test(series, lags=20, name=name)
        self.jbera_test(series)
        self.sw_test(series)
        self.plot_acf_pacf(series, lags=int(20), name=name)

    def adf_test(self, series):
        print('#' * 50, '\nADF Test')
        dftest = adfuller(series)
        adf = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            adf['Critical Value (%s)' % key] = value
        print(adf)
        return int(adf[2])

    def lbox_test(self, series, lags, name):
        print('#' * 50, 'Ljung-Box Test')
        lb_ans = lb_test(series, lags=lags, boxpierce=False, return_df=False)
        if self.args.other_draw:
            fig = plt.figure(figsize=(10, 4))
            pd.Series(lb_ans[1]).plot(label="Ljung-Box Test p-values")
            plt.xlabel('Lag')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.args.fig_path + name + 'Ljung-Box.png')
            plt.show()
        print('p-values:', np.sum(lb_ans[1]))

    def jbera_test(self, series):
        jb_ans = jb_test(series)
        print('#' * 50, '\nJarque-Bera Test')
        print('Test value:', jb_ans[0])
        print('P value:', jb_ans[1])
        print('Skewness:', jb_ans[2])
        print('Kurtosis:', jb_ans[3])

    def sw_test(self, series):
        print('#' * 50, '\nSW Test')
        sw = stats.shapiro(series)
        print("Shapiro wilk Test Statistic:{}   Pvalue:{}".format(sw.statistic, sw.pvalue))

    def plot_acf_pacf(self, series, lags, name):
        pacf_values = pacf(series, nlags=lags)
        indices = np.where(pacf_values > 0.5)[0]
        print("pacf indices:", indices + 1)
        if self.args.other_draw:
            print('#' * 50, '\nACF and PACF')
            fig = plt.figure(figsize=(16, 8))
            fig1 = fig.add_subplot(211)
            plot_acf(series, lags=lags, ax=fig1)
            plt.ylim(-0.4, 1.2)
            fig2 = fig.add_subplot(212)
            plot_pacf(series, lags=lags, ax=fig2)
            plt.ylim(-0.4, 1.2)
            plt.savefig(self.args.fig_path + name + '.png')
            plt.tight_layout()
            plt.show()
        return indices

    def lag_len(self, series_y, max_lag=10):
        """使用 MIC 计算最优滞后长度，统计 MIC 值超过 mic_threshold 的个数"""
        mic_values = []
        for k in range(1, max_lag + 1):
            X_k = series_y[k:]
            X_original = series_y[:-k]
            mine = MINE()
            mine.compute_score(X_original, X_k)
            mic_values.append(mine.mic())

        print(f"MIC值: {mic_values}")
        count = sum(1 for value in mic_values if value > self.args.mic_threshold)
        print(f"MIC值大于{self.args.mic_threshold}的个数: {count}")
        return count
