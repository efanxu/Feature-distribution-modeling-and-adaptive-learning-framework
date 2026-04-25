import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL


class SignalDecomposition:
    def __init__(self, args, data_path):
        self.args = args
        self.data_path = data_path

    def decompose(self, series, name):
        if not isinstance(series, pd.Series):
            series = (series.values if isinstance(series, pd.DataFrame) else series).flatten()
            series = pd.Series(series)

        if self.args.other_run:
            dec_df = self.stl_decom(series, self.args.dec_k, self.args.dec_extra)
        else:
            dec_df = pd.read_csv(self.data_path + name + '.csv')

        if self.args.other_draw:
            self.draw_dec(series, dec_df, name, self.args.other_save)

        if self.args.other_save:
            pd.DataFrame.to_csv(dec_df, self.data_path + name + '.csv', index=False)

        return dec_df

    def draw_dec(self, data, S, name, other_save):
        series = data[-len(S):]
        S_num = int(len(S.columns))
        fig, axs = plt.subplots(S_num + 1, 1, figsize=(12, 10))
        for i in range(S_num + 1):
            ax = axs[i]
            ax.plot(series if i == 0 else S.iloc[:, i - 1])
            ax.set_ylabel('Original' if i == 0 else 'S' + str(i), fontweight='bold')
        axs[0].set_title(str.upper(name), fontweight='bold')
        fig.align_labels()
        plt.tight_layout()
        if other_save:
            plt.savefig(self.args.fig_path + name + '.png')
        plt.pause(2)
        plt.close()

    def stl_decom(self, series, k, extra):
        extra = int(extra)
        stl = STL(series, period=extra)
        result = stl.fit()
        trend_series = pd.Series(result.trend)
        seasonal_series = pd.Series(result.seasonal)
        resid_series = pd.Series(result.resid)
        stl_df = pd.concat([resid_series, seasonal_series, trend_series], axis=1)
        stl_df.columns = ['S1', 'S2', 'S3']
        return stl_df
