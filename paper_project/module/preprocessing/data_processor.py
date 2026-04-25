import pandas as pd
import numpy as np
import os


class DataProcessor:
    def __init__(self, args):
        self.base_path = args.base_path + 'checkpoint/'
        self.exp_path = self.base_path + args.exp_path + '/'
        self.args = args

    def initialize_paths(self):
        sub_paths = {
            'global_data_path': os.path.join(self.base_path, 'global_data_path/'),
            'local_data_path': os.path.join(self.exp_path, '1.local_data_path/'),
            'args_path': os.path.join(self.exp_path, '2.args_path/'),
            'models_path': os.path.join(self.exp_path, '3.models_path/'),
            'fig_path': os.path.join(self.exp_path, '4.fig_path/')
        }

        for sub_path in sub_paths.values():
            os.makedirs(sub_path, exist_ok=True)

        self.args.global_data_path, self.args.local_data_path, self.args.args_path, \
            self.args.models_path, self.args.fig_path = tuple(sub_paths.values())

    def read_data(self):
        """读取指定文件中的数据"""
        file_path = os.path.join(self.args.data_path, self.args.filename)

        if self.args.filename.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif self.args.filename.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        elif self.args.filename.endswith('.txt'):
            data = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError("不支持的文件格式: 请使用 .csv, .xlsx 或 .txt 文件.")
        print(f"成功读取文件: {file_path}")
        return data

    def split_data(self, df_x, df_y, in_start=0):
        df_x = np.array(df_x)
        df_y = np.array(df_y)
        if df_x.ndim == 1:
            df_x = df_x.reshape(-1, 1)
        if df_y.ndim == 1:
            df_y = df_y.reshape(-1, 1)

        datax, datay = [], []
        for i in range(len(df_x) - self.args.seq_len - self.args.label_len + 1):
            in_end = int(in_start + self.args.seq_len)
            out_end = int(in_end + self.args.label_len)
            if out_end < len(df_x) + 1:
                a = df_x[in_start:in_end]
                if df_y.ndim == 1:
                    b = df_y[in_end:out_end]
                else:
                    b = df_y[in_end:out_end, -1].reshape(-1, )
                datax.append(a)
                datay.append(b)
            in_start += 1

        datax, datay = np.array(datax), np.array(datay)
        print('相空间完成', datax.shape, datay.shape)
        return datax, datay
