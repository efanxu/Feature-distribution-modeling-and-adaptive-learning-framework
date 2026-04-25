import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

from module.preprocessing import data_processor, decomposition
from module.index import data_statistical, evaluation
from module.models import model_trainer
from module.utils.dtw_extension import find_segment, find_similar_segment


def generate_filename(data_name='', season='', parameter_set='', model='', step=''):
    return f"{data_name}+{season}+{parameter_set}+{model}+{step}_step"


class TimeSeriesForecasting:

    def __init__(self, args):
        self.args = args
        self.statistical = data_statistical.StatisticalTests(args)
        self.processor = data_processor.DataProcessor(args)
        self.processor.initialize_paths()
        self.signal_decomp = decomposition.SignalDecomposition(args, data_path=self.args.local_data_path)
        self.trainer = model_trainer.ModelTrainer(args)

    def initialize(self):
        data = self.processor.read_data()
        data['时间索引'] = pd.to_datetime(data['日期'] + ' ' + data['时间'])
        data = data.drop(columns=['日期', '时间'])
        data.set_index('时间索引', inplace=True)

        season_ranges = {
            '春季': ('2022-03-01', '2022-05-31'),
            '夏季': ('2022-06-01', '2022-08-31'),
            '秋季': ('2022-09-01', '2022-11-30'),
            '冬季': ('2022-12-01', '2023-02-28'),
        }
        if self.args.season in season_ranges:
            start_date, end_date = season_ranges[self.args.season]
            data = data[(data.index >= start_date) & (data.index <= end_date)]

        if self.args.features == 'M':
            self.data_x = data
        elif self.args.features == 'S':
            self.data_x = pd.DataFrame(data[[self.args.target]], columns=[self.args.target])

        self.data_y = data[[self.args.target]]
        self.series_y = self.data_x[self.args.target]

        self.args.val_len = int(len(self.data_y) * self.args.test_rate)
        self.args.seq_len = self.statistical.adf_test(self.series_y[:-self.args.pred_len])

        if self.args.is_norm:
            scaler_x = MinMaxScaler(feature_range=(0, 1))
            scaler_x.fit(self.data_x.iloc[:-self.args.pred_len, :].values)
            self.data_sc_x = scaler_x.transform(self.data_x.values)

            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler.fit(self.data_y.iloc[:-self.args.pred_len, :].values)
            self.data_sc_y = self.scaler.transform(self.data_y.values)

    def split_data(self, df_x, df_y, in_start=0):
        df_x = np.array(df_x)
        df_y = np.array(df_y)
        if df_x.ndim == 1:
            df_x = df_x.reshape(-1, 1)
        if df_y.ndim == 1:
            df_y = df_y.reshape(-1, 1)

        datax, datay = [], []
        for i in range(len(df_x) - self.args.label_len + 1):
            in_end = in_start
            out_end = int(in_end + self.args.label_len) + 1
            if out_end < len(df_x) + 1:
                a = df_x[in_end]
                if df_y.ndim == 1:
                    b = df_y[in_end + 1:out_end]
                else:
                    b = df_y[in_end + 1:out_end, -1].reshape(-1, )
                datax.append(a)
                datay.append(b)
            in_start += 1

        datax, datay = np.array(datax), np.array(datay)
        print('相空间完成', datax.shape, datay.shape)
        return datax, datay

    def _run_single_experiment(self, ext, w, th, rmse_w, mmd_w, dec_ext, mic_th, init=500):
        # 将所有参数赋值给 args
        self.args.extend_len = ext
        self.args.rmse_weight = rmse_w
        self.args.mmd_weight = mmd_w
        self.args.dec_extra = str(dec_ext)

        self.args.mic_threshold = mic_th
        self.args.map_len = self.statistical.lag_len(self.data_sc_y.ravel())
        self.args.seq_len = self.args.map_len

        map_len = self.args.map_len

        print(
            f"\n>>> 开始评估参数组合: Season={self.args.season}, Extend={ext}, Window={w}, "
            f"Threshold={th}, rmse_w={rmse_w}, mmd_w={mmd_w}, dec_ext={dec_ext}, "
            f"mic_th={mic_th}, map_len={map_len}"
        )

        tag = f"ext{ext}_w{w}_th{th}_rw{rmse_w}_mw{mmd_w}_de{dec_ext}_mt{mic_th}_ml{map_len}"

        dtw_name = f"{self.args.filename[:2]}+{self.args.season}+DTW_{tag}"
        file_path = os.path.join(self.args.global_data_path, dtw_name + '.csv')

        if os.path.exists(file_path):
            print(f"找到已有缓存文件，直接读取: {file_path}")
            map_x1 = pd.read_csv(file_path).values
        else:
            print("缓存文件不存在，开始执行 DTW 计算...")
            dtw_time1 = time.time()
            map_x1_df = pd.DataFrame()

            for idx in range(init, len(self.data_sc_y) + 1):
                cur_data = self.data_sc_y[:idx].copy()
                result = find_segment(cur_data)

                similar, next_seg = find_similar_segment(
                    X=result, Y=cur_data[:-self.args.extend_len], YY=cur_data,
                    extend_len=self.args.extend_len, window=w, dynamic_threshold=th
                )
                concat_data = np.concatenate((cur_data, next_seg), axis=0)

                map_x1_df = pd.concat(
                    [map_x1_df, pd.DataFrame(concat_data[-self.args.extend_len - 2 * map_len:]).T],
                    axis=0, ignore_index=True
                )

            dtw_time2 = time.time()
            print('dtw时间:', dtw_time2 - dtw_time1)

            map_x1_df.to_csv(file_path, index=False)

            map_x1 = map_x1_df.values

        # 2. 映射模型
        self.args.model_select = 'ELM'
        self.args.opt_model = 'ELM'
        self.args.is_opt = 1

        map_time1 = time.time()
        dec_train_data = self.signal_decomp.decompose(self.data_sc_y[:-self.args.val_len], name='')

        map_y_list, true_y_list = [], []
        for idx in range(init, len(dec_train_data) + 1):
            update_data = dec_train_data.iloc[idx - map_len:idx, :]
            map_y_list.append(update_data.values)
        for idx in range(init, len(self.data_sc_y) + 1):
            update_y = self.data_sc_y[idx - map_len:idx]
            true_y_list.append(update_y)

        true_y = np.array(true_y_list).reshape(-1, map_len)
        map_y = np.stack(map_y_list, axis=0)

        predictions = {}
        for s_num in range(1, 3):
            map_sy = map_y[:, :, s_num]
            train_map_x = map_x1[:-self.args.val_len]
            train_map_y = map_sy

            map_name = generate_filename(
                data_name=self.args.filename[:2], season=self.args.season,
                parameter_set=f"{self.args.dec_method}+S{s_num + 1}_Map_{tag}",
                model=self.args.model_select, step=self.args.label_len
            )
            optimized_params = self.trainer._load_or_optimize_parameters(map_name, train_map_x, train_map_y)
            for key, value in optimized_params.items():
                self.args.p[key] = value

            map_model = self.trainer._model_set(map_name, train_map_x, train_map_y, shape=train_map_x.shape)
            predictions[f"S{s_num + 1}"] = map_model.predict(map_x1)
            pd.DataFrame.to_csv(
                pd.DataFrame(predictions[f"S{s_num + 1}"]),
                self.args.local_data_path + map_name + '.csv', index=False
            )

        S2, S3 = predictions['S2'], predictions['S3']
        S1 = true_y - S2 - S3
        S = np.concatenate((S1, S2, S3), axis=1)

        map_time2 = time.time()
        map_time = map_time2 - map_time1

        self.args.elm_type = 'reg'
        pred_time1 = time.time()
        datax, datay = self.split_data(S, self.data_sc_y[-len(S):])
        train_x, test_x, train_y, test_y = train_test_split(datax, datay, test_size=self.args.val_len, shuffle=False)

        ensemble_name = generate_filename(
            data_name=self.args.filename[:2], season=self.args.season,
            parameter_set=f"{self.args.dec_method}+Ensemble_{tag}",
            model=self.args.model_select, step=self.args.label_len
        )
        optimized_params = self.trainer._load_or_optimize_parameters(ensemble_name, train_x, train_y)
        for key, value in optimized_params.items():
            self.args.p[key] = value

        ensemble_model = self.trainer._model_set(ensemble_name, train_x, train_y, shape=test_x.shape)
        y_pred = ensemble_model.predict(test_x)

        pred_time2 = time.time()
        pred_time = pred_time2 - pred_time1

        y_pred = self.scaler.inverse_transform(y_pred)
        test_y = self.scaler.inverse_transform(test_y)
        pd.DataFrame.to_csv(pd.DataFrame(y_pred), self.args.local_data_path + ensemble_name + '.csv', index=False)

        total_time = map_time + pred_time
        evaluation_metrics = evaluation.EvaluationMetrics(self.args.base_path, name=ensemble_name)
        evaluation_metrics.deter_metrices(y_pred, test_y, run_time=total_time)

        y_pred_flat = np.array(y_pred).ravel()
        test_y_flat = np.array(test_y).ravel()
        mae = mean_absolute_error(test_y_flat, y_pred_flat)
        rmse = mean_squared_error(test_y_flat, y_pred_flat, squared=False)
        mape = mean_absolute_percentage_error(test_y_flat, y_pred_flat)
        ia = evaluation_metrics.calculate_ia(test_y_flat, y_pred_flat)

        record = {
            'Season': self.args.season,
            'Extend_Len': ext,
            'Window': w,
            'Threshold': th,
            'RMSE_Weight': rmse_w,
            'MMD_Weight': mmd_w,
            'Dec_Extra': dec_ext,
            'MIC_Threshold': mic_th,
            'Map_Len': map_len,
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'MAPE': round(mape, 4),
            'IA': round(ia, 4),
            'Total_Time': round(total_time, 2),
        }
        return mae, record

    def run(self):
        seasons = ['春季', '夏季', '秋季', '冬季']

        extend_lens        = [10, 15, 20, 25, 30]
        windows            = [1, 2, 3, 4, 5]
        dynamic_thresholds = [3.0, 5.0, 7.0, 9.0, 11.0]
        rmse_weights       = [0.1, 0.5, 1.0, 1.5, 2.0]
        mmd_weights        = [0.01, 0.05, 0.1, 0.2, 0.5]
        dec_extras         = [3, 5, 7, 9, 11]
        mic_thresholds     = [0.10, 0.15, 0.20, 0.25, 0.30]

        DEF_EXT   = extend_lens[2]
        DEF_W     = windows[2]
        DEF_TH    = dynamic_thresholds[2]
        DEF_RW    = rmse_weights[2]
        DEF_MW    = mmd_weights[2]
        DEF_DE    = dec_extras[2]
        DEF_MT    = mic_thresholds[2]

        all_results = []
        experiment_cache = {}

        for season in seasons:
            self.args.season = season
            self.initialize()
            print('验证集长度:', self.args.val_len)
            print('测试集长度:', self.args.pred_len)

            self.args.elm_type = 'custom'
            init = 500

            def run_with_cache(ext, w, th, rw, mw, de, mt):
                key = (season, ext, w, th, rw, mw, de, mt)
                if key in experiment_cache:
                    return experiment_cache[key]['mae'], experiment_cache[key]['record'].copy()
                mae, record = self._run_single_experiment(ext, w, th, rw, mw, de, mt, init)
                experiment_cache[key] = {'mae': mae, 'record': record.copy()}
                return mae, record.copy()

            best_ext = DEF_EXT
            best_w   = DEF_W
            best_th  = DEF_TH
            best_rw  = DEF_RW
            best_mw  = DEF_MW
            best_de  = DEF_DE
            best_mt  = DEF_MT

            # ── 阶段 1：最优 Extend Length ──────────────────────────────────
            print(f"\n--- 【阶段1】寻找最优 Extend Length "
                  f"(固定 w={best_w}, th={best_th}, rw={best_rw}, mw={best_mw}, de={best_de}, mt={best_mt}) ---")
            min_mae = float('inf')
            for ext in extend_lens:
                mae, record = run_with_cache(ext, best_w, best_th, best_rw, best_mw, best_de, best_mt)
                record['Stage'] = '1_Opt_Extend'
                all_results.append(record)
                if mae < min_mae:
                    min_mae, best_ext = mae, ext
            print(f">>> 阶段1完成: 选定 Extend Length = {best_ext} (MAE: {min_mae:.3f})\n")

            # ── 阶段 2：最优 Window ─────────────────────────────────────────
            print(f"\n--- 【阶段2】寻找最优 Window "
                  f"(固定 ext={best_ext}, th={best_th}, rw={best_rw}, mw={best_mw}, de={best_de}, mt={best_mt}) ---")
            min_mae = float('inf')
            for w in windows:
                mae, record = run_with_cache(best_ext, w, best_th, best_rw, best_mw, best_de, best_mt)
                record['Stage'] = '2_Opt_Window'
                all_results.append(record)
                if mae < min_mae:
                    min_mae, best_w = mae, w
            print(f">>> 阶段2完成: 选定 Window = {best_w} (MAE: {min_mae:.3f})\n")

            # ── 阶段 3：最优 Dynamic Threshold ──────────────────────────────
            print(f"\n--- 【阶段3】寻找最优 Dynamic Threshold "
                  f"(固定 ext={best_ext}, w={best_w}, rw={best_rw}, mw={best_mw}, de={best_de}, mt={best_mt}) ---")
            min_mae = float('inf')
            for th in dynamic_thresholds:
                mae, record = run_with_cache(best_ext, best_w, th, best_rw, best_mw, best_de, best_mt)
                record['Stage'] = '3_Opt_Threshold'
                all_results.append(record)
                if mae < min_mae:
                    min_mae, best_th = mae, th
            print(f">>> 阶段3完成: 选定 Dynamic Threshold = {best_th} (MAE: {min_mae:.3f})\n")

            # ── 阶段 4：最优 STL 周期 (dec_extra) ───────────────────────────
            print(f"\n--- 【阶段4】寻找最优 STL 周期 "
                  f"(固定 ext={best_ext}, w={best_w}, th={best_th}, rw={best_rw}, mw={best_mw}, mt={best_mt}) ---")
            min_mae = float('inf')
            for de in dec_extras:
                mae, record = run_with_cache(best_ext, best_w, best_th, best_rw, best_mw, de, best_mt)
                record['Stage'] = '4_Opt_DecExtra'
                all_results.append(record)
                if mae < min_mae:
                    min_mae, best_de = mae, de
            print(f">>> 阶段4完成: 选定 STL 周期 = {best_de} (MAE: {min_mae:.3f})\n")

            # ── 阶段 5：最优 RMSE 权重 ───────────────────────────────────────
            print(f"\n--- 【阶段5】寻找最优 RMSE 权重 "
                  f"(固定 ext={best_ext}, w={best_w}, th={best_th}, mw={best_mw}, de={best_de}, mt={best_mt}) ---")
            min_mae = float('inf')
            for rw in rmse_weights:
                mae, record = run_with_cache(best_ext, best_w, best_th, rw, best_mw, best_de, best_mt)
                record['Stage'] = '5_Opt_RmseWeight'
                all_results.append(record)
                if mae < min_mae:
                    min_mae, best_rw = mae, rw
            print(f">>> 阶段5完成: 选定 RMSE 权重 = {best_rw} (MAE: {min_mae:.3f})\n")

            # ── 阶段 6：最优 MMD 权重 ────────────────────────────────────────
            print(f"\n--- 【阶段6】寻找最优 MMD 权重 "
                  f"(固定 ext={best_ext}, w={best_w}, th={best_th}, rw={best_rw}, de={best_de}, mt={best_mt}) ---")
            min_mae = float('inf')
            for mw in mmd_weights:
                mae, record = run_with_cache(best_ext, best_w, best_th, best_rw, mw, best_de, best_mt)
                record['Stage'] = '6_Opt_MmdWeight'
                all_results.append(record)
                if mae < min_mae:
                    min_mae, best_mw = mae, mw
            print(f">>> 阶段6完成: 选定 MMD 权重 = {best_mw} (MAE: {min_mae:.3f})\n")

            # ── 阶段 7：最优 MIC 阈值 ────────────────────────────────────────
            print(f"\n--- 【阶段7】寻找最优 MIC 阈值 "
                  f"(固定 ext={best_ext}, w={best_w}, th={best_th}, rw={best_rw}, mw={best_mw}, de={best_de}) ---")
            min_mae = float('inf')
            for mt in mic_thresholds:
                mae, record = run_with_cache(best_ext, best_w, best_th, best_rw, best_mw, best_de, mt)
                record['Stage'] = '7_Opt_MicThreshold'
                all_results.append(record)
                if mae < min_mae:
                    min_mae, best_mt = mae, mt
            print(f">>> 阶段7完成: 选定 MIC 阈值 = {best_mt} (MAE: {min_mae:.3f})\n")

            print(
                f"========== {season} \033[1m最终最优参数组合\033[0m: "
                f"Extend=\033[1m{best_ext}\033[0m, "
                f"Window=\033[1m{best_w}\033[0m, "
                f"Threshold=\033[1m{best_th}\033[0m, "
                f"STL周期=\033[1m{best_de}\033[0m, "
                f"RMSE权重=\033[1m{best_rw}\033[0m, "
                f"MMD权重=\033[1m{best_mw}\033[0m, "
                f"MIC阈值=\033[1m{best_mt}\033[0m "
                f"(最优MAE: \033[1m{min_mae:.3f}\033[0m) ==========\n"
            )

        results_df = pd.DataFrame(all_results)
        output_csv = os.path.join(
            self.args.base_path + 'checkpoint/' + self.args.exp_path + '/',
            "parameter_sensitivity_results + 2_step.csv"
        )
        results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n 所有敏感性分析结果已保存至: {output_csv}")
