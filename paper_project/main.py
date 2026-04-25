import random
import argparse
import numpy as np

from pipeline import TimeSeriesForecasting


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Wind Speed Forecasting')

    # 运行控制
    parser.add_argument('--is_run', type=int, default=1,
                        help='是否训练模型，1表示训练，0表示读取预测结果')
    parser.add_argument('--is_save', type=int, default=1,
                        help='是否保存预测结果和预测图')
    parser.add_argument('--is_draw', type=int, default=1,
                        help='是否展示结果图')
    parser.add_argument('--other_run', type=int, default=1,
                        help='是否运行预处理方法，1表示运行，0表示读取')
    parser.add_argument('--other_save', type=int, default=0,
                        help='是否保存预处理结果和图')
    parser.add_argument('--other_draw', type=int, default=0,
                        help='是否展示预处理结果图')

    # 数据集参数
    parser.add_argument('--base_path', type=str, default='D:/Codes/project/paper4/paper_project/',
                        help='项目路径')
    parser.add_argument('--data_path', type=str, default='D:/Codes/dataset/wind/陆上风/',
                        help='数据集路径')
    parser.add_argument('--filename', type=str, default='海南.csv',
                        help='数据集文件名')
    parser.add_argument('--target', type=str, default='风速(m/s)',
                        help='预测的列名')
    parser.add_argument('--exp_path', type=str, default='exp_dtwe海南',
                        help='实验名称')
    parser.add_argument('--features', type=str, default='S',
                        help='预测任务: M(多因素预测) 或 S(单变量预测)')
    parser.add_argument('--is_norm', type=int, default=1,
                        help='是否归一化，1表示是，0表示否')
    parser.add_argument('--test_rate', type=float, default=0.2,
                        help='测试集比率')

    # 预测任务参数
    parser.add_argument('--seq_len', type=int, default=20,
                        help='输入序列长度（滞后阶数）')
    parser.add_argument('--label_len', type=int, default=2,
                        help='预测步长')
    parser.add_argument('--pred_len', type=int, default=1,
                        help='测试集长度')
    parser.add_argument('--pred_mode', type=str, default='Direct',
                        help='预测模式: MIMO, RecMo, Direct')
    parser.add_argument('--total_len', type=int, default=12,
                        help='多步预测滚动窗口长度')
    parser.add_argument('--model_select', type=str, default='ELM',
                        help='预测模型选择')
    parser.add_argument('--opt_model', type=str, default='ELM',
                        help='需要优化的模型名称')
    parser.add_argument('--is_opt', type=int, default=1,
                        help='是否优化模型，1表示优化，0表示读取已优化参数，-1表示默认参数')

    # STL 分解参数
    parser.add_argument('--dec_method', type=str, default='STL',
                        help='分解方法（当前仅支持 STL）')
    parser.add_argument('--dec_k', type=int, default=3,
                        help='STL 分解层数')
    parser.add_argument('--dec_extra', type=str, default='7',
                        help='STL 周期参数')

    # 损失函数权重
    parser.add_argument('--mmd_weight', type=float, default=0.1,
                        help='MMD 损失权重（用于 custom 类型 ELM 训练和 SSA 适应度函数）')
    parser.add_argument('--rmse_weight', type=float, default=1.0,
                        help='RMSE 损失权重（用于 SSA 适应度函数）')

    # MIC 滞后选择阈值
    parser.add_argument('--mic_threshold', type=float, default=0.2,
                        help='MIC 滞后选择阈值，统计 MIC 值超过该值的个数作为滞后长度')

    args = parser.parse_args()

    # SSA 优化边界（ELM 参数: [elm_filter, C2]）
    args.ssa_bounds = {
        'LB': [8, 1],
        'UB': [1024, 2048],
    }

    # 模型默认超参数
    args.p = {
        'elm_filter': 32,
        'C2': 1,
        'learning_rate': 0.01,
        'Dropout': 0.1,
        'verbose': 0,
        'epochs': 100,
        'batch_size': 32,
        'validation_split': 0.1,
        'patience': 20,
    }

    forecasting = TimeSeriesForecasting(args)
    forecasting.run()


if __name__ == '__main__':
    main()
