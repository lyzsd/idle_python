import numpy as np
import pandas as pd
import warnings
import pickle
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tqdm import tqdm
import matplotlib.pyplot as plt
F = [
    [300000, 403200, 499200, 595200, 691200, 806400, 902400, 998400, 1094400, 1209600, 1305600, 1401600, 1497600,
     1612800, 1708800, 1804800],
    [710400, 844800, 960000, 1075200, 1209600, 1324800, 1440000, 1555200, 1670400, 1766400, 1881600, 1996800, 2112000,
     2227200, 2342400, 2419200],
    [844800, 960000, 1075200, 1190400, 1305600, 1420800, 1555200, 1670400, 1785600, 1900800, 2035200, 2150400, 2265600,
     2380800, 2496000, 2592000, 2688000, 2764800, 2841600]
]

opp0 = {
            300000: 584, 403200: 584, 499200: 584,
            595200: 584, 691200: 584, 806400: 600,
            902400: 616, 998400: 636, 1094400: 648,
            1209600: 672, 1305600: 696, 1401600: 716,
            1497600: 746, 1612800: 768, 1708800: 784,
            1804800: 808
        }
opp4 = {
    710400: 612, 844800: 636, 960000: 656,
    1075200: 680, 1209600: 692, 1324800: 716,
    1440000: 736, 1555200: 764, 1670400: 788,
    1766400: 808, 1881600: 836, 1996800: 864,
    2112000: 896, 2227200: 924, 2342400: 956,
    2419200: 988
}
opp7 = {
    844800: 628, 960000: 640, 1075200: 644,
    1190400: 652, 1305600: 656, 1420800: 668,
    1555200: 692, 1670400: 708, 1785600: 732,
    1900800: 752, 2035200: 776, 2150400: 808,
    2265600: 840, 2380800: 872, 2496000: 896,
    2592000: 932, 2688000: 956, 2764800: 976,
    2841600: 996
}
opp0_power = {300000: 9, 403200: 12, 499200: 15,
        595200: 18, 691200: 21, 806400: 26,
        902400: 31, 998400: 36, 1094400: 42,
        1209600: 49, 1305600: 57, 1401600: 65, 1497600: 0,
        1612800: 89, 1708800: 100, 1804800: 115}

opp4_power = {710400: 125, 844800: 161, 960000: 198,
        1075200: 236, 1209600: 275, 1324800: 327,
        1440000: 380, 1555200: 443, 1670400: 512,
        1766400: 575, 1881600: 655, 1996800: 750,
        2112000: 853, 2227200: 965, 2342400: 1086,
        2419200: 1178}

opp7_power = { 844800: 221, 960000: 266, 1075200: 306,
    1190400: 356, 1305600: 401, 1420800: 458,
    1555200: 540, 1670400: 614, 1785600: 695,
    1900800: 782, 2035200: 893, 2150400: 1035,
    2265600: 1203, 2380800: 1362, 2496000: 1536,
    2592000: 1725, 2688000: 1898, 2764800: 2017,
    2841600: 2141}
def filter_false(lst):
    return list(filter(bool, lst))
def find_temp_interval(total_lenght , number, total_intervals):
    #print(total_intervals)
    interval_size = total_lenght / total_intervals
    interval_index = int(number / interval_size)
    return interval_index
def translate_list(matrix):
    matrix = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
    return matrix
def idle_model_make(file_name, app, ii, itemp):
    model_data = pd.read_csv(r'data_process/{}/{}_{}_{}/power_model_file.csv'.format(file_name, app, ii, itemp))
    model_data = model_data[:len(model_data)-11]
    for i in range(len(model_data)):
        model_data['power'][i] = model_data['power'][i] / 10 * 3

    result_df = pd.DataFrame(columns=model_data.columns)
    # 循环遍历原始DataFrame，累加行数随机
    i = 0
    while i < len(model_data):
        j = np.random.randint(1, 10)  # 随机选择累加的行数，范围为1到10
        end_index = min(i + j, len(model_data))
        rows_to_sum = model_data.iloc[i:end_index]
        summed_row = rows_to_sum.sum()
        result_df = result_df.append(summed_row, ignore_index=True)
        i = end_index
    # print(model_data)
    # print(result_df)
    model_data = result_df
    X = sm.add_constant(model_data[['rt', 'idle1', 'idle0']])
    # 创建 OLS 模型对象

    model = sm.OLS(model_data['power'], X)
    # 拟合模型
    constraints = [{'loc': 'coef', 'type': 'ineq', 'fun': lambda x: x[1:]}, ]
    results = model.fit(constraints=constraints)
    # 查看回归结果
    print(results.summary())
    with open('model_pkl/big/ols_model_{}.pkl'.format(ii), 'wb') as f:
        pickle.dump(results, f)


def idle_model_test(file_name, app, ii, itemp):
    model_data = pd.read_csv(r'data_process/{}/{}_{}_{}/power_model_file.csv'.format(file_name, app, ii, itemp))
    with open('model_pkl/big/ols_model_{}.pkl'.format(ii), 'rb') as f:
        loaded_model = pickle.load(f)
    # 假设有新的数据需要预测
    model_data = model_data[:len(model_data) - 11]
    for i in range(len(model_data)):
        model_data['power'][i] = model_data['power'][i] / 10 * 3
    new_data = model_data[['rt', 'idle1', 'idle0']]
    # 添加截距项
    new_data = sm.add_constant(new_data)
    # 进行预测
    predictions = loaded_model.predict(new_data)

    y_true = model_data['power']
    print(predictions)
    print(y_true)
    y_pred = predictions
    mse = mean_squared_error(y_true, y_pred)
    print("MSE:", mse)
    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mse)
    print("RMSE:", rmse)
    # 计算平均绝对误差（MAE）
    mae = mean_absolute_error(y_true, y_pred)
    print("MAE:", mae)
    # 计算决定系数（R-squared）
    r2 = r2_score(y_true, y_pred)
    print("R-squared:", r2)
    # 计算误差率
    error_rate = ((y_pred - y_true) / y_true) * 100

    # 输出误差率
    print("误差率:", np.mean(abs(error_rate)))

    print("随机采样功耗")
    result_df = pd.DataFrame(columns=model_data.columns)
    # 循环遍历原始DataFrame，累加行数随机
    i = 0
    while i < len(model_data):
        j = np.random.randint(1, 10)  # 随机选择累加的行数，范围为1到10
        end_index = min(i + j, len(model_data))
        rows_to_sum = model_data.iloc[i:end_index]
        summed_row = rows_to_sum.sum()
        result_df = result_df.append(summed_row, ignore_index=True)
        i = end_index
    model_data = result_df
    new_data = model_data[['rt', 'idle1', 'idle0']]
    # 添加截距项
    new_data = sm.add_constant(new_data)
    # 进行预测
    predictions = loaded_model.predict(new_data)
    y_true = model_data['power']
    y_pred = predictions
    mse = mean_squared_error(y_true, y_pred)
    print("MSE:", mse)
    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mse)
    print("RMSE:", rmse)
    # 计算平均绝对误差（MAE）
    mae = mean_absolute_error(y_true, y_pred)
    print("MAE:", mae)
    # 计算决定系数（R-squared）
    r2 = r2_score(y_true, y_pred)
    print("R-squared:", r2)
    # 计算误差率
    error_rate = ((y_pred - y_true) / y_true) * 100

    # 输出误差率
    print("误差率:", np.mean(abs(error_rate)))
    # plt.scatter(range(len(y_pred)), y_pred,c='blue')
    # plt.scatter(range(len(y_pred)), y_true, c='red')
    # plt.show()
    return np.mean(abs(error_rate))
def idle_model_read(file_name, app, ii, itemp):
    with open('model_pkl/big/ols_model_{}.pkl'.format(ii), 'rb') as f:
        loaded_model = pickle.load(f)
    print(loaded_model.summary())