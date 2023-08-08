import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

little_freq = [300000, 403200, 499200, 595200, 691200, 806400, 902400, 998400, 1094400, 1209600, 1305600, 1401600,
               1497600, 1612800, 1708800, 1804800]

big_freq = [710400, 844800, 960000, 1075200, 1209600, 1324800, 1440000, 1555200, 1670400, 1766400, 1881600, 1996800,
            2112000, 2227200, 2342400, 2419200]

s_big_freq = [844800, 960000, 1075200, 1190400, 1305600, 1420800, 1555200, 1670400, 1785600, 1900800, 2035200, 2150400,
              2265600, 2380800, 2496000, 2592000, 2688000, 2764800, 2841600]


def opp_power(num):
    opp0 = [[300000, 9], [403200, 12], [499200, 15], [595200, 18], [691200, 21], [806400, 26], [902400, 31],
            [998400, 36], [1094400, 42], [1209600, 49], [1305600, 57], [1401600, 65], [1497600, 0], [1612800, 89],
            [1708800, 100], [1804800, 115]]

    opp4 = [[710400, 125], [844800, 161], [960000, 198], [1075200, 236], [1209600, 275], [1324800, 327],
            [1440000, 380], [1555200, 443], [1670400, 512], [1766400, 575], [1881600, 655], [1996800, 750],
            [2112000, 853], [2227200, 965], [2342400, 1086], [2419200, 1178]]

    opp7 = [[844800, 221], [960000, 266], [1075200, 306], [1190400, 356], [1305600, 401], [1420800, 458],
            [1555200, 540], [1670400, 614], [1785600, 695], [1900800, 782], [2035200, 893], [2150400, 1035],
            [2265600, 1203], [2380800, 1362], [2496000, 1536], [2592000, 1725], [2688000, 1898], [2764800, 2017],
            [2841600, 2141]]

    opp = []
    if num == 0:
        opp = opp0
    elif num == 4:
        opp = opp4
    elif num == 7:
        opp = opp7

    return opp  # opp能耗功率对应表


# 按 ⌃R 执行或将其替换为您的代码。
# 按 双击 ⇧ 在所有地方搜索类、文件、工具窗口、操作和设置。


def get_freq(util_path: str = 'trace', out_path: str = '') -> None:
    """
    在adb shell上使用simpleperf获取l3d_cache_refill、mem_access、ipc
    对数据适当处理后输出到csv文件中

    Args:
        util_path : 输入文件路径
        out_path :输出文件路径

    """
    big_clu = []
    little_clu = []
    flag = False
    each_little = []
    each_big = []
    # thislen = 0
    # thisall = 0
    for line in open(util_path):
        if line[0] == '#':
            continue
        line = line.strip().split(': ')
        if line[1] != 'get_my_util':
            continue
        time = float(line[0].split()[-1])
        cpu = int(line[-1].split()[0])
        util = int(line[-1].split()[1])
        if not flag and not (cpu == 0 or cpu == 4):
            continue
        flag = True
        if cpu == 0:
            each_little = [time, util]
        elif cpu == 4:
            each_big = [time, util]
        if 4 > cpu > 0:
            each_little.append(util)
        elif cpu > 4:
            each_big.append(util)
        if cpu == 3:
            if len(each_little) == 5:
                little_clu.append(each_little)
            each_little = []
        elif cpu == 6:
            if len(each_big) == 4:
                big_clu.append(each_big)
            each_big = []

    df = pd.DataFrame(data=little_clu)
    df.columns = ['timestamp', 'cpu0', 'cpu1', 'cpu2', 'cpu3']
    # df.to_csv('little_util.csv', header=['timestamp', 'cpu0', 'cpu1', 'cpu2', 'cpu3'], index=False)

    # df = pd.read_csv('little_util.csv', header=0)
    result = df.iloc[:, 1:].copy()
    df_1 = result.applymap(lambda x: int(x * 1.25 * 1804800 / 325))
    little_freq.sort()
    df_mapped = df_1.applymap(
        lambda x: little_freq[np.searchsorted(little_freq, x)] if little_freq[0] <= x <= little_freq[-1] else (
            little_freq[0] if x < little_freq[0] else little_freq[-1]))
    df_mapped['diff_freq'] = (df_mapped.max(axis=1) - df_mapped.min(axis=1)) / df_mapped.max(axis=1)
    df_mapped.to_csv(out_path + 'cal_lit_freq.csv', index=False)

    df = pd.DataFrame(data=big_clu)
    df.columns = ['timestamp', 'cpu4', 'cpu5', 'cpu6']
    # df.to_csv('big_util.csv', header=['timestamp', 'cpu4', 'cpu5', 'cpu6'], index=False)

    # df = pd.read_csv('big_util.csv', header=0)
    result = df.iloc[:, 1:].copy()
    df_1 = result.applymap(
        lambda x: int(x * 1.25 * 2419200 / 828) if x < 2112000 else int(x * 100 * 2419200 / (828 * 95)))
    big_freq.sort()
    df_mapped = df_1.applymap(
        lambda x: big_freq[np.searchsorted(big_freq, x)] if big_freq[0] <= x <= big_freq[-1] else (
            big_freq[0] if x < big_freq[0] else big_freq[-1]))
    df_mapped['diff_freq'] = (df_mapped.max(axis=1) - df_mapped.min(axis=1)) / df_mapped.max(axis=1)
    df_mapped.to_csv(out_path + 'cal_big_freq.csv', index=False)


def del_0_util():
    # df = pd.read_csv('big_util.csv', header=0)
    # result = df.iloc[:, 1:].copy()
    # result = result.loc[~(result == 0).all(axis=1)]
    # df_1 = result.applymap(lambda x: int(x * 1.25 * 2803200 / 1024))
    # big_freq.sort()
    # df_mapped = df_1.applymap(
    #     lambda x: big_freq[np.searchsorted(big_freq, x)] if big_freq[0] <= x <= big_freq[-1] else (
    #         big_freq[0] if x < big_freq[0] else big_freq[-1]))
    # df_mapped.to_csv('big_del0_freq.csv', index=False)

    df = pd.read_csv('big_del0_freq.csv', header=0)
    unique_counts = df.nunique(axis=1)
    equal_ratio = len(unique_counts[unique_counts == 1]) / len(df)
    print('去0\n大核簇各个核计算出的频率全部相等的比例：', "{:.2f}%".format(equal_ratio * 100))
    ratio_values = (df.max(axis=1) - df.min(axis=1)) / df.max(axis=1)
    print('大核簇差值与最大值的比例:')
    print('均值：', "{:.2f}".format(ratio_values.mean()))
    print('方差：', "{:.2f}".format(ratio_values.var()))
    print('最大值：', "{:.2f}".format(ratio_values.max()))

    df = pd.read_csv('cal_big_freq.csv')
    df = df.loc[~(df == 825600).all(axis=1)]
    unique_counts = df.nunique(axis=1)
    equal_ratio = len(unique_counts[unique_counts == 1]) / len(df)
    print('去最小频\n大核簇各个核计算出的频率全部相等的比例：', "{:.2f}%".format(equal_ratio * 100))
    ratio_values = (df.max(axis=1) - df.min(axis=1)) / df.max(axis=1)
    print('大核簇差值与最大值的比例:')
    print('均值：', "{:.2f}".format(ratio_values.mean()))
    print('方差：', "{:.2f}".format(ratio_values.var()))
    print('最大值：', "{:.2f}".format(ratio_values.max()))

    # df = df.loc[~(df.iloc[:, 1:] == 0).all(axis=1)]
    # df.to_csv('big_del0_freq.csv', index=False)


def cal_opp():
    df = pd.read_csv('cal_lit_freq.csv')
    for i, row in enumerate(opp_power(0)):
        df.replace({row[0]: row[1]}, inplace=True)
    # 打印替换后的DataFrame
    df.to_csv('cal_lit_opp.csv', index=False)

    df = pd.read_csv('cal_big_freq.csv')
    for i, row in enumerate(opp_power(4)):
        df.replace({row[0]: row[1]}, inplace=True)
    # 打印替换后的DataFrame
    df.to_csv('cal_big_opp.csv', index=False)


def analyze_freq(in_path: str = '', outpath=''):
    df = pd.read_csv(in_path + 'cal_lit_freq.csv')
    unique_counts = df.nunique(axis=1)
    equal_ratio = len(unique_counts[unique_counts == 1]) / len(df)
    # print('小核簇各个核计算出的频率全部相等的比例：', "{:.2f}%".format(equal_ratio * 100))
    ratio_values = df['diff_freq']
    # x = range(len(ratio_values))
    # plt.plot(x, ratio_values)
    # plt.title("小核簇差值与最大值的比例")
    # plt.show()
    result1 = []
    result1.append(['小核簇各个核计算出的频率全部相等的比例：', "{:.2f}%".format(equal_ratio * 100)])
    result1.append(['小核簇差值与最大值的比例:'])
    result1.append(['均值：', "{:.2f}".format(ratio_values.mean())])
    result1.append(['方差：', "{:.2f}".format(ratio_values.var())])
    result1.append(['最大值：', "{:.2f}".format(ratio_values.max())])
    # print(len(ratio_values))
    # print('小核簇差值与最大值的比例:')
    # print('均值：', "{:.2f}".format(ratio_values.mean()))
    # print('方差：', "{:.2f}".format(ratio_values.var()))
    # print('最大值：', "{:.2f}".format(ratio_values.max()))
    # # 绘制比例的柱状图
    # plt.plot(df.index, ratio_values)
    # plt.ylabel('差值与最大值的比例')
    # plt.title('每一行的差值与最大值的比例')
    # plt.show()

    df = pd.read_csv(in_path + 'cal_big_freq.csv')
    unique_counts = df.nunique(axis=1)
    equal_ratio = len(unique_counts[unique_counts == 1]) / len(df)
    # print('大核簇各个核计算出的频率全部相等的比例：', "{:.2f}%".format(equal_ratio * 100))
    ratio_values = df['diff_freq']
    # print(len(ratio_values))
    # print('大核簇差值与最大值的比例:')
    # print('均值：', "{:.2f}".format(ratio_values.mean()))
    # print('方差：', "{:.2f}".format(ratio_values.var()))
    # print('最大值：', "{:.2f}".format(ratio_values.max()))
    result1.append(['大核簇各个核计算出的频率全部相等的比例：', "{:.2f}%".format(equal_ratio * 100)])
    result1.append(['大核簇差值与最大值的比例:'])
    result1.append(['均值：', "{:.2f}".format(ratio_values.mean())])
    result1.append(['方差：', "{:.2f}".format(ratio_values.var())])
    result1.append(['最大值：', "{:.2f}".format(ratio_values.max())])
    df = pd.DataFrame(result1)
    df.to_csv(outpath + 'result_u_to_f.csv', index=False, header=False)


def analyze_freq1():
    result = []
    df = pd.read_csv('cal_lit_freq.csv')
    unique_counts = df.nunique(axis=1)
    equal_ratio = len(unique_counts[unique_counts == 1]) / len(df)
    result.append([])
    print('小核簇各个核计算出的频率全部相等的比例：', "{:.2f}%".format(equal_ratio * 100))
    ratio_values = (df.max(axis=1) - df.min(axis=1)) / df.max(axis=1)
    print('小核簇差值与最大值的比例:')
    print('均值：', "{:.2f}".format(ratio_values.mean()))
    print('方差：', "{:.2f}".format(ratio_values.var()))
    print('最大值：', "{:.2f}".format(ratio_values.max()))
    # # 绘制比例的柱状图
    # plt.plot(df.index, ratio_values)
    # plt.ylabel('差值与最大值的比例')
    # plt.title('每一行的差值与最大值的比例')
    # plt.show()

    df = pd.read_csv('cal_big_freq.csv')
    unique_counts = df.nunique(axis=1)
    equal_ratio = len(unique_counts[unique_counts == 1]) / len(df)
    print('大核簇各个核计算出的频率全部相等的比例：', "{:.2f}%".format(equal_ratio * 100))
    ratio_values = (df.max(axis=1) - df.min(axis=1)) / df.max(axis=1)
    print('大核簇差值与最大值的比例:')
    print('均值：', "{:.2f}".format(ratio_values.mean()))
    print('方差：', "{:.2f}".format(ratio_values.var()))
    print('最大值：', "{:.2f}".format(ratio_values.max()))


def analyze_opp():
    df = pd.read_csv('cal_lit_opp.csv')
    unique_counts = df.nunique(axis=1)
    equal_ratio = len(unique_counts[unique_counts == 1]) / len(df)
    print('小核簇各个核计算出的功耗全部相等的比例：', "{:.2f}%".format(equal_ratio * 100))
    ratio_values = (df.max(axis=1) - df.min(axis=1)) / df.max(axis=1)
    print('小核簇差值与最大值的比例:')
    print('均值：', "{:.2f}".format(ratio_values.mean()))
    print('方差：', "{:.2f}".format(ratio_values.var()))
    print('最大值：', "{:.2f}".format(ratio_values.max()))
    # # 绘制比例的柱状图
    # plt.plot(df.index, ratio_values)
    # plt.ylabel('差值与最大值的比例')
    # plt.title('每一行的差值与最大值的比例')
    # plt.show()

    df = pd.read_csv('cal_big_opp.csv')
    unique_counts = df.nunique(axis=1)
    equal_ratio = len(unique_counts[unique_counts == 1]) / len(df)
    print('大核簇各个核计算出的功耗全部相等的比例：', "{:.2f}%".format(equal_ratio * 100))
    ratio_values = (df.max(axis=1) - df.min(axis=1)) / df.max(axis=1)
    print('大核簇差值与最大值的比例:')
    print('均值：', "{:.2f}".format(ratio_values.mean()))
    print('方差：', "{:.2f}".format(ratio_values.var()))
    print('最大值：', "{:.2f}".format(ratio_values.max()))


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # print_hi('PyCharm')
    # read_util('trace.txt')
    # read_cluster('trace.txt')
    # anyalyze_clu('cluster_util.csv')
    # count_freq('')

    # analyze_freq()
    # # cal_opp()
    # analyze_opp()
    # del_0_util()
    # get_freq()
    get_freq(sys.argv[1], sys.argv[2])
    # analyze_freq()
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
