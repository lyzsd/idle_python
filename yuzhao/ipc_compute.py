import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
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
def ipc_read_plt(file_name, app, ii, itemp):
    import codecs

    with codecs.open(r'data_trace/{}/{}_{}_{}/pmu_cpu_ipc.txt'.format(file_name, app, ii, itemp), 'rb') as file:
        Kernel_Trace_Data = pd.read_table(file, header=None, error_bad_lines=False, warn_bad_lines=False,
                                          encoding='utf-8')


    Kernel_Trace_Data_List = Kernel_Trace_Data.values.tolist()
    up = 'update_cpu_busy_time:'
    snf = 'sugov_next_freq_shared:'
    sus = 'sugov_update_single:'
    for i in range(len(Kernel_Trace_Data_List)):
        Kernel_Trace_Data_List[i] = Kernel_Trace_Data_List[i][0].split("=")
        # Kernel_Trace_Data_List[i] = Kernel_Trace_Data_List[i].split(",")
    for i in range(len(Kernel_Trace_Data_List)):
        tmp = []
        tmp1 = []
        tmp2 = []
        # print('Kernel_Trace_Data_List', Kernel_Trace_Data_List[i])
        for j in range(len(Kernel_Trace_Data_List[i])):
            if (len(Kernel_Trace_Data_List[i][j].split()) > 3):
                tmp2 = Kernel_Trace_Data_List[i][j].split()
                tmp1 = tmp2[3]
                tmp1 = tmp1[0:-1]
                tmp2[3] = tmp1
                # Kernel_Trace_Data_List[i][j].split()[3] = tmp1
                tmp += tmp2
                # print('Kernel_Trace_Data_List[i][j].split()', Kernel_Trace_Data_List[i][j].split())
                # print(i)
            else:
                tmp += Kernel_Trace_Data_List[i][j].split()
        Kernel_Trace_Data_List[i] = tmp
    Kernel_Trace_Data_List = filter_false(Kernel_Trace_Data_List)
    Kernel_Trace_Data_List = Kernel_Trace_Data_List[11:]
    df1 = Kernel_Trace_Data_List[:0]
    for i in range(len(Kernel_Trace_Data_List)):
        df1.append(Kernel_Trace_Data_List[i])

    small_ipc = []
    big_ipc = []
    super_ipc = []
    small_freq_ipc_list = [[] for _ in range(16)]
    big_freq_ipc_list = [[] for _ in range(16)]
    super_freq_ipc_list = [[] for _ in range(19)]

    #温度统计
    temp_df = pd.read_csv(r'data_process/{}/{}_{}_{}/battery_file.csv'.format(file_name, app, ii, itemp))
    temp_lenght = len(temp_df['temp0'])
    total_lenght = float(df1[-1][3]) - float(df1[0][3])
    for x in tqdm(range(0, len(df1))):
        if(x == 0):
            first_time = timestamp=float(df1[x][3])
        for y in range(len(df1[x])):
            timestamp = float(df1[x][3]) - first_time
            temp_index = find_temp_interval(total_lenght,timestamp,temp_lenght)
            if (df1[x][y] == 'cpu'):
                cpu = int(df1[x][y + 1])
            if (df1[x][y] == 'cpufreq'):
                freq = int(df1[x][y + 1])
            if (df1[x][y] == 'cpu_util'):
                util = int(df1[x][y + 1])
            if (df1[x][y] == 'cycle'):
                p1 = int(df1[x][y + 1])
                p2 = int(df1[x][y + 3])

            # 8print(1)

        if(p1 > 0 and freq and util):
            if (cpu in [0, 1, 2, 3]):
                ipc = p2 / p1
                small_ipc.append(ipc)

                small_freq_ipc_list[F[0].index(freq)].append(ipc)
            if (cpu in [4, 5, 6]):
                ipc = p2 / p1
                big_ipc.append(ipc)
                big_freq_ipc_list[F[1].index(freq)].append(ipc)
            if (cpu in [7]):
                ipc = p2 / p1
                super_ipc.append(ipc)
                super_freq_ipc_list[F[2].index(freq)].append(ipc)

    small_freq_ipc = []
    big_freq_ipc = []
    super_freq_ipc = []
    for i in range(16):
        small_freq_ipc.append(np.mean(small_freq_ipc_list[i]))
        big_freq_ipc.append(np.mean(big_freq_ipc_list[i]))
    for i in range(19):
        super_freq_ipc.append(np.mean(super_freq_ipc_list[i]))

    print(np.mean(small_ipc), np.mean(big_ipc), np.mean(super_ipc))
    print(small_freq_ipc)
    print(big_freq_ipc)
    print(super_freq_ipc)

    # fig, ax = plt.subplots(figsize=(8, 6))
    # # 绘制第一组数据的箱线图和散点图
    # ax.scatter(small_list[2], small_list[0], color='r', alpha=0.5, s=0.1)
    # ax.scatter(small_list[2], small_list[1], color='b', alpha=0.5, s=0.1)
    # plt.xlabel('time(s)')
    # plt.ylabel('power(mw)')
    # plt.show()
    # fig, ax = plt.subplots(figsize=(8, 6))
    # # 绘制第一组数据的箱线图和散点图
    # ax.scatter(big_list[2], big_list[0], color='r', alpha=0.5, s=0.1)
    # ax.scatter(big_list[2], big_list[1], color='b', alpha=0.5, s=0.1)
    #
    # plt.show()
    # fig, ax = plt.subplots(figsize=(8, 6))
    # # 绘制第一组数据的箱线图和散点图
    # ax.scatter(super_list[2], super_list[0], color='r', alpha=0.5, s=0.1)
    # ax.scatter(super_list[2], super_list[1], color='b', alpha=0.5, s=0.1)
    #
    # plt.show()


