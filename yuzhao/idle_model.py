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
def translate_list(matrix):
    matrix = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
    return matrix
def idle_model_process(file_name, app, ii, itemp):
    import codecs

    with codecs.open(r'data_trace/{}/{}_{}_{}/trace.txt'.format(file_name, app, ii, itemp), 'rb') as file:
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
    k = Kernel_Trace_Data_List
    df1 = Kernel_Trace_Data_List[:0]
    for i in range(len(Kernel_Trace_Data_List)):
        # print(i)
        if (k[i][4] == 'pmu_power_model:'):
            df1.append(Kernel_Trace_Data_List[i])

    first_time = 0
    # 温度统计
    total_time = (float(df1[-1][3]) - float(df1[0][3]) + 1) * 10
    # idle0_time,idle1_time,running_time,power
    small_list = [[0] * int(total_time), [0] * int(total_time), [0] * int(total_time), [0] * int(total_time)]
    small_count = [[0] * int(total_time), [0] * int(total_time), [0] * int(total_time), [0] * int(total_time)]
    big_list = [[0] * int(total_time), [0] * int(total_time), [0] * int(total_time), [0] * int(total_time)]
    big_count = [[0] * int(total_time), [0] * int(total_time), [0] * int(total_time), [0] * int(total_time)]
    super_list = [[0] * int(total_time), [0] * int(total_time), [0] * int(total_time), [0] * int(total_time)]
    super_count = [[0] * int(total_time), [0] * int(total_time), [0] * int(total_time), [0] * int(total_time)]
    last_timestamp = [0] * 8
    for x in tqdm(range(0, len(df1))):
        if(x == 0):
            first_time = timestamp = float(df1[x][3])
        for y in range(len(df1[x])):
            timestamp = float(df1[x][3]) - first_time
            if (df1[x][y] == 'cpu'):
                cpu = int(df1[x][y + 1])
            if (df1[x][y] == 'cpufreq'):
                freq = int(df1[x][y + 1])
            if (df1[x][y] == 'cpu_util'):
                util = int(df1[x][y + 1])
            if (df1[x][y] == 'temp'):
                temps = int(df1[x][y + 1])

            if (df1[x][y] == 'p1'):
                p1 = int(df1[x][y + 1])
                p2 = int(df1[x][y + 3])
                p3 = int(df1[x][y + 5])
                p4 = int(df1[x][y + 7])
                p5 = int(df1[x][y + 9])
                p6 = int(df1[x][y + 11])
                p7 = int(df1[x][y + 13])
        delta_time = timestamp - last_timestamp[cpu]
        last_timestamp[cpu] = timestamp
        time_index = int(timestamp * 10)
        if(p1 > 0 and delta_time):
            if (cpu in [0, 1, 2, 3]):
                t1 = p1 - p3
                vol = opp0[freq]
                opp_power = opp0_power[freq]
                curr_cap = freq * 325 / F[0][15]
                temps = temps/1000

                power = (9.936e-14 * t1 + 1.341e-13 * p2 + 1.919e-13 * p3 + 2.724e-12 * p4 + 9.763e-14 * p5) * \
                        vol * vol / delta_time / 4
                static = -(
                    8.937e-06) - 0.0004 * temps + 0.0002 * vol + 0.0023 * temps * vol - 0.0223 * temps * temps + 2.261e-05 * \
                         temps * temps * vol
                # power = power + static
                small_list[3][time_index] = small_list[3][time_index] + power
                small_count[3][time_index] = small_count[3][time_index] + 1

            if (cpu in [4, 5, 6]):
                vol = opp4[freq]
                opp_power = opp4_power[freq]
                curr_cap = freq * 828 / F[1][15]
                t1 = p1 - p6 - p2
                t2 = p6 - p3
                temps = temps/1000
                power = (
                                    1.269e-12 * t1 + 1.914e-13 * p2 + 9.8e-13 * p3 - 1.668e-10 * p4 + 9.486e-14 * p5 +
                                    8.461e-13 * t2 - 5.714e-12 * p7) * vol * vol  / delta_time / 3
                static = 0.0006 + 0.0061 * temps + 0.2607 * vol - 0.0122 * temps * vol - 0.0152 * temps ** 2 + \
                             0.0002 * temps ** 2 * vol
                # power = power + static
                big_list[3][time_index] = big_list[3][time_index] + power
                big_count[3][time_index] = big_count[3][time_index] + 1

            if (cpu in [7]):
                t1 = p1 - p2 - p6
                t2 = p6 - p3
                vol = opp7[freq]
                opp_power = opp7_power[freq]
                curr_cap = freq * 1024 / F[2][18]
                power = (5.932e-13 * t1 + 1.403e-13 * p2 + 4.584e-13 * p3 - 2.524e-10 * p4 + 5.406e-14 * p5 + (
                    3.687e-13) * t2 + 1.561e-12 * p7) * vol * vol / delta_time
                temps = temps/1000
                static = 0.0009 + 0.0099 * temps + 0.3513 * vol - 0.0134 * temps * vol - 0.0257 * temps * temps + 0.0002 * \
                               temps * temps * vol
                # power = power + static
                super_list[3][time_index] = super_list[3][time_index] + power
                super_count[3][time_index] = super_count[3][time_index] + 1

    # 读取idle csv以及runtime csv
    idle_data = pd.read_csv(r'data_process/{}/{}_{}_{}/idle_file.csv'.format(file_name, app, ii, itemp))
    rt_data = pd.read_csv(r'data_process/{}/{}_{}_{}/runtime_file.csv'.format(file_name, app, ii, itemp))
    for i in tqdm(range(len(idle_data)), desc='reading trace file idle'):
        timestamp = idle_data['timestamp'][i] - first_time
        time_index = int(timestamp * 10)
        # print(time_index)
        idle_state = idle_data['state'][i]
        cpu = idle_data['cpu'][i]
        delta_time =  idle_data['delta_time'][i]
        if cpu in [0, 1, 2, 3]:
            if idle_state == 0:
                small_list[0][time_index] = small_list[0][time_index] + delta_time
                small_count[0][time_index] = small_count[0][time_index] + 1
            elif idle_state == 1:
                small_list[1][time_index] = small_list[1][time_index] + delta_time
                small_count[1][time_index] = small_count[1][time_index] + 1
        if cpu in [4, 5, 6]:
            if idle_state == 0:
                big_list[0][time_index] = big_list[0][time_index] + delta_time
                big_count[0][time_index] = big_count[0][time_index] + 1
            elif idle_state == 1:
                big_list[1][time_index] = big_list[1][time_index] + delta_time
                big_count[1][time_index] = big_count[1][time_index] + 1
        if cpu in [7]:
            if idle_state == 0:
                super_list[0][time_index] = super_list[0][time_index] + delta_time
                super_count[0][time_index] = super_count[0][time_index] + 1
            elif idle_state == 1:
                super_list[1][time_index] = super_list[1][time_index] + delta_time
                super_count[1][time_index] = super_count[1][time_index] + 1

    for i in tqdm(range(len(rt_data)), desc='reading trace file runtime'):
        timestamp = rt_data['timestamp'][i] - first_time
        time_index = int(timestamp * 10)
        cpu = rt_data['cpu'][i]
        delta_time = rt_data['delta_time'][i]
        pid = rt_data['pid'][i]
        if pid:
            if cpu in [0, 1, 2, 3]:
                small_list[2][time_index] = small_list[2][time_index] + delta_time
                small_count[2][time_index] = small_count[2][time_index] + 1
            if cpu in [4, 5, 6]:
                big_list[2][time_index] = big_list[2][time_index] + delta_time
                big_count[2][time_index] = big_count[2][time_index] + 1
            if cpu in [7]:
                super_list[2][time_index] = super_list[2][time_index] + delta_time
                super_count[2][time_index] = super_count[2][time_index] + 1


    for i in [3]:
        for j in range(int(total_time)):
            small_list[i][j] = small_list[i][j] / small_count[i][j] if small_count[i][j] > 0 else 1
            big_list[i][j] = big_list[i][j] / big_count[i][j] if big_count[i][j] > 0 else 1
            super_list[i][j] = super_list[i][j] / super_count[i][j] if super_count[i][j] > 0 else 1

    df_data = pd.DataFrame(translate_list(big_list), columns=['idle0', 'idle1', 'rt', 'power'])
    idle_file_path = r'data_process/{}/{}_{}_{}/power_model_file.csv'.format(file_name, app, ii, itemp)
    df_data.to_csv(idle_file_path)
    # print(df_data)
    # 输出皮尔森相关性
    result5 = df_data.corr()
    print(result5)
    # plt.scatter(big_list[1],big_list[3])
    # plt.show()



