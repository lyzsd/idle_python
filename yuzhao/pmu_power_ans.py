import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm

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

opp7_power = {844800: 221, 960000: 266, 1075200: 306,
              1190400: 356, 1305600: 401, 1420800: 458,
              1555200: 540, 1670400: 614, 1785600: 695,
              1900800: 782, 2035200: 893, 2150400: 1035,
              2265600: 1203, 2380800: 1362, 2496000: 1536,
              2592000: 1725, 2688000: 1898, 2764800: 2017,
              2841600: 2141}

new_small_cost_dy_30_shot = [100, 81, 72, 65, 56, 55, 55, 56, 56, 58, 61, 63, 67, 69, 70, 74]
new_big_cost_dy_30_shot = [292, 309, 326, 343, 361, 383, 396, 433, 452, 475, 505, 548, 579, 630, 667, 716]
new_super_cost_dy_30_shot = [520, 542, 507, 519, 542, 469, 564, 593, 648, 607, 666, 726, 785, 871, 921, 982, 1016, 1100,
                             1074]

new_small_cost_dy_50_shot = [115, 93, 81, 73, 63, 62, 62, 63, 63, 65, 68, 70, 74, 77, 78, 81]
new_big_cost_dy_50_shot = [342, 353, 368, 382, 396, 417, 429, 465, 483, 505, 535, 578, 608, 659, 696, 745]
new_super_cost_dy_50_shot = [674, 681, 633, 634, 648, 569, 661, 686, 740, 697, 754, 814, 873, 960, 1008, 1071, 1104,
                             1188, 1163]

new_small_cost_dy_70_shot = [120, 97, 84, 76, 65, 64, 65, 67, 67, 70, 73, 75, 80, 83, 84, 88]
new_big_cost_dy_70_shot = [489, 483, 487, 493, 497, 514, 521, 554, 569, 589, 617, 658, 688, 737, 773, 823]
new_super_cost_dy_70_shot = [1124, 1087, 998, 970, 957, 859, 938, 952, 998, 948, 998, 1057, 1114, 1200, 1245, 1310,
                             1342, 1424, 1398]

new_small_cost_dy_30_long = [100, 81, 72, 65, 56, 55, 55, 56, 56, 58, 61, 63, 67, 69, 70, 74]
new_big_cost_dy_30_long = [280, 293, 303, 321, 330, 353, 369, 394, 417, 440, 471, 503, 538, 576, 613, 659]
new_super_cost_dy_30_long = [489, 481, 484, 476, 471, 490, 510, 514, 530, 567, 595, 628, 682, 740, 764, 840, 876, 904,
                             937]

new_small_cost_dy_50_long = [115, 93, 81, 73, 63, 62, 62, 63, 63, 65, 68, 70, 74, 77, 78, 81]
new_big_cost_dy_50_long = [329, 338, 344, 359, 366, 387, 402, 426, 448, 470, 501, 532, 567, 605, 642, 688]
new_super_cost_dy_50_long = [642, 620, 610, 592, 578, 591, 607, 607, 621, 656, 682, 716, 770, 828, 851, 929, 964, 992,
                             1025]

new_small_cost_dy_70_long = [120, 97, 84, 76, 65, 64, 65, 67, 67, 70, 73, 75, 80, 83, 84, 88]
new_big_cost_dy_70_long = [476, 467, 463, 471, 467, 483, 494, 514, 534, 554, 583, 613, 646, 683, 719, 766]
new_super_cost_dy_70_long = [1093, 1027, 975, 927, 886, 881, 884, 873, 880, 908, 926, 959, 1012, 1068, 1088, 1167, 1202,
                             1229, 1261]

# basicmath
new_small_cost_math_30_shot = [100, 81, 72, 65, 56, 55, 55, 56, 56, 58, 61, 63, 67, 69, 70, 74]
new_big_cost_math_30_shot = [324, 344, 365, 391, 400, 427, 444, 481, 505, 520, 562, 598, 646, 685, 717, 782]
new_super_cost_math_30_shot = [655, 674, 669, 683, 683, 701, 743, 769, 813, 854, 905, 978, 1051, 1123, 1184, 1285, 1352,
                               1408, 1457]

new_small_cost_math_50_shot = [115, 93, 81, 73, 63, 62, 62, 63, 63, 65, 68, 70, 74, 77, 78, 81]
new_big_cost_math_50_shot = [374, 388, 406, 429, 435, 461, 476, 513, 536, 550, 592, 627, 675, 714, 746, 812]
new_super_cost_math_50_shot = [809, 814, 795, 798, 789, 802, 840, 862, 905, 944, 993, 1066, 1139, 1211, 1271, 1374,
                               1440, 1496, 1545]

new_small_cost_math_70_shot = [120, 97, 84, 76, 65, 64, 65, 67, 67, 70, 73, 75, 80, 83, 84, 88]
new_big_cost_math_70_shot = [521, 518, 525, 541, 536, 557, 568, 602, 622, 634, 674, 708, 754, 792, 823, 889]
new_super_cost_math_70_shot = [1259, 1220, 1160, 1134, 1097, 1092, 1117, 1127, 1164, 1195, 1237, 1309, 1381, 1452, 1508,
                               1613, 1677, 1732, 1780]

new_small_cost_math_30_long = [100, 81, 72, 65, 56, 55, 55, 56, 56, 58, 61, 63, 67, 69, 70, 74]
new_big_cost_math_30_long = [311, 333, 352, 377, 386, 411, 430, 460, 484, 506, 536, 569, 607, 641, 685, 726]
new_super_cost_math_30_long = [616, 647, 643, 658, 658, 675, 712, 741, 784, 821, 864, 928, 996, 1069, 1120, 1207, 1275,
                               1332, 1379]

new_small_cost_math_50_long = [115, 93, 81, 73, 63, 62, 62, 63, 63, 65, 68, 70, 74, 77, 78, 81]
new_big_cost_math_50_long = [361, 377, 393, 416, 422, 445, 463, 492, 515, 536, 566, 599, 637, 670, 714, 755]
new_super_cost_math_50_long = [769, 786, 769, 774, 764, 776, 808, 835, 876, 911, 952, 1016, 1084, 1157, 1208, 1295,
                               1363, 1420, 1468]

new_small_cost_math_70_long = [120, 97, 84, 76, 65, 64, 65, 67, 67, 70, 73, 75, 80, 83, 84, 88]
new_big_cost_math_70_long = [508, 507, 512, 527, 523, 542, 555, 581, 601, 620, 648, 679, 716, 748, 792, 833]
new_super_cost_math_70_long = [1220, 1193, 1134, 1109, 1072, 1066, 1085, 1100, 1134, 1162, 1196, 1259, 1326, 1397, 1444,
                               1534, 1600, 1657, 1703]

# //mem_bench
new_small_cost_mem_30_shot = [100, 81, 72, 65, 56, 55, 55, 56, 56, 58, 61, 63, 67, 69, 70, 74]
new_big_cost_mem_30_shot = [347, 367, 385, 410, 420, 448, 472, 501, 530, 561, 599, 637, 685, 727, 777, 828]
new_super_cost_mem_30_shot = [627, 642, 637, 646, 646, 667, 710, 742, 783, 821, 872, 943, 1016, 1091, 1145, 1243, 1309,
                              1367, 1421]

new_small_cost_mem_50_shot = [115, 93, 81, 73, 63, 62, 62, 63, 63, 65, 68, 70, 74, 77, 78, 81]
new_big_cost_mem_50_shot = [396, 411, 426, 449, 456, 482, 504, 533, 561, 591, 629, 667, 714, 756, 806, 857]
new_super_cost_mem_50_shot = [781, 781, 762, 762, 752, 768, 807, 835, 875, 911, 960, 1031, 1104, 1179, 1232, 1332, 1397,
                              1455, 1509]

new_small_cost_mem_70_shot = [120, 97, 84, 76, 65, 64, 65, 67, 67, 70, 73, 75, 80, 83, 84, 88]
new_big_cost_mem_70_shot = [543, 541, 545, 560, 557, 579, 596, 622, 647, 675, 711, 747, 794, 834, 883, 934]
new_super_cost_mem_70_shot = [1231, 1187, 1128, 1097, 1061, 1058, 1084, 1101, 1133, 1162, 1204, 1274, 1346, 1419, 1469,
                              1571, 1634, 1691, 1744]

new_small_cost_mem_30_long = [100, 81, 72, 65, 56, 55, 55, 56, 56, 58, 61, 63, 67, 69, 70, 74]
new_big_cost_mem_30_long = [294, 305, 319, 339, 343, 366, 390, 411, 433, 454, 473, 517, 544, 572, 614, 652]
new_super_cost_mem_30_long = [534, 546, 536, 532, 545, 574, 574, 618, 628, 661, 695, 750, 799, 866, 912, 998, 1040,
                              1097, 1163]

new_small_cost_mem_50_long = [115, 93, 81, 73, 63, 62, 62, 63, 63, 65, 68, 70, 74, 77, 78, 81]
new_big_cost_mem_50_long = [344, 349, 360, 378, 379, 400, 423, 443, 464, 484, 503, 547, 573, 601, 643, 681]
new_super_cost_mem_50_long = [687, 685, 661, 648, 652, 674, 671, 711, 720, 751, 782, 838, 887, 955, 999, 1087, 1128,
                              1185, 1251]

new_small_cost_mem_70_long = [120, 97, 84, 76, 65, 64, 65, 67, 67, 70, 73, 75, 80, 83, 84, 88]
new_big_cost_mem_70_long = [491, 479, 479, 489, 480, 497, 514, 532, 550, 568, 585, 627, 652, 679, 720, 758]
new_super_cost_mem_70_long = [1138, 1092, 1027, 983, 960, 964, 948, 976, 979, 1002, 1026, 1081, 1129, 1195, 1236, 1326,
                              1366, 1422, 1487]


def filter_false(lst):
    return list(filter(bool, lst))


def find_temp_interval(total_lenght, number, total_intervals):
    # print(total_intervals)
    interval_size = total_lenght / total_intervals
    interval_index = int(number / interval_size)
    return interval_index


def line_correct_rate(data, test_data):
    list_rate = []
    for i in range(len(data)):
        rate = abs(data[i] - test_data[i]) / data[i]
        list_rate.append(rate)
    return np.mean(list_rate)


def pmu_read_plt(file_name, app, ii, itemp):
    import codecs

    with codecs.open(r'data_trace/{}/{}_{}_{}/pmu_power_model.txt'.format(file_name, app, ii, itemp), 'rb') as file:
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

    small_list = [[], [], [], []]
    big_list = [[], [], [], []]
    super_list = [[], [], [], []]
    first_time = 0
    runtime_small = 0
    runtime_big = 0
    runtime_super = 0
    cpu_runtime = [0] * 8

    # 温度统计
    temp_df = pd.read_csv(r'data_process/{}/{}_{}_{}/battery_file.csv'.format(file_name, app, ii, itemp))
    temp_lenght = len(temp_df['temp0'])
    total_lenght = float(df1[-1][3]) - float(df1[0][3])
    for x in tqdm(range(0, len(df1))):
        if (x == 0):
            first_time = timestamp = float(df1[x][3])
        for y in range(len(df1[x])):

            timestamp = float(df1[x][3]) - first_time
            temp_index = find_temp_interval(total_lenght, timestamp, temp_lenght)
            if (df1[x][y] == 'cpu'):
                cpu = int(df1[x][y + 1])
            if (df1[x][y] == 'cpufreq'):
                freq = int(df1[x][y + 1])
            if (df1[x][y] == 'cpu_util'):
                util = int(df1[x][y + 1])

            if (df1[x][y] == 'p1'):
                p1 = int(df1[x][y + 1])
                p2 = int(df1[x][y + 3])
                p3 = int(df1[x][y + 5])
                p4 = int(df1[x][y + 7])
                p5 = int(df1[x][y + 9])
                p6 = int(df1[x][y + 11])
                p7 = int(df1[x][y + 13])

            # 8print(1)

        if (p1 > 0 and freq):
            if (cpu in [0, 1, 2, 3]):
                t1 = p1 - p3
                vol = opp0[freq]
                opp_power = opp0_power[freq]
                curr_cap = freq * 325 / F[0][15]
                temps = temp_df['temp0'][temp_index % temp_lenght] / 1000
                if temps < 40:
                    cost = new_small_cost_mem_30_long[F[0].index(freq)]
                elif 40 <= temps < 60:
                    cost = new_small_cost_mem_50_long[F[0].index(freq)]
                else:
                    cost = new_small_cost_mem_70_long[F[0].index(freq)]

                power = (
                                9.936e-14 * t1 + 1.341e-13 * p2 + 1.919e-13 * p3 + 2.724e-12 * p4 + 9.763e-14 * p5) * vol * vol * freq / p1 * 1000 / 4
                power = (
                                9.936e-14 * t1 + 1.341e-13 * p2 + 1.919e-13 * p3 + 2.724e-12 * p4 + 9.763e-14 * p5) * vol * vol / 4
                static = -(
                    8.937e-06) - 0.0004 * temps + 0.0002 * vol + 0.0023 * temps * vol - 0.0223 * temps * temps + 2.261e-05 * \
                         temps * temps * vol
                static =(-(
                    8.937e-06) - 0.0004 * temps + 0.0002 * vol + 0.0023 * temps * vol - 0.0223 * temps * temps + 2.261e-05 * \
                         temps * temps * vol) * p1 / freq / 1000
                # power=power+static
                if (power > 0):
                    opp_compute_power = util / curr_cap * opp_power
                    dynopp_compute_power = util * cost / 325
                    dynopp_compute_power = cost * freq / F[0][15] * p1 / freq / 1000
                    opp_compute_power = opp_power * p1 / freq / 1000
                    small_list[0].append(power)
                    small_list[1].append(opp_compute_power * 4)
                    small_list[2].append(timestamp)
                    small_list[3].append((static / 4 + power) * 4)
                    runtime_small = runtime_small + p1 / (freq * 1000)

            if (cpu in [4, 5, 6]):
                vol = opp4[freq]
                opp_power = opp4_power[freq]
                curr_cap = freq * 828 / F[1][15]
                t1 = p1 - p6 - p2
                t2 = p6 - p3
                temps = temp_df['temp4'][temp_index % temp_lenght] / 1000
                if temps < 40:
                    cost = new_big_cost_mem_30_long[F[1].index(freq)]
                elif 40 <= temps < 60:
                    cost = new_big_cost_mem_50_long[F[1].index(freq)]
                else:
                    cost = new_big_cost_mem_70_long[F[1].index(freq)]
                power = (
                                1.269e-12 * t1 + 1.914e-13 * p2 + 9.8e-13 * p3 - 1.668e-10 * p4 + 9.486e-14 * p5 +
                                8.461e-13 * t2 - 5.714e-12 * p7) * vol * vol * freq / p1 * 1000 / 3
                power = (
                                1.269e-12 * t1 + 1.914e-13 * p2 + 9.8e-13 * p3 - 1.668e-10 * p4 + 9.486e-14 * p5 +
                                8.461e-13 * t2 - 5.714e-12 * p7) * vol * vol / 3
                static = 0.0006 + 0.0061 * temps + 0.2607 * vol - 0.0122 * temps * vol - 0.0152 * temps ** 2 + \
                         0.0002 * temps ** 2 * vol
                static = (0.0006 + 0.0061 * temps + 0.2607 * vol - 0.0122 * temps * vol - 0.0152 * temps ** 2 + \
                         0.0002 * temps ** 2 * vol) * p1 / freq / 1000
                if (power > 0):
                    opp_compute_power = util / curr_cap * opp_power
                    dynopp_compute_power = util * cost / 828
                    dynopp_compute_power = cost * freq / F[1][15] * p1 / freq / 1000
                    opp_compute_power = opp_power * p1 / freq / 1000
                    big_list[0].append(power)
                    big_list[1].append(opp_compute_power * 3)
                    big_list[2].append(timestamp)
                    big_list[3].append((static / 3 + power) * 3)
                    runtime_big = runtime_big + p1 / (freq * 1000)

            if (cpu in [7]):
                t1 = p1 - p2 - p6
                t2 = p6 - p3
                vol = opp7[freq]
                opp_power = opp7_power[freq]
                curr_cap = freq * 1024 / F[2][18]
                # power = (5.932e-13 * t1 + 1.403e-13 * p2 + 4.584e-13 * p3 - 2.524e-10 * p4 + 5.406e-14 * p5 + (
                #     3.687e-13) * t2 + 1.561e-12 * p7) * vol * vol * freq / p1 * 1000
                power = (5.932e-13 * t1 + 1.403e-13 * p2 + 4.584e-13 * p3 - 2.524e-10 * p4 + 5.406e-14 * p5 + (
                    3.687e-13) * t2 + 1.561e-12 * p7) * vol * vol
                temps = temp_df['temp7'][temp_index % temp_lenght] / 1000
                if temps < 40:
                    cost = new_super_cost_mem_30_long[F[2].index(freq)]
                elif 40 <= temps < 60:
                    cost = new_super_cost_mem_50_long[F[2].index(freq)]
                else:
                    cost = new_super_cost_mem_70_long[F[2].index(freq)]
                # static = (0.0009 + 0.0099 * temps + 0.3513 * vol - 0.0134 * temps * vol - 0.0257 * temps * temps + 0.0002 * \
                #          temps * temps * vol) * p1 / freq / 1000
                static = (
                    0.0009 + 0.0099 * temps + 0.3513 * vol - 0.0134 * temps * vol - 0.0257 * temps * temps + 0.0002 * \
                                     temps * temps * vol) * p1 / freq / 1000
                if (power > 0):
                    # opp_compute_power = opp_power * p1 / freq / 1000
                    dynopp_compute_power = cost * freq / F[2][18] * p1 / freq / 1000
                    opp_compute_power = opp_power * p1 / freq / 1000
                    # dynopp_compute_power = cost * util / 1024
                    super_list[0].append(power)
                    super_list[1].append(opp_compute_power)
                    super_list[2].append(timestamp)
                    super_list[3].append(static + power)
                    runtime_super = runtime_super + p1 / (freq * 1000)
            cpu_runtime[cpu] = cpu_runtime[cpu] + p1 / (freq * 1000)

    print(runtime_small, runtime_big, runtime_super)
    print(cpu_runtime)
    print('cpu总功耗为 小核簇、大核簇、超大核簇、全局')
    total_small = np.sum(small_list[3])
    total_big = np.sum(big_list[3])
    total_super = np.sum(super_list[3])
    print(total_small, total_big, total_super, total_small + total_big + total_super)
    total_small = np.mean(small_list[1])
    total_big = np.mean(big_list[1])
    total_super = np.mean(super_list[1])
    print(total_small, total_big, total_super, total_small + total_big + total_super)
    print("精度")
    print(line_correct_rate(small_list[3], small_list[1]))
    print(line_correct_rate(big_list[3], big_list[1]))
    print(line_correct_rate(super_list[3], super_list[1]))
    # fig, ax = plt.subplots(figsize=(8, 6))
    # # 绘制第一组数据的箱线图和散点图
    # ax.scatter(small_list[2], small_list[3], color='r', alpha=0.5, s=0.1)
    # ax.scatter(small_list[2], small_list[1], color='b', alpha=0.5, s=0.1)
    # plt.xlabel('time(s)')
    # plt.ylabel('power(mw)')
    # plt.show()
    # fig, ax = plt.subplots(figsize=(8, 6))
    # # 绘制第一组数据的箱线图和散点图
    # ax.scatter(big_list[2], big_list[3], color='r', alpha=0.5, s=0.1)
    # ax.scatter(big_list[2], big_list[1], color='b', alpha=0.5, s=0.1)
    #
    # plt.show()
    # fig, ax = plt.subplots(figsize=(8, 6))
    # # 绘制第一组数据的箱线图和散点图
    # ax.scatter(super_list[2], super_list[3], color='r', alpha=0.5, s=0.1)
    # ax.scatter(super_list[2], super_list[1], color='b', alpha=0.5, s=0.1)
    #
    # plt.show()

    # #拟合dyn opp和pmu
    # X = sm.add_constant(small_list[3])
    # # 创建 OLS 模型对象
    #
    # model = sm.OLS(small_list[1], X)
    # # 拟合模型
    # constraints = [{'loc': 'coef', 'type': 'ineq', 'fun': lambda x: x[1:]}, ]
    # results = model.fit(constraints=constraints)
    # # 查看回归结果
    # print(results.summary())
