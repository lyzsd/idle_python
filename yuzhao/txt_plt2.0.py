import pandas as pd
import numpy as np
from matplotlib import cm
import seaborn as sns
import sys
little_power_data = [
    [300000, 9, 576],
    [403200, 13, 576],
    [499200, 16, 576],
    [595200, 19, 576],
    [691200, 22, 576],
    [806400, 28, 592],
    [902400, 32, 604],
    [998400, 38, 620],
    [1094400, 44, 636],
    [1209600, 52, 656],
    [1305600, 61, 688],
    [1401600, 71, 716],
    [1497600, 83, 748],
    [1612800, 97, 776],
    [1708800, 109, 800],
    [1804800, 122, 824]
]

big_power_data = [
    [710400, 132, 612],
    [844800, 170, 636],
    [960000, 206, 656],
    [1075200, 245, 676],
    [1209600, 282, 684],
    [1324800, 334, 712],
    [1440000, 389, 736],
    [1555200, 448, 760],
    [1670400, 517, 788],
    [1766400, 575, 808],
    [1881600, 649, 832],
    [1996800, 743, 864],
    [2112000, 838, 892],
    [2227200, 940, 920],
    [2342400, 1050, 948],
    [2419200, 1216, 1004]
]

super_power_data = [
    [844800, 259, 636],
    [960000, 302, 644],
    [1075200, 343, 648],
    [1190400, 389, 656],
    [1305600, 432, 660],
    [1420800, 481, 668],
    [1555200, 572, 696],
    [1670400, 643, 712],
    [1785600, 734, 736],
    [1900800, 834, 760],
    [2035200, 950, 784],
    [2150400, 1077, 812],
    [2265600, 1214, 840],
    [2380800, 1362, 868],
    [2496000, 1495, 888],
    [2592000, 1667, 920],
    [2688000, 1805, 940],
    [2764800, 2001, 976],
    [2841600, 2211, 1012]

]

def read_freq_runtime(file_name, app, ii, itemp):
    Kernel_Trace_Data = pd.read_table(r'data_trace/{}/{}_{}_{}/freq_runtime_freq.txt'.format(file_name, app, ii, itemp),
                                      header=None,
                                      error_bad_lines=False)  #
    Kernel_Trace_Data_List = Kernel_Trace_Data.values.tolist()
    for i in range(len(Kernel_Trace_Data_List)):
        Kernel_Trace_Data_List[i] = Kernel_Trace_Data_List[i][0].split()
    power_list = [0] * 4
    for i in range(0, len(Kernel_Trace_Data_List)):
        for j in range(len(Kernel_Trace_Data_List[i])):
            if (i == 0):
                power = little_power_data[0][j]
            elif (i == 1):
                power = big_power_data[1][j]
            else:
                power = super_power_data[2][j]

            power_list[i] = power_list[i] + power * Kernel_Trace_Data_List[i][j]
    power_list[3] = power_list[0] + power_list[1] + power_list[2]
    print(power_list)




