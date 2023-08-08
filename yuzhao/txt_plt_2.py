import pandas as pd
import numpy as np
from matplotlib import cm
import seaborn as sns
import sys
import matplotlib.pyplot as plt  # matplotlib数据可视化神器
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
F = [
    [300000, 403200, 499200, 595200, 691200, 806400, 902400, 998400, 1094400, 1209600, 1305600, 1401600, 1497600,
     1612800, 1708800, 1804800],
    [710400, 844800, 960000, 1075200, 1209600, 1324800, 1440000, 1555200, 1670400, 1766400, 1881600, 1996800, 2112000,
     2227200, 2342400, 2419200],
    [844800, 960000, 1075200, 1190400, 1305600, 1420800, 1555200, 1670400, 1785600, 1900800, 2035200, 2150400, 2265600,
     2380800, 2496000, 2592000, 2688000, 2764800, 2841600]
]
small_cap = [0] * 16
big_cap = [0] * 16
super_cap = [0] * 19
for i in range(16):
    small_cap[i] = F[0][i] * 325 / F[0][15]
    big_cap[i] = F[1][i] * 828 / F[1][15]
for i in range(19):
    super_cap[i] = F[2][i] * 1024 / F[2][18]
def read_fps(file_name, app, ii, itemp):
    Kernel_Trace_Data = pd.read_table(r'data_trace/{}/{}_{}_{}/FPS.txt'.format(file_name, app, ii, itemp),
                                      header=None,
                                      error_bad_lines=False)  #
    # print(Kernel_Trace_Data)
    # Kernel_Trace_Data_List=Kernel_Trace_Data
    Kernel_Trace_Data_List = Kernel_Trace_Data.values.tolist()
    #print(Kernel_Trace_Data_List)
    # print(Kernel_Trace_Data_List)
    Total=0
    jank=0
    for i in range(len(Kernel_Trace_Data_List)):
        Kernel_Trace_Data_List[i] = Kernel_Trace_Data_List[i][0].split()
    print(Kernel_Trace_Data_List)
    for i in range(len(Kernel_Trace_Data_List)):
        #print(Kernel_Trace_Data_List[i][0])
        # Kernel_Trace_Data_List[i][4]=float(Kernel_Trace_Data_List[i][4])*1000000
        if (Kernel_Trace_Data_List[i][0] == 'Total' and Kernel_Trace_Data_List[i][1] == 'frames'):
            Total=int(Kernel_Trace_Data_List[i][3])
            #u=pd.DataFrame(Kernel_Trace_Data_List[i][2])
            #print(Kernel_Trace_Data_List[i][0])
        if (Kernel_Trace_Data_List[i][0] == 'Janky'and Kernel_Trace_Data_List[i][1] == 'frames:'):
            jank = int(Kernel_Trace_Data_List[i][2])
            print(jank)
    return jank/Total
def battery_read(file_name, app, ii, itemp):
    data1 = pd.read_csv(r'data_process/{}/{}_{}_{}/battery_file.csv'.format(file_name, app, ii, itemp))
    power = data1['power']
    temp0 = data1['temp0']
    temp4 = data1['temp4']
    temp7 = data1['temp7']

    return np.mean(list(power)),np.mean(list(temp0)),np.mean(list(temp4)),np.mean(list(temp7))
def pltscatter_freq(l1,l2,l3,y):
    plt.figure(figsize=(16, 10))
    colors = ['blue','red','blue','red','blue','red','red','red','red','red']
    for i in range(len(l1)):

        plt.plot(small_cap, l1[i], marker='o', markerfacecolor='white', linestyle='-',color='green', label='small_{}'.format(i))
    for i in range(len(l2)):

        plt.plot(big_cap, l2[i], marker='x', markerfacecolor='white', linestyle='-',color='blue', label='big_{}'.format(i))
    for i in range(len(l3)):
        plt.plot(super_cap, l3[i], marker='o', markerfacecolor='white', linestyle='--',color='red', label='super_{}'.format(i))
    plt.legend()
    plt.xlabel('capacity')
    plt.ylabel(y)

    plt.show()
def read_freq_runtime(file_name, app, ii, itemp):

    Kernel_Trace_Data = pd.read_table(r'data_trace/{}/{}_{}_{}/freq_running_time.txt'.format(file_name, app, ii, itemp),
                                          header=None,
                                          error_bad_lines=False)  #
    Kernel_Trace_Data_List = Kernel_Trace_Data.values.tolist()
    small_runtime_list = [0] * 16
    big_runtime_list = [0] * 16
    super_runtime_list = [0] * 19
    for i in range(len(Kernel_Trace_Data_List)):
        Kernel_Trace_Data_List[i] = Kernel_Trace_Data_List[i][0].split()
    power_list = [0]*4
    for i in range(0, len(Kernel_Trace_Data_List)):
        for j in range(len(Kernel_Trace_Data_List[i])):
            Kernel_Trace_Data_List[i][j] = float(Kernel_Trace_Data_List[i][j])
            if(i == 0):
                power = little_power_data[j][1]
                small_runtime_list[j] = Kernel_Trace_Data_List[i][j]
            elif(i == 1):
                power = big_power_data[j][1]
                big_runtime_list[j] = Kernel_Trace_Data_List[i][j]
            else:
                power = super_power_data[j][1]
                super_runtime_list[j] = Kernel_Trace_Data_List[i][j]

            power_list[i] = power_list[i] + power * Kernel_Trace_Data_List[i][j]
    power_list[3] = power_list[0] + power_list[1] + power_list[2]
    print(power_list)
    sum_runtime = np.sum(small_runtime_list) + np.sum(big_runtime_list) + np.sum(super_runtime_list)
    for i in range(16):
        small_runtime_list[i] = small_runtime_list[i]/sum_runtime
        big_runtime_list[i] = big_runtime_list[i]/sum_runtime
    for i in range(19):
        super_runtime_list[i] = super_runtime_list[i]/sum_runtime
    return small_runtime_list,big_runtime_list,super_runtime_list

def read_sys_freq_runtime(file_name, app, ii, itemp):

    Kernel_Trace_Data = pd.read_table(r'data_trace/{}/{}_{}_{}/freq-running-time.txt'.format(file_name, app, ii, itemp),
                                          header=None,
                                          error_bad_lines=False)  #
    Kernel_Trace_Data_List = Kernel_Trace_Data.values.tolist()
    small_runtime_list = [0] * 16
    big_runtime_list = [0] * 16
    super_runtime_list = [0] * 19
    for i in range(len(Kernel_Trace_Data_List)):
        Kernel_Trace_Data_List[i] = Kernel_Trace_Data_List[i][0].split()
    power_list = [0] * 4
    cluster = 0
    for i in range(0, len(Kernel_Trace_Data_List)):
        # print(Kernel_Trace_Data_List[i][0])
        if(Kernel_Trace_Data_List[i][0] == '----------------') :

            if(cluster == 0):
                x = 0
                for j in range(i + 1, i + 17):
                    small_runtime_list[x] = int(Kernel_Trace_Data_List[j][1])
                    x = x + 1
            if(cluster == 1):
                x = 0
                for j in range(i + 1, i + 17):
                    big_runtime_list[x] = int(Kernel_Trace_Data_List[j][1])
                    x = x + 1
            if (cluster == 2):
                x = 0
                for j in range(i + 1, i + 20):
                    super_runtime_list[x] = int(Kernel_Trace_Data_List[j][1])
                    x = x + 1

            cluster = cluster + 1

    sum_runtime = np.sum(small_runtime_list) + np.sum(big_runtime_list) + np.sum(super_runtime_list)
    for i in range(16):
        small_runtime_list[i] = small_runtime_list[i]/sum_runtime
        big_runtime_list[i] = big_runtime_list[i]/sum_runtime
    for i in range(19):
        super_runtime_list[i] = super_runtime_list[i]/sum_runtime
    return small_runtime_list,big_runtime_list,super_runtime_list





