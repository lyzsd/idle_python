import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import pandas as pd
import numpy as np
F = [
    [300000, 403200, 499200, 595200, 691200, 806400, 902400, 998400, 1094400, 1209600, 1305600, 1401600, 1497600,
     1612800, 1708800, 1804800],
    [710400, 844800, 960000, 1075200, 1209600, 1324800, 1440000, 1555200, 1670400, 1766400, 1881600, 1996800, 2112000,
     2227200, 2342400, 2419200],
    [844800, 960000, 1075200, 1190400, 1305600, 1420800, 1555200, 1670400, 1785600, 1900800, 2035200, 2150400, 2265600,
     2380800, 2496000, 2592000, 2688000, 2764800, 2841600]
]
def filter_false(lst):
    return list(filter(bool, lst))

def read_sugov_next_freq(file_name, app, ii, itemp):
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
        for j in range(len(Kernel_Trace_Data_List[i])):
            if (len(Kernel_Trace_Data_List[i][j].split()) > 3):
                tmp2 = Kernel_Trace_Data_List[i][j].split()
                tmp1 = tmp2[3]
                tmp1 = tmp1[0:-1]
                tmp2[3] = tmp1
                tmp += tmp2

            else:
                tmp += Kernel_Trace_Data_List[i][j].split()
        Kernel_Trace_Data_List[i] = tmp

    Kernel_Trace_Data_List = filter_false(Kernel_Trace_Data_List)


    Kernel_Trace_Data_List = Kernel_Trace_Data_List[11:]
    k = Kernel_Trace_Data_List
    # print(k)
    df1 = Kernel_Trace_Data_List[:0]
    small_freq_list = [0] * 16
    big_freq_list = [0] * 16
    super_freq_list = [0] * 19
    diff_count = 0
    total_count = 0
    for i in range(len(Kernel_Trace_Data_List)):
        if (k[i][4] == 'sugov_next_freq:'):
            df1.append(Kernel_Trace_Data_List[i])
    for x in tqdm(range(0, len(df1))):
        total_count = total_count + 1
        for y in range(len(df1[x])):
            if (df1[x][y] == 'cpu'):
                cpu = int(df1[x][y + 1])
            if (df1[x][y] == 'freq'):
                freq = int(df1[x][y + 1])
            if (df1[x][y] == 'req_freq'):
                req_freq = int(df1[x][y + 1])
                # freq = int(df1[x][y + 1])
        if(req_freq != freq):
            diff_count = diff_count + 1
        if(cpu == 0):
            small_freq_list[F[0].index(freq)] = small_freq_list[F[0].index(freq)] + 1
        if(cpu == 4):
            big_freq_list[F[1].index(freq)] = big_freq_list[F[1].index(freq)] + 1
        if(cpu == 7):
            super_freq_list[F[2].index(freq)] = super_freq_list[F[2].index(freq)] + 1
    sum_runtime = np.sum(small_freq_list) + np.sum(big_freq_list) + np.sum(super_freq_list)
    for i in range(16):
        small_freq_list[i] = small_freq_list[i] / sum_runtime
        big_freq_list[i] = big_freq_list[i] / sum_runtime
    for i in range(19):
        super_freq_list[i] = super_freq_list[i] / sum_runtime
    print(diff_count,total_count)
    return small_freq_list, big_freq_list, super_freq_list

def read_cfs_cpu_selection(file_name, app, ii, itemp):
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
        for j in range(len(Kernel_Trace_Data_List[i])):
            if (len(Kernel_Trace_Data_List[i][j].split()) > 3):
                tmp2 = Kernel_Trace_Data_List[i][j].split()
                tmp1 = tmp2[3]
                tmp1 = tmp1[0:-1]
                tmp2[3] = tmp1
                tmp += tmp2

            else:
                tmp += Kernel_Trace_Data_List[i][j].split()
        Kernel_Trace_Data_List[i] = tmp

    Kernel_Trace_Data_List = filter_false(Kernel_Trace_Data_List)


    Kernel_Trace_Data_List = Kernel_Trace_Data_List[11:]
    k = Kernel_Trace_Data_List
    # print(k)
    df1 = Kernel_Trace_Data_List[:0]
    cpu_count_list = [0] * 8

    for i in range(len(Kernel_Trace_Data_List)):
        if (k[i][4] == 'sched_cpu_selection:'):
            df1.append(Kernel_Trace_Data_List[i])
    for x in tqdm(range(0, len(df1))):
        for y in range(len(df1[x])):
            if (df1[x][y] == 'target_cpu'):
                cpu = int(df1[x][y + 1])
        cpu_count_list[cpu] = cpu_count_list[cpu] + 1
    print("选核分布")
    print(cpu_count_list)


