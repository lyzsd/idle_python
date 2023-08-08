import os.path
import re
import time

import scipy
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
from yuzhao import Constant
F = [
    [300000, 403200, 499200, 595200, 691200, 806400, 902400, 998400, 1094400, 1209600, 1305600, 1401600, 1497600,
     1612800, 1708800, 1804800],
    [710400, 844800, 960000, 1075200, 1209600, 1324800, 1440000, 1555200, 1670400, 1766400, 1881600, 1996800, 2112000,
     2227200, 2342400, 2419200],
    [844800, 960000, 1075200, 1190400, 1305600, 1420800, 1555200, 1670400, 1785600, 1900800, 2035200, 2150400, 2265600,
     2380800, 2496000, 2592000, 2688000, 2764800, 2841600]
]
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
RUNTIME_PATTERN = r'\d+.\d+: sched_stat_runtimes: \d \d+'
IDLE_PATTERN = r'\d+.\d+: csh_cpu_idle: state=\d+ cpu_id=\d'
TASK_MESSAGE_PATTERN = r'\d+.\d+: sched_task_message: cpu_id=\d \w* pid=\d+'
WAKEUP_PATTERN = r'\d+.\d+: sched_wakeup: comm=.*'

STATE0_LATENCY = 43
STATE1_LATENCY = 531
STATE1_LATENCY_BIG = 1061

ENTER_IDLE0 = 0
ENTER_IDLE1 = 1
EXIT_IDLE = 2
WAKEUP = 3


def idle_runtime_wakeup_trace_analysis(file_name, app, ii, temp, exp_time_list=None, runtime_list=None, ):
    need_cal_runtime = False
    # curr_dir = os.getcwd()
    # absolute_path = __file__
    # absolute_path = os.path.abspath(os.path.join(absolute_path, os.pardir))
    # absolute_path = os.path.abspath(os.path.join(absolute_path, os.pardir))
    # absolute_path = os.path.abspath(os.path.join(absolute_path, os.pardir))
    # trace_file_name = os.path.join(absolute_path, 'data_trace', file_name, app + '_' + str(ii) + '_' + str(temp),
    #                                'trace.txt')
    trace_file_name = os.path.join('data_trace', file_name, app + '_' + str(ii) + '_' + str(temp), 'trace.txt')
    trace_file = open(trace_file_name, encoding="utf-8")
    lines = trace_file.readlines()
    stat0_list = []
    stat1_list = []
    if exp_time_list is None:
        exp_time_list = []
    if (runtime_list is None):
        exec_info = {'timestamp': [], 'cpu': [], 'event': [], 'pid': [], 'comm': [], 'freq0': [], 'freq4': [], 'freq7': []}
        runtime_list = []
        need_cal_runtime = True
    consecutive_idle_gap_list = []
    per_cpu_info = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
    cpu_idle_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

    #定义idle统计的csv
    idle_columns = ['timestamp', 'delta_time', 'state', 'cpu']

    list_idle = []

    list_runtime = []
    list_cpu = []
    list_time_stamp = []
    list_pid = []
    list_freq = []

    print('===== idle_runtime_wakeup_trace_analysis =====')
    for line in tqdm(lines[11:], desc='reading trace file'):
        # runtime = pattern_runtime.findall(line)
        s = line.split('cpu_idle:')
        if len(s) != 1:
            timestamp = float(s[0].split()[-1].split(':')[0])
            # state=4294967295 cpu_id=0 freq0=1708800 freq4=1996800 freq7=2592000
            info = s[1].split()
            cpu_number = int(info[1].split('=')[-1])
            per_cpu_info[cpu_number].append((info[0], timestamp))
            cpu_idle_count[cpu_number] += 1
            continue
        s = line.split('sched_wakeup:')
        if len(s) != 1:
            timestamp = float(s[0].split()[-1].split(':')[0])
            cpu_number = int(s[1].split()[-1].split('=')[-1])
            per_cpu_info[cpu_number].append(('wakeup', timestamp))
            continue
        s = line.split('sched_task_message:')
        if need_cal_runtime and len(s) != 1:
            timestamp = float(s[0].split()[-1].split(':')[0])
            # cpu_id=3 enqueue pid=27080 freq0=1708800 freq4=1996800 freq7=2592000 demand=198438 pred_demand=137261 comm='poll_avcD_185'
            info = s[1].split()
            cpu_number = int(info[0].split('=')[-1])
            exec_info['timestamp'].append(timestamp)
            exec_info['cpu'].append(cpu_number)
            exec_info['event'].append(info[1])
            exec_info['pid'].append(int(info[2].split('=')[-1]))
            exec_info['freq0'].append(int(info[3].split('=')[-1]))
            exec_info['freq4'].append(int(info[4].split('=')[-1]))
            exec_info['freq7'].append(int(info[5].split('=')[-1]))
            exec_info['comm'].append(info[8])
            continue


    if need_cal_runtime:
        exec_df = pd.DataFrame(exec_info)
        exec_df['event'].replace('start_exec', 0, inplace=True)
        exec_df['event'].replace('end_exec', 1, inplace=True)
        exec_df['event'].replace('dequeue', 1, inplace=True)
        end_timestamp = exec_df.tail(1)['timestamp'].to_numpy()
        exec_df.drop(exec_df[exec_df['event'] == 'enqueue'].index, inplace=True)

    # value -- list of original trace with respect to cpu
    # key -- cpu number
    # 遍历，获取runtime idle0 idle1 idle_exit_latency 连续的idle中间的差
    print('reading done.')
    print(cpu_idle_count)
    print('idle count:', sum(value for value in cpu_idle_count.values() if isinstance(value, int)))

    for key, value in per_cpu_info.items():
        print('cpu', key, 'total record', len(value))
        idle0 = idle1 = consecutive_idle_gap = 0.0
        idle0_exit_latency = []
        idle1_exit_latency = []
        idle0_enter_count = idle1_enter_count = consecutive_idle_count = exit_with_no_wakeup = 0

        last_event = last_idle_event = -1
        first_event = -1
        # first_timestamp = last_timestamp = float(value[0].split(': ')[0])
        first_timestamp = last_timestamp = value[0][1]
        last_enter_idle_timestamp = 0.0
        if 'state=0' in value[0]:
            last_enter_idle_timestamp = last_timestamp
            first_event = last_event = last_idle_event = 0
        elif 'state=1' in value[0]:
            last_enter_idle_timestamp = last_timestamp
            first_event = last_event = last_idle_event = 1
        elif 'state=4294967295' in value[0]:
            first_event = last_event = 2

        i = 1
        start_time = time.time()

        # Ideal trace:        Real trace1:      Real trace2:
        #  0 enter idle       0 enter idle      0 enter idle
        #  1 wakeup           1 wakeup1         1 exit idle
        #  2 exit idle        2 wakeup2         2 enter idle
        #  3 start exec       3 exit idle       3 exit idle
        for i in tqdm(range(len(value))):
            step = 1
            item = value[i]
            # curr_timestamp = float(item.split(': ')[0])
            curr_timestamp = item[1]
            if 'state=0' in item:
                idle0_enter_count += 1
                last_enter_idle_timestamp = curr_timestamp
                if last_event == 2:  # idle exit->idle enter 连续进入idle
                    consecutive_idle_count += 1
                    consecutive_idle_gap += curr_timestamp - last_timestamp
                last_event = last_idle_event = Constant.ENTER_IDLE0
            elif 'state=1' in item:
                idle1_enter_count += 1
                last_enter_idle_timestamp = curr_timestamp
                if last_event == 2:  # idle exit->idle enter 连续进入idle
                    consecutive_idle_count += 1
                    consecutive_idle_gap += curr_timestamp - last_timestamp
                last_event = last_idle_event = Constant.ENTER_IDLE1
            # 每次退出idle状态，更新
            # 1.对应idle状态时间
            # 2.是否没有唤醒就退出了
            # 3.退出该idle状态的延迟
            elif 'state=4294967295' in item:
                # 上个事件是进入idle0或1，表示没有唤醒就退出了idle
                if last_event == Constant.ENTER_IDLE0:  # idle enter->idle exit
                    idle0 = curr_timestamp - last_enter_idle_timestamp
                    idle_df_to_add = [last_enter_idle_timestamp, idle0, 0, key]
                    list_idle.append(idle_df_to_add)
                #     idle_df = idle_df.append(pd.Series(idle_df_to_add, index=idle_columns), ignore_index=True)
                elif last_event == Constant.ENTER_IDLE1:
                    idle1 = curr_timestamp - last_enter_idle_timestamp
                    idle_df_to_add = [last_enter_idle_timestamp, idle1, 1, key]
                    list_idle.append(idle_df_to_add)
            # i = i + step

        if need_cal_runtime:
            # 用于记录idle列表index
            idle_index = 0
            curr_cpu_exec = exec_df[exec_df['cpu'] == key]
            curr_cpu_exec = curr_cpu_exec.sort_values(by=['pid', 'timestamp']).reset_index(drop=True)
            # prev_pid = curr_cpu_exec['pid'][0]
            for i in tqdm(range(len(curr_cpu_exec))):
                pid = curr_cpu_exec['pid'][i]
                # print(pid)
                next_index = min(i + 1, len(curr_cpu_exec) - 1)
                next_pid = curr_cpu_exec['pid'][next_index]
                curr_timestamp = curr_cpu_exec['timestamp'][i]
                if key <= 3:
                    freq = curr_cpu_exec['freq0'][i]
                elif 7 > key > 3:
                    freq = curr_cpu_exec['freq4'][i]
                else:
                    freq = curr_cpu_exec['freq7'][i]
                next_timestamp = curr_cpu_exec['timestamp'][next_index]
                if curr_cpu_exec['event'][i] == 0 and next_pid == pid:
                    if pid != 1:
                        runtime = next_timestamp - curr_timestamp
                    else:
                        runtime = next_timestamp - curr_timestamp - list_idle[idle_index][1]
                        idle_index = idle_index + 1
                    list_cpu.append(key)
                    list_pid.append(pid)
                    list_runtime.append(runtime)
                    list_time_stamp.append(next_timestamp)
                    list_freq.append(freq)



    # 将列表转换成DataFrame
    idle_df = pd.DataFrame(list_idle, columns=idle_columns)
    runtime_df = pd.DataFrame(
        {'timestamp': list_time_stamp, 'delta_time': list_runtime, 'cpu': list_cpu,
         'pid': list_pid,'freq' : list_freq})
    runtime_df = runtime_df.sort_values(by='timestamp')
    runtime_df = runtime_df.reset_index(drop=True)
    idle_df = idle_df.sort_values(by='timestamp')
    idle_df = idle_df.reset_index(drop=True)
    rt_file_path = r'data_process/{}/{}_{}_{}/runtime_file.csv'.format(file_name, app, ii, temp)
    idle_file_path = r'data_process/{}/{}_{}_{}/idle_file.csv'.format(file_name, app, ii, temp)
    idle_df.to_csv(idle_file_path)
    runtime_df.to_csv(rt_file_path)


def ddr_trace_read(file_name, app, ii, temp):
    trace_file_name = os.path.join('data_trace', file_name, app + '_' + str(ii) + '_' + str(temp), 'ddr_freq.txt')
    trace_file = open(trace_file_name, encoding="utf-8")
    lines = trace_file.readlines()
    ddr_label = [312, 365, 477, 585, 643, 982, 1302, 1467, 1827, 2216, 5000]
    ddr_label_new = [0] * 11
    for i in range(len(ddr_label)):
        ddr_label_new[10 - i] = int(10e6 / ddr_label[i])
    ddr_freq_list = [0] * 11
    for line in tqdm(lines[11:], desc='reading trace file'):
        # runtime = pattern_runtime.findall(line)
        s = line.split('ddr_freq:')
        if len(s) != 1:
            timestamp = float(s[0].split()[-1].split(':')[0])
            # state=4294967295 cpu_id=0 freq0=1708800 freq4=1996800 freq7=2592000
            info = s[1].split()
            ddr_freq = int(10e6 / (int(info[1].split('=')[-1])))
            ddr_index = ddr_label_new.index(ddr_freq)
            ddr_freq_list[ddr_index] = ddr_freq_list[ddr_index] + 1
            continue
    print(ddr_freq_list)


def idle_runtime_read(file_name, app, ii, itemp):
    idle_data = pd.read_csv(r'data_process/{}/{}_{}_{}/idle_file.csv'.format(file_name, app, ii, itemp))
    rt_data = pd.read_csv(r'data_process/{}/{}_{}_{}/runtime_file.csv'.format(file_name, app, ii, itemp))
    # for i in tqdm(range(len(idle_data)), desc='reading trace file idle'):
    #     timestamp = idle_data['timestamp'][i] - first_time
    #     time_index = int(timestamp * 10)
    #     # print(time_index)
    #     idle_state = idle_data['state'][i]
    #     cpu = idle_data['cpu'][i]
    #     delta_time = idle_data['delta_time'][i]
    #     if cpu in [0, 1, 2, 3]:
    #         if idle_state == 0:
    #             small_list[0][time_index] = small_list[0][time_index] + delta_time
    #         elif idle_state == 1:
    #             small_list[1][time_index] = small_list[1][time_index] + delta_time
    #     if cpu in [4, 5, 6]:
    #         if idle_state == 0:
    #             big_list[0][time_index] = big_list[0][time_index] + delta_time
    #         elif idle_state == 1:
    #             big_list[1][time_index] = big_list[1][time_index] + delta_time
    #     if cpu in [7]:
    #         if idle_state == 0:
    #             super_list[0][time_index] = super_list[0][time_index] + delta_time
    #         elif idle_state == 1:
    #             super_list[1][time_index] = super_list[1][time_index] + delta_time
    small_list = [0] * 16
    big_list = [0] * 16
    super_list = [0] * 19
    for i in tqdm(range(len(rt_data)), desc='reading trace file runtime'):

        cpu = rt_data['cpu'][i]
        delta_time = rt_data['delta_time'][i]
        pid = rt_data['pid'][i]
        freq = rt_data['freq'][i]
        if cpu in [0, 1, 2, 3]:
            small_list[F[0].index(freq)] = small_list[F[0].index(freq)] + delta_time
        if cpu in [4, 5, 6]:
            big_list[F[1].index(freq)] = big_list[F[1].index(freq)] + delta_time
        if cpu in [7]:
            super_list[F[2].index(freq)] = super_list[F[2].index(freq)] + delta_time
    print(small_list)
    print(big_list)
    print(super_list)
    small_energy = 0
    big_energy = 0
    super_energy = 0
    for i in range(16):
        power0 = opp0_power[F[0][i]]
        energy0 = power0 * small_list[i]
        small_energy = small_energy + energy0
        power4 = opp4_power[F[1][i]]
        energy4 = power4 * big_list[i]
        big_energy = big_energy + energy4
    for i in range(19):
        power7 = opp7_power[F[2][i]]
        energy7 = power7 * super_list[i]
        super_energy = super_energy + energy7

    f = open(r'data_trace/{}/{}_{}_{}/freq_running_time.txt'.format(file_name, app, ii, itemp), 'w')
    f.close()
    f = open(r'data_trace/{}/{}_{}_{}/freq_running_time.txt'.format(file_name, app, ii, itemp), 'a')
    print(*small_list, file=f)
    print(*big_list, file=f)
    print(*super_list, file=f)
    f.close()
    print('小核簇能量、大核簇能量、超大核簇能量、总能量')
    print(small_energy,big_energy,super_energy,small_energy + big_energy + super_energy)




