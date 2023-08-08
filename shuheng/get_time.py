import csv
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
from pylab import *
from sklearn.ensemble import IsolationForest
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import time
F = [
    [300000, 403200, 499200, 595200, 691200, 806400, 902400, 998400, 1094400, 1209600, 1305600, 1401600, 1497600,
     1612800, 1708800, 1804800],
    [710400, 844800, 960000, 1075200, 1209600, 1324800, 1440000, 1555200, 1670400, 1766400, 1881600, 1996800, 2112000,
     2227200, 2342400, 2419200],
    [844800, 960000, 1075200, 1190400, 1305600, 1420800, 1555200, 1670400, 1785600, 1900800, 2035200, 2150400, 2265600,
     2380800, 2496000, 2592000, 2688000, 2764800, 2841600]
]

mpl.rcParams['font.sans-serif'] = ['SimHei']
sns.set_style('whitegrid', {'font.sans-serif': ['simhei', 'FangSong']})
matplotlib.rcParams['axes.unicode_minus'] = False

def compute_freq_running_time(running_time, curr_cpu, curr_freq, small_list, big_list, super_list):
    if (curr_cpu in [0, 1, 2, 3]):
        idx = F[0].index(curr_freq)
        small_list[idx] = small_list[idx] + running_time
    elif (curr_cpu in [4, 5, 6]):
        idx = F[1].index(curr_freq)
        big_list[idx] = big_list[idx] + running_time
    else:
        idx = F[2].index(curr_freq)
        super_list[idx] = super_list[idx] + running_time


def get_task_time(file_name, app, ii, itemp):
    runtime_list = []
    able_list = []

    trace_file_name = os.path.join('data_trace', file_name, app + '_' + str(ii) + '_' + str(itemp), 'trace.txt')
    trace_file = open(trace_file_name, encoding="utf-8")
    lines = trace_file.readlines()

    time_list = []
    cpu_list = []
    state_list = []
    pid_list = []
    freq0_list = []
    freq4_list = []
    freq7_list = []
    demand_list = []
    pred_demand_list = []
    comm_list = []

    pattern_tmsg = r"(\d+\.\d+):\s+sched_task_message:\s+cpu_id=(\S+)\s+(\S+)\s+pid=(\d+)\s+freq0=(\d+)\s+freq4=(\d+)\s+freq7=(\d+)\s+demand=(\d+)\s+pred_demand=(\d+)\s+comm='(.+)'"
    pattern_idle = r"(\d+\.\d+):\s+cpu_idle:\s+state=(\d+)\s+cpu_id=(\d+)"
    for line in tqdm(lines):
        tmsg = re.search(pattern_tmsg, line)
        idle = re.search(pattern_idle, line)
        if tmsg:
            time_list.append(float(tmsg.group(1)))
            cpu_list.append(int(tmsg.group(2)))
            state_list.append(tmsg.group(3))
            pid_list.append(int(tmsg.group(4)))
            freq0_list.append(int(tmsg.group(5)))
            freq4_list.append(int(tmsg.group(6)))
            freq7_list.append(int(tmsg.group(7)))
            demand_list.append(int(tmsg.group(8)))
            pred_demand_list.append(int(tmsg.group(9)))
            comm_list.append(tmsg.group(10))
        if idle:
            time_list.append(float(idle.group(1)))
            state_list.append(idle.group(2))
            cpu_list.append(int(idle.group(3)))
            pid_list.append(0)
            freq0_list.append('')
            freq4_list.append('')
            freq7_list.append('')
            demand_list.append('')
            pred_demand_list.append('')
            comm_list.append('')
    # for line in tqdm(lines)
    df = pd.DataFrame(
        {'time': time_list, 'cpu': cpu_list, 'state': state_list, 'pid': pid_list, 'comm': comm_list,
         'freq0': freq0_list, 'freq4': freq4_list, 'freq7': freq7_list, 'demand': demand_list,
         'pred_demand': pred_demand_list})
    df.to_csv(f'data_process/{file_name}/{app}_{ii}_{itemp}/dataframe.csv', index=False)

    group = df.groupby(['cpu', 'pid'])

    begin_time = df.iloc[0]['time']
    final_time = df.iloc[-1]['time']
    total_time = final_time - begin_time
    print('total time:', total_time)

    comm_list = []
    pid_list = []
    runnable_list = []
    running_list = []
    idle_list = []

    cpu_running_sum = 0
    cpu_runnable_sum = 0
    cpu_idle_sum = 0
    cpu = 0

    #用于统计频点分布
    small_list=[0 for i in range(16)]
    big_list=[0 for i in range(16)]
    super_list=[0 for i in range(19)]
    for key, value in tqdm(group):
        if (cpu != key[0]):
            print('')
            print(f'cpu{cpu} idle time:', cpu_idle_sum)
            print(f'cpu{cpu} running time:', cpu_running_sum)
            print(f'cpu{cpu} runnable time:', cpu_runnable_sum)
            print(f'cpu{cpu} idle + running time:', cpu_idle_sum + cpu_running_sum)
            runtime_list.append(cpu_running_sum)
            able_list.append(cpu_runnable_sum)
            rows = zip(running_list, runnable_list, idle_list, pid_list, comm_list)
            with open(f'data_process/{file_name}/{app}_{ii}_{itemp}/cpu{cpu}_ing_able_result.csv', 'w',
                      newline="") as f:
                writer = csv.writer(f)
                writer.writerow(['running_time', 'runnable_time', 'idle_time', 'pid', 'comms'])
                for row in rows:
                    writer.writerow(row)
            comm_list = []
            pid_list = []
            runnable_list = []
            running_list = []
            idle_list = []
            cpu_running_sum = 0
            cpu_runnable_sum = 0
            cpu_idle_sum = 0
            cpu += 1
        # if (cpu != key[0])
        running_count = 0
        runnable_count = 0
        idle_max = 0
        running_max = 0
        runnable_max = 0
        idle_max_time = 0
        running_max_time = 0
        runnable_max_time = 0
        idle_time = 0
        runnable_time = 0
        running_time = 0
        if (value.iloc[0]['pid'] != 0):
            enqueue_temp = 0
            enqueue_time = 0
            start_temp = 0
            start_time = 0
            dequeue_temp = 0
            dequeue_time = 0
            end_temp = 0
            end_time = 0
            # 1. enqueue -> dequeue
            # 2. enqueue -> start -> end -> start -> dequeue
            # 3. start -> end -> start -> end
            # 4. enqueue -> start -> dequeue -> enqueue -> end -> start -> dequeue
            # 5. enqueue -> start -> enqueue -> dequeue -> start -> dequeue
            for i in range(len(value)):
                #用于统计当前cpu当前频点的运行时间
                curr_running_time = 0
                curr_cpu = value.iloc[i]['cpu']
                curr_freq = 0
                if(curr_cpu in [0 ,1 ,2 ,3] ):
                    curr_freq = value.iloc[i]['freq0']
                elif(curr_cpu in [4, 5, 6]):
                    curr_freq = value.iloc[i]['freq4']
                else:
                    curr_freq = value.iloc[i]['freq7']


                if (value.iloc[i]['state'] == 'enqueue'):
                    enqueue_temp = 1
                    enqueue_time = value.iloc[i]['time']
                    if (start_temp == 1 and dequeue_temp != 1):
                        enqueue_temp = 2
                    dequeue_temp = 0
                    end_temp = 0
                elif (value.iloc[i]['state'] == 'start_exec'):
                    running_count += 1
                    start_temp = 1
                    start_time = value.iloc[i]['time']
                    if (enqueue_temp == 0):
                        runnable_time += start_time - begin_time
                    elif (enqueue_temp == 1 and end_temp == 0):
                        runnable_time += start_time - enqueue_time
                        enqueue_temp = 1
                    elif (end_temp == 1):
                        runnable_time += start_time - end_time
                    end_temp = 0
                elif (value.iloc[i]['state'] == 'dequeue'):
                    dequeue_temp = 1
                    dequeue_time = value.iloc[i]['time']
                    enqueue_temp -= 1
                    if (enqueue_temp == -1):
                        if (start_temp == 1):
                            running_time += dequeue_time - start_time
                            curr_running_time = dequeue_time - start_time
                        elif (end_temp == 1):
                            runnable_time += dequeue_time - end_time
                        else:
                            continue
                    elif (enqueue_temp == 1):
                        running_time += dequeue_time - start_time
                        curr_running_time = dequeue_time - start_time
                    elif (start_temp == 0):
                        runnable_time += dequeue_time - enqueue_time
                    elif (start_temp == 1):
                        time_temp = max(start_time, enqueue_time)
                        running_time += dequeue_time - time_temp
                        curr_running_time = dequeue_time - time_temp
                    elif (end_temp == 1):
                        runnable_time += dequeue_time - end_time
                elif (value.iloc[i]['state'] == 'end_exec'):
                    runnable_count += 1
                    end_temp = 1
                    end_time = value.iloc[i]['time']
                    if (start_temp == 1):
                        running_time += end_time - start_time
                        curr_running_time = end_time - start_time
                    elif (enqueue_temp == 1):
                        enqueue_time = end_time
                    start_temp = 0
                if(curr_running_time) :
                    compute_freq_running_time(curr_running_time, curr_cpu, curr_freq, small_list, big_list, super_list)
        elif (value.iloc[0]['pid'] == 0):
            start_time = 0
            start_temp = 0
            end_time = 0
            end_temp = 0
            idle_b_time = 0
            idle_b_temp = 0
            idle_e_time = 0
            idle_e_temp = 0

            for i in range(len(value)):
                curr_running_time = 0
                curr_cpu = value.iloc[i]['cpu']
                curr_freq = 0
                if (curr_cpu in [0, 1, 2, 3]):
                    curr_freq = value.iloc[i]['freq0']
                elif (curr_cpu in [4, 5, 6]):
                    curr_freq = value.iloc[i]['freq4']
                else:
                    curr_freq = value.iloc[i]['freq7']

                if (value.iloc[i]['state'] == 'start_exec'):
                    runnable_count += 1
                    start_temp = 1
                    start_time = value.iloc[i]['time']
                    if (end_temp == 1):
                        runnable_time += start_time - end_time
                    end_temp = 0
                elif (value.iloc[i]['state'] == '0' or value.iloc[i]['state'] == '1'):
                    running_count += 1
                    idle_b_temp = 1
                    idle_b_time = value.iloc[i]['time']
                    if (idle_e_temp == 1):
                        running_time += idle_b_time - idle_e_time
                        curr_running_time = idle_b_time - idle_e_time
                    else:
                        running_time += idle_b_time - start_time
                        curr_running_time = idle_b_time - start_time
                    idle_e_temp = 0
                elif (value.iloc[i]['state'] == '4294967295' or value.iloc[i]['state'] == '-1'):
                    idle_e_temp = 1
                    idle_e_time = value.iloc[i]['time']
                    if (idle_b_temp == 0): continue
                    idle_time += idle_e_time - idle_b_time
                    idle_b_temp = 0
                elif (value.iloc[i]['state'] == 'end_exec'):
                    running_count += 1
                    end_temp = 1
                    end_time = value.iloc[i]['time']
                    if (start_temp == 0):
                        running_time += end_time - begin_time
                        curr_running_time = end_time - begin_time
                    elif (idle_e_temp == 0):
                        running_time += end_time - start_time
                        curr_running_time = end_time - start_time
                    else:
                        running_time += end_time - idle_e_time
                        curr_running_time = end_time - idle_e_time
                    start_temp = 0
                    idle_e_temp = 0
                if (curr_running_time and curr_freq != '' ):
                    compute_freq_running_time(curr_running_time, curr_cpu, curr_freq, small_list, big_list, super_list)
            # for i in range(len(value))
        # if (value.iloc[0]['pid'] == 0)
        running_list.append(running_time)
        runnable_list.append(runnable_time)
        idle_list.append(idle_time)
        pid_list.append(value.iloc[0]['pid'])
        comm_now_list = []
        for i in range(len(value)):
            if (i == 0):
                comm_now_list.append(value.iloc[i]['comm'])
            else:
                comm_temp = 0
                for j in range(len(comm_now_list)):
                    if (value.iloc[i]['comm'] == comm_now_list[j]):
                        comm_temp = 1
                        break
                if (comm_temp == 0):
                    comm_now_list.append(value.iloc[i]['comm'])
        comm_list.append(comm_now_list)

        cpu_running_sum += running_time
        cpu_runnable_sum += runnable_time
        cpu_idle_sum += idle_time
    # for key, value in tqdm(group)
    runtime_list.append(cpu_running_sum)
    able_list.append(cpu_runnable_sum)
    f = open(r'data_trace/{}/{}_{}_{}/freq_running_time.txt'.format(file_name, app, ii, itemp), 'w')
    print(*small_list, file=f)
    print(*big_list, file=f)
    print(*super_list, file=f)
    f.close()
    return runtime_list, able_list, total_time


# def task_plt(file_name, app, ii, itemp):
#     # writer.writerow(["comm", "runnable_time"])
#     data = pd.read_csv(f'data_process/{file_name}/{app}_{ii}_{itemp}/cpu.csv')
#     data = data.sort_values('running_time', ascending=False)
#     labels = []
#     sizes = []
#     for i in range(20):
#         labels.append(data.iloc[i]['comm'])
#         sizes.append(data.iloc[i]['runnable_time'])
#     plt.figure(figsize=(10, 8))
#     # 创建饼图
#     plt.pie(sizes, labels=labels, autopct='%1.1f%%', wedgeprops={'alpha': 0.5}, pctdistance=0.9)
#
#     # 添加标题
#     plt.title('top20 任务的 runnable_time')
#
#     plt.savefig(r'data_plt/{}/{}_{}_{}/tasktop20_runnabletime_{}.png'.format(file_name, app, ii, itemp, app))
#     print(r'保存图像文件tasktop20_runnabletime_{}.png'.format(app))


def cpu_plt(file_name, app, ii, itemp, running_list, runnable_list,total_time):
    cpu = [0, 1, 2, 3, 4, 5, 6, 7]
    loading_list = []
    for i in range(8):
        loading_list.append(running_list[i]*100/total_time)
    plt.figure(figsize=(10, 8))
    # 创建柱状图
    plt.bar(cpu, loading_list)

    # 添加标题和坐标轴名称
    plt.title(f'percpu 的 loading 占比（总时长：{total_time}）')
    plt.xlabel('8 个 CPU')
    plt.ylabel('loading 占比（%）')

    # 在每个柱子顶部显示数值
    for a, b, c in zip(cpu, loading_list, running_list):
        plt.text(a, b, f'runtime：{c}', ha='center', va='bottom')

    plt.savefig(f'data_plt/{file_name}/{app}_{ii}_{itemp}/cpuall_loading_{app}.png')
    print(f'保存图像文件cpuall_loading_{app}.png')

    plt.figure(figsize=(10, 8))
    # 创建柱状图
    plt.bar(cpu, runnable_list)

    # 添加标题和坐标轴名称
    plt.title(f'percpu 的 runnable_time（总时长：{total_time}）')
    plt.xlabel('8 个 CPU')
    plt.ylabel('runnable 时间')

    # 在每个柱子顶部显示数值
    for a, b in zip(cpu, runnable_list):
        plt.text(a, b, f'runnabletime：{b}', ha='center', va='bottom')

    plt.savefig(f'data_plt/{file_name}/{app}_{ii}_{itemp}/cpuall_runnable_{app}.png')
    print(r'保存图像文件cpuall_loading_{}.png'.format(app))

