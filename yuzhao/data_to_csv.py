import matplotlib.pyplot as plt  # matplotlib数据可视化神器
from scipy.stats import norm
from scipy.stats import laplace
import numpy as np  # numpy是Python中科学计算的核心库
import pandas as pd
import seaborn as sns
import warnings
import pickle
import sys
import os
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
def filter_false(lst):
    return list(filter(bool, lst))
def read_test(file_name, app, ii, itemp):
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
    # print("到这1")
    Kernel_Trace_Data_List = filter_false(Kernel_Trace_Data_List)

    # print("到这1")
    # print(Kernel_Trace_Data_List)
    Kernel_Trace_Data_List = Kernel_Trace_Data_List[11:]
    k = Kernel_Trace_Data_List
    # print(k)
    df1 = Kernel_Trace_Data_List[:0]
    df2 = Kernel_Trace_Data_List[:0]
    #vol = Kernel_Trace_Data_List[:0]
    ecost = Kernel_Trace_Data_List[:0]
    for i in range(len(Kernel_Trace_Data_List)):
        # print(i)
        if (k[i][4] == 'ddr_freq:' or k[i][4] == 'sched_cpu_migration:' or k[i][4]=='wakeup_migrate:' or  k[i][4]=='sched_stat_runtimes:' or k[i][4] == 'sched_migrate_task:' or k[i][4] == 'sched_eas:' or k[i][4] == 'get_my_util:'):
            if (k[i][0] == '0'):
                print(i)
                df1.append(Kernel_Trace_Data_List[i])
                # df1.append(Kernel_Trace_Data_List[i])
            else:
                df1.append(Kernel_Trace_Data_List[i])
                # print(df1)

    df2 = pd.DataFrame(df2)


    import sys
    from tqdm import tqdm
    #print(df1)
    start=1
    tick=0
    runtime_df=pd.DataFrame(columns=['timestamp','pid','comm','runtime','cpu','freq','cpu_util'])
    migrate_df=pd.DataFrame(columns=['timestamp','prev_cpu','cpu','wake_cpu','wake_freq','task_util','cpu_util','delta_time','pid'])
    eas_df=pd.DataFrame(columns=['pid','comm','prev','next','best','prev_delta','best_delta','base_energy','select_way'])
    ddr_df = pd.DataFrame(columns=['ddr_freq','delta_time','llc_freq'])
    task_mg_df = pd.DataFrame(columns=['timestamp','orig_cpu','dst_cpu','pid','running'])
    util_df = pd.DataFrame(columns=['cpu' , 'util'])

    start_time=0

    #runtime分析列表
    rt_cpu=[]
    rt_ts=[]
    rt_pid=[]
    rt_freq=[]
    rt_rt=[]
    rt_comm=[]
    rt_cpuutil = []

    #迁核列表
    mg_ts=[]
    mg_prev_cpu=[]
    mg_cpu=[]
    mg_wake_cpu=[]
    mg_wake_freq=[]
    mg_task_util=[]
    mg_cpu_util=[]
    mg_pid = []

    #eas分析列表
    eas_comm=[]
    eas_prev=[]
    eas_next=[]
    eas_pid=[]
    eas_select_way=[]
    eas_best=[]
    eas_prev_delta = []
    eas_best_delta = []
    eas_base_energy = []

    # sched_task_migrate
    task_mg_orig_cpu = []
    task_mg_dst_cpu = []
    task_mg_pid = []
    task_mg_running = []
    task_timestamp = []


    mg_delta_time=[]
    #ddr 分析列表
    ddr_freq = []
    llc_freq = []
    ddr_delta_time = []
    ddr_start = 0
    last_timestamp = 0

    #调频util分析列表
    util_cpu = []
    util_util = []
    for x in tqdm(range(0,len(df1))):
        if(x==0):
            start_time=float(df1[x][3])


        if (df1[x][4] == 'sched_stat_runtimes:' and runtime_ans==1):
            cpu = int(df1[x][5])
            freq=0
            timestamp=float(df1[x][3])-start_time
            pi=0
            for y in range(len(df1[x])):
                #if(df1[x][y] == 'last_cpu'):
                    #t_miarte.append(migrate_type(int(df1[x][y+1]),cpu))
                    #last_cpu=int(df1[x][y+1])
                if (df1[x][y] == 'comm'):
                    co=df1[x][y + 1]
                if (df1[x][y] == 'pid'):
                    pi = int(df1[x][y + 1])
                    cpu_util = int(df1[x][y + 3])
            freq0 = int(df1[x][7])
            freq4 = int(df1[x][8])
            freq7 = int(df1[x][9])
            rt = int(df1[x][6])/1e9

            if (cpu in [0, 1, 2, 3]):
                freq=freq0
            if (cpu in [4, 5, 6]):
                freq=freq4
            if (cpu in [7]):
                freq=freq7
            rt_rt.append(rt)
            rt_ts.append(timestamp)
            rt_pid.append(pi)
            rt_cpu.append(cpu)
            rt_freq.append(freq)
            rt_comm.append(co)
            rt_cpuutil.append(cpu_util)
        if (df1[x][4] == 'wakeup_migrate:' and migrate_ans ==1):
            timestamp = float(df1[x][3]) - start_time
            mg_ts.append(timestamp)
            mg_prev_cpu.append(int(df1[x][6]))
            mg_cpu.append(int(df1[x][8]))
            mg_wake_cpu.append(int(df1[x][10]))
            mg_wake_freq.append(int(df1[x][12]))
            mg_cpu_util.append(int(df1[x][14]))
            mg_task_util.append(int(df1[x][16]))
            mg_delta_time.append(0)
            mg_pid.append(int(df1[x][18]))
        if (df1[x][4] == 'sched_eas:' and eas_ans ==1):

            eas_1 = 0

            for y in range(len(df1[x])):
                if (df1[x][y] == 'comm' ):
                    eas_comm_ = df1[x][y+1]

                if (df1[x][y] == 'pid' ):
                    eas_pid_ = int(df1[x][y+1])
                    eas_1 = 1

                if (df1[x][y] == 'best_cpu'):
                    eas_next_ = int(df1[x][y+1])
                    eas_prev_ = int(df1[x][y+3])
                    eas_best_ = int(df1[x][y+5])

                if (df1[x][y] == 'prev_delta'):
                    eas_prev_delta_=int(df1[x][y + 1])
                    eas_best_delta_=int(df1[x][y + 3])
                    eas_base_energy_=int(df1[x][y + 5])
                    eas_select_way_=int(df1[x][y + 7])
            if(eas_1) :
                eas_select_way.append(eas_select_way_)
                eas_prev_delta.append(eas_prev_delta_)
                eas_next.append(eas_next_)
                eas_pid.append(eas_pid_)
                eas_best.append(eas_best_)
                eas_prev.append(eas_prev_)
                eas_comm.append(eas_comm_)
                eas_base_energy.append(eas_base_energy_)
                eas_best_delta.append(eas_best_delta_)




        if (df1[x][4] == 'ddr_freq:' and ddr_ans == 1):
            timestamp = float(df1[x][3])
            ddr_rate = int(df1[x][6])
            llc_rate = int(df1[x][8])
            if(ddr_start == 0):
                delta_time = 0
                ddr_start = 1
            else:
                delta_time = timestamp - last_timestamp
            last_timestamp = timestamp
            ddr_delta_time.append(delta_time)
            ddr_freq.append(ddr_rate)
            llc_freq.append(llc_rate)

        if (df1[x][4] == 'sched_migrate_task:' and  task_migrate_ans ==1):
            timestamp = float(df1[x][3]) - start_time
            task_timestamp.append(timestamp)
            orig_cpu = -1
            pid = -1
            dst_cpu = -1
            running = -1
            for y in range(len(df1[x])):
                if(df1[x][y] == 'orig_cpu'):
                    orig_cpu = (int(df1[x][y + 1]))
                if (df1[x][y] == 'dest_cpu'):
                    dst_cpu = (int(df1[x][y + 1]))
                if (df1[x][y] == 'pid'):
                    pid = (int(df1[x][y + 1]))
                if (df1[x][y] == 'running'):
                    running = (int(df1[x][y + 1]))
            task_mg_orig_cpu.append(orig_cpu)
            task_mg_dst_cpu.append(dst_cpu)
            task_mg_pid.append(pid)
            task_mg_running.append(running)

        if(df1[x][4] == 'get_my_util:' and  util_ans == 1 ) :
            util_cpu.append(int(df1[x][5]))
            util_util.append(int(df1[x][6]))


    if(util_ans) :
        util_df['cpu'] = util_cpu
        util_df['util'] = util_util
        util_file_path = r'data_process/{}/{}_{}_{}/util_file.csv'.format(file_name, app, ii, itemp)
        runtime_df.to_csv(util_file_path)

    if(runtime_ans) :
        runtime_df['timestamp'] = rt_ts
        runtime_df['cpu'] = rt_cpu
        runtime_df['freq'] = rt_freq
        runtime_df['pid'] = rt_pid
        runtime_df['runtime'] = rt_rt
        runtime_df['comm'] = rt_comm
        runtime_df['cpu_util'] = rt_cpuutil
        print(runtime_df)
        rt_file_path = r'data_process/{}/{}_{}_{}/runtime_file.csv'.format(file_name, app, ii, itemp)
        runtime_df.to_csv(rt_file_path)

    if(migrate_ans):
       migrate_df['timestamp'] = mg_ts
       migrate_df['cpu'] = mg_cpu
       migrate_df['prev_cpu'] = mg_prev_cpu
       migrate_df['wake_cpu'] = mg_wake_cpu
       migrate_df['wake_freq'] = mg_wake_freq
       migrate_df['cpu_util'] = mg_cpu_util
       migrate_df['task_util'] = mg_task_util
       migrate_df['delta_time'] = mg_delta_time
       migrate_df['pid'] = mg_pid
       print(migrate_df)
       mg_file_path = r'data_process/{}/{}_{}_{}/migrate_file.csv'.format(file_name, app, ii, itemp)
       migrate_df.to_csv(mg_file_path)


    if(eas_ans):
        eas_df['prev'] = eas_prev
        eas_df['next'] = eas_next
        eas_df['best'] = eas_best
        eas_df['pid'] = eas_pid
        eas_df['comm'] = eas_comm
        eas_df['select_way'] = eas_select_way
        # fix delta就是通过fix power table计算的能量差值占比，dyn delta则是通过 dyn power table计算的对应值
        eas_df['prev_delta'] = eas_prev_delta
        eas_df['best_delta'] = eas_best_delta
        eas_df['base_energy'] = eas_base_energy
        mg_file_path = r'data_process/{}/{}_{}_{}/eas_file.csv'.format(file_name, app, ii, itemp)
        eas_df.to_csv(mg_file_path)

    if (ddr_ans):
        ddr_df['ddr_freq'] = ddr_freq
        ddr_df['delta_time'] = ddr_delta_time
        ddr_df['llc_freq'] = llc_freq
        mg_file_path = r'data_process/{}/{}_{}_{}/ddr_file.csv'.format(file_name, app, ii, itemp)
        ddr_df.to_csv(mg_file_path)

    if (task_migrate_ans):
        task_mg_df['timestamp'] = task_timestamp
        task_mg_df['orig_cpu'] = task_mg_orig_cpu
        task_mg_df['dst_cpu'] = task_mg_dst_cpu
        task_mg_df['running'] = task_mg_running
        task_mg_df['pid'] = task_mg_pid

        mg_file_path = r'data_process/{}/{}_{}_{}/task_migrate_file.csv'.format(file_name, app, ii, itemp)
        task_mg_df.to_csv(mg_file_path)
    zero_count = 0
    count = 0
    base = 1000000000000
    for i in ddr_freq :
        if i < 0 :
            zero_count = zero_count +1
        count = count + 1
    #print(zero_count/count)
    print(count)
    print(np.mean(ddr_delta_time))
    print(np.var(ddr_delta_time))




if __name__ == '__main__':
    # runtime ans migrate ans 和scm ans分别对应三种文件的分析，即runtime migrate 和scm
    file_name = sys.argv[1]
    app = sys.argv[2]
    ii = int(sys.argv[3])
    itemp = int(sys.argv[4])
    runtime_ans = int(sys.argv[7])
    migrate_ans = 0
    util_ans = 0
    scm_ans = 0
    eas_ans = 0
    ddr_ans = int(sys.argv[5])
    task_migrate_ans = int(sys.argv[6])
    small_scale = 325
    big_scale = 828
    super_scale = 1024
    small_max = 1804800
    big_max = 2419200
    super_max = 2841600
    #input_1, data, input_2 = read_test(file_name, app, ii, itemp)
    small=[]
    big=[]
    super=[]
    l_power=[]
    read_test(file_name, app, ii, itemp)
    #生成csv文件后就不需要源文件了，所以这里删除源文件
    #os.remove(r'data_trace/{}/{}_{}_{}/trace.txt'.format(file_name, app, ii, itemp))
