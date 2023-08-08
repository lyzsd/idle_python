import matplotlib.pyplot as plt  # matplotlib数据可视化神器
from scipy.stats import norm
from scipy.stats import laplace
import numpy as np  # numpy是Python中科学计算的核心库
import pandas as pd
import seaborn as sns
import warnings
import pickle
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from collections import defaultdict
from collections import Counter
import sys
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

opp7_power = { 844800: 221, 960000: 266, 1075200: 306,
    1190400: 356, 1305600: 401, 1420800: 458,
    1555200: 540, 1670400: 614, 1785600: 695,
    1900800: 782, 2035200: 893, 2150400: 1035,
    2265600: 1203, 2380800: 1362, 2496000: 1536,
    2592000: 1725, 2688000: 1898, 2764800: 2017,
    2841600: 2141}



def in_cluster(a,b):
    if(a in b):
        return True
    else:return False

def migrate_type_0(cpu,next_cpu):
    small=[0,1,2,3]
    big=[4,5,6]
    if(cpu==next_cpu):
        return 0
    else:
        if(cpu in small):
            if(next_cpu in small):
                return 1
            else :return 2
        if (cpu in big):
            if (next_cpu in big):
                return 1
            else:
                return 2
        if (cpu ==7):
            return 2
def migrate_type(cpu,next_cpu):
    small=[0,1,2,3]
    big=[4,5,6]
    if(cpu==next_cpu):
        if(cpu in small):
            return 1
        if (cpu in big):
            return 5
        if (cpu ==7):
            return 9

    else:
        if(cpu in small):
            if(next_cpu in small):
                return 2
            if(next_cpu in big):
                return 3
            else : return 4
        if(cpu in big):
            if (next_cpu in small):
                return 6
            if (next_cpu in big):
                return 7
            else:
                return 8
        if(cpu ==7):
            if (next_cpu in small):
                return 10
            if (next_cpu in big):
                return 11
def getpackname(app):
    pack=""
    if(app=="dy"):
        pack="droid.ugc.aweme"
    if(app=="_tt"):
        pack="com.ss.android.article.news"
    if(app=="_txsp"):
        pack="com.tencent.qqlive"
    if(app=="_wangzhe"):
        pack="com.tencent.tmgp.sgame"
    if(app=="_tb"):
        pack="com.taobao.taobao"
    if(app=="_wx"):
        pack="com.tencent.mm"
    if (app == "_antutu"):
        pack = "com.antutu.ABenchMark"
    return pack
def ipc(x):
  if x['cycle'] == 0:
    return 0
  else:
    return x['inst'] / x['cycle']
def app_plt(app):
    data1 = pd.read_csv(r'result/time_if_jank_{}.csv'.format(app))
    data2 = pd.read_csv(r'result/pmu_{}.csv'.format(app))
    #data3 = pd.read_csv(r'result/every_jank_{}.csv'.format(app))

    # 图片尺寸

    plt.figure(figsize=(200, 50))
    # 图片拼接
    # 图1：ipc数据
    big=[4,5,6,7]
    data2['cpu'] = data2['cpu'].astype(int)
    big_df=data2[data2['cpu'].isin(big)]
    #print(big_df)
    big_df['ipc'] = big_df.apply(ipc, axis=1)
    plt.subplot(3, 1, 1)
    #print(big_df)
    plt.scatter(big_df["pmu_time"], big_df["ipc"], label='ipc-retired_big', c="black",s=1)
    plt.scatter(big_df["pmu_time"], big_df["freq"]/1000000, label='ipc-freq_big', c="red",s=1)
    # 根据every_jank_app.csv（每一帧）划分丢帧区域
    for i, row in data1.iterrows():
        if (row['if_jank'] == 1):
            plt.axvspan(row['vsync_time'], row['start_time'] + 0.0167, alpha=0.2, color='red')
        else:
            plt.axvspan(row['vsync_time'], row['start_time'] + 0.0167, alpha=0.2, color='green')



    plt.legend()
    plt.xlabel('time')
    plt.ylabel('ipc')
    plt.ylabel('freq')
    plt.title('retiredipc&jank')
    i=0
    time=[]
    inst=[]
    #print(big_df['pmu_time'][100])
    while (i < len(data2['pmu_time'])):
        inst_count=0
        #print(i)
        start_time=data2['pmu_time'][i]
        while(data2['pmu_time'][i] < start_time+0.016):
            inst_count=inst_count+data2['cycle'][i]
            i=i+1
        time.append(start_time+0.0008 )
        i=i+1

        inst.append(inst_count)

    print(inst)
    plt.subplot(3, 1, 2)
    # print(big_df)
    plt.scatter(time, inst, label='ipc-retired_big', c="black", s=1)
    #plt.scatter(big_df["pmu_time"], big_df["freq"] / 10000, label='ipc-freq_big', c="red", s=1)
    # 根据every_jank_app.csv（每一帧）划分丢帧区域

    for i, row in data1.iterrows():
        if (row['if_jank'] == 1):
            plt.axvspan(row['vsync_time'], row['start_time'] + 0.0167, alpha=0.2, color='red')
        else:
            plt.axvspan(row['vsync_time'], row['start_time'] + 0.0167, alpha=0.2, color='green')

    plt.legend()
    plt.xlabel('time')
    plt.ylabel('ipc')
    plt.ylabel('freq')
    plt.title('retiredipc&jank')
    plt.subplot(3, 1, 3)
    # print(big_df)
    time = []
    inst = []
    # print(big_df['pmu_time'][100])
    i=0
    while (i < len(data2['pmu_time'])):
        inst_count = 0
        # print(i)
        start_time = data2['pmu_time'][i]
        while (data2['pmu_time'][i] < start_time + 0.016):
            inst_count = inst_count + data2['inst'][i]
            i = i + 1
        time.append(start_time + 0.0008)
        i = i + 1

        inst.append(inst_count)
    plt.scatter(time, inst, label='ipc-retired_big', c="black", s=1)
    plt.scatter(big_df["pmu_time"], big_df["freq"] / 10000, label='ipc-freq_big', c="red", s=1)
    # 根据every_jank_app.csv（每一帧）划分丢帧区域
    for i, row in data1.iterrows():
        if (row['if_jank'] == 1):
            plt.axvspan(row['vsync_time'], row['start_time'] + 0.0167, alpha=0.2, color='red')
        else:
            plt.axvspan(row['vsync_time'], row['start_time'] + 0.0167, alpha=0.2, color='green')

    plt.legend()
    plt.xlabel('time')
    plt.ylabel('ipc')
    plt.ylabel('freq')
    plt.title('retiredipc&jank')
    plt.show()
    jank_label = ['no jank', 'jank < 5', 'jank in [5,20]', 'jank in [20,]']
    jank_list = []
    i=0
    while (i < len(data1['if_jank'])):
        if (data1['if_jank'][i] == 1):
            #print(1)
            count_jank = 0
            while (data1['if_jank'][i] != 0):
                print(1)
                count_jank = count_jank + 1
                i = i + 1
            if (count_jank < 5):
                jank_list.append(jank_label[1])
            if (count_jank >= 5 and count_jank < 20):
                jank_list.append(jank_label[2])
            if (count_jank >= 20):
                jank_list.append(jank_label[3])
        else:
            jank_list.append(jank_label[0])
        i = i + 1
    print(len(jank_list))
    plot_pie(jank_list)
def plot_bar(data1,data2,data3):
    # 设置表头
    headers = [300000, 403200, 499200, 595200, 691200, 806400, 902400, 998400, 1094400, 1209600, 1305600, 1401600,
               1497600, 1612800, 1708800, 1804800]

    # 创建数据

    x1 = np.arange(19)
    x = np.arange(16)


    width = 0.2
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(11, 9))


    rects1 = ax[0].bar(x - width, data1[0], width, label='total')
    rects2 = ax[0].bar(x, data1[1], width, label='main task')


    ax[0].set_xticks(x)
    ax[0].legend()
    rects1 = ax[1].bar(x - width, data2[0], width, label='total')
    rects2 = ax[1].bar(x, data2[1], width, label='main task')


    ax[1].set_xticks(x)
    ax[1].legend()

    rects1 = ax[2].bar(x1 - width, data3[0], width, label='total')
    rects2 = ax[2].bar(x1, data3[1], width, label='main task')




    ax[2].set_xticks(x)
    ax[2].legend()

    fig.tight_layout()
    plt.show()

def plot_density(data1,data2,data3):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))

    # 绘制分布图
    sns.set(style='dark')
    sns.set_style("dark", {"axes.facecolor": "#e9f3ea"})

    # 第一张图
    sns.distplot(data1,
                 hist=True,
                 kde=True,
                 kde_kws={'linestyle': '--', 'linewidth': '1', 'color': '#c72e29'},
                 fit=norm,
                 color='#098154',
                 axlabel='cpu',
                 ax=axs[0]
                 )
    axs[0].legend(labels=['$Density$', '$Normal distribution$', '$mean$'])

    # 第二张图
    sns.distplot(data2,
                 hist=True,
                 kde=True,
                 kde_kws={'linestyle': '--', 'linewidth': '1', 'color': '#c72e29'},
                 fit=norm,
                 color='#098154',
                 axlabel=' in cluster',
                 ax=axs[1]
                 )
    axs[1].legend(labels=['$Density$', '$Normal distribution$', '$mean$'])

    # 第三张图
    sns.distplot(data3,
                 hist=True,
                 kde=True,
                 kde_kws={'linestyle': '--', 'linewidth': '1', 'color': '#c72e29'},
                 fit=norm,
                 color='#098154',
                 axlabel='out_cluster',
                 ax=axs[2]
                 )
    axs[2].legend(labels=['$Density$', '$Normal distribution$', '$mean$'])

    # 调整布局和子图间距
    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(wspace=0.1)

    # 显示图像
    plt.show()
def dis_plot(length_1,length_2,length_3):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

    # 绘制分布图
    sns.set(style='dark')
    sns.set_style("dark", {"axes.facecolor": "#e9f3ea"})

    # 第一张图
    sns.distplot(length_1,
                 hist=True,
                 kde=True,
                 kde_kws={'linestyle': '--', 'linewidth': '1', 'color': 'red'},
                 fit=norm,
                 color='#098154',
                 axlabel='cpu',
                 ax=axs
                 )
    axs.legend(labels=['$Density$', '$Normal distribution$', '$mean$'])
    sns.distplot(length_2,
                 hist=True,
                 kde=True,
                 kde_kws={'linestyle': '--', 'linewidth': '1', 'color': 'blue'},
                 fit=norm,
                 color='blue',
                 axlabel='cpu',
                 ax=axs
                 )

    fig.tight_layout(pad=2.0)
    fig.subplots_adjust(wspace=0.1)

    # 显示图像
    plt.show()

def scatter_box_plot(data1,data2,data3):
    # 创建画布和子图
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制第一组数据的箱线图和散点图

    ax.boxplot(data1, positions=[1], vert=True,showfliers=False, showmeans=True, boxprops = {'color':'blue'})
    #ax.scatter([1] * len(data1), data1, color='b', alpha=0.5, s=0.1)

    # 绘制第二组数据的箱线图和散点图
    ax.boxplot(data2, positions=[2], vert=True,showfliers=False, showmeans=True,boxprops = {'color':'green'})
    #ax.scatter([2] * len(data2), data2, color='r', alpha=0.5, s=0.1)

    # 绘制第三组数据的箱线图和散点图
    ax.boxplot(data3, positions=[3], vert=True,showfliers=False, showmeans=True,boxprops = {'color':'red'})
    #ax.scatter([3] * len(data3), data3, color='g', alpha=0.5, s=0.1)

    # 设置坐标轴和标题
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Test 1', 'Test 2', 'Test 3'])
    ax.set_ylabel('Value')
    ax.set_title('Multiple Box plot with scatter')

    # 显示图形
    plt.show()

    # 显示图形
    #plt.show()
def plot_pie(l_irq_id):
    # 使用Counter类进行频次统计，得到一个字典形式的结果
    plt.figure(figsize=(20, 6.5))
    result = dict(Counter(l_irq_id))
    print(result)

    # 计算所有数据的总数
    total = sum(result.values())

    # 将频次占比小于1%的数据合并到“其它”中
    threshold = total * 0.000001
    other_count = 0
    for k in list(result.keys()):
        if result[k] < threshold:
            other_count += result[k]
            del result[k]
    #result['Other'] = other_count

    # 生成饼图
    labels = list(result.keys())
    l_labels=[]


    sizes = [result[k] for k in labels]
    count=np.sum(sizes)
    #计算结果保留两位小数
    for i in range(len(labels)):
        l_labels.append(labels[i]+' : {}'.format(sizes[i]) +' : {}%'.format(round(sizes[i]/count*100,2)))
    #print(sizes)
    wedgeprops = {'linewidth': 1, 'edgecolor': 'black'}
    plt.pie(sizes,  startangle=90,labeldistance=1.1, pctdistance=1.2)
    plt.legend(labels=l_labels)
    plt.title('Pie chart')

    # 显示图像
    plt.show()

import string
def find_string(string_list, string):
    if string in string_list:
        return string_list.index(string)+1
    else:
        string_list.append(string)
        return len(string_list)

def runtime_plt():
    data1 = pd.read_csv(r'data_process/{}/{}_{}_{}/runtime_file.csv'.format(file_name, app, ii, itemp))


    small=[0,1,2,3]
    big=[4,5,6]
    super=[7]
    #统计频点信息
    small_list=[0 for i in range(16)]
    big_list=[0 for i in range(16)]
    super_list=[0 for i in range(19)]
    small_list_util = [[] for i in range(16)]
    big_list_util = [[] for i in range(16)]
    super_list_util = [[] for i in range(19)]
    comm_small_list = [0 for i in range(19)]
    comm_big_list = [0 for i in range(19)]
    comm_super_list = [0 for i in range(19)]
    cpu_runtime_list=[0]*8
    cpu_runtime_list_mt = [0] * 8
    comm=getpackname(app_way)
    #统计功耗信息列表
    print(comm)
    small_energy=[]
    big_energy=[]
    super_energy=[]
    rt=[]
    #判断是否为主线程
    runtime_rank_dict = defaultdict(int)

    lenght=len(data1['cpu'])
    total_time=data1['timestamp'][lenght-1]-data1['timestamp'][0]
    for i in tqdm(range(len(data1['cpu']))):
        freq = data1['freq'][i]
        cpu = data1['cpu'][i]
        task_comm = data1['comm'][i]
        #print(comm)
        mthread = 0
        # 对comm的running_time统计
        item = [task_comm, data1['runtime'][i]]
        runtime_rank_dict[item[0]] += item[1]

        cpu_runtime_list[cpu] = cpu_runtime_list[cpu]+data1['runtime'][i]
        cpu_util = data1['cpu_util'][i]
        #if (int(data1['pid'][i]) in  main_thread_list) :
        if (int(data1['pid'][i])  == main_thread_pid):
            mthread = 1
            cpu_runtime_list_mt[cpu] = cpu_runtime_list_mt[cpu] + data1['runtime'][i]

        if(data1['cpu'][i] in small):
            idx=F[0].index(freq)
            runtime=data1['runtime'][i]
            small_list[idx]=small_list[idx]+runtime
            if(cpu_util) :
                small_list_util[idx].append(cpu_util)
            power=opp0_power[freq]
            small_energy.append(power*runtime)
            if(mthread == 1):
                comm_small_list[idx] = comm_small_list[idx] + runtime
        if (data1['cpu'][i] in big):
            idx = F[1].index(freq)
            runtime = data1['runtime'][i]
            big_list[idx] = big_list[idx] + runtime
            if(cpu_util) :
                big_list_util[idx].append(cpu_util)
            #计算功耗
            power = opp4_power[freq]
            big_energy.append(power * runtime)
            if (mthread == 1):
                comm_big_list[idx] = comm_big_list[idx] + runtime
        if (data1['cpu'][i] in super):
            idx = F[2].index(freq)
            runtime = data1['runtime'][i]
            super_list[idx] = super_list[idx] + runtime
            if(cpu_util) :
                super_list_util[idx].append(cpu_util)
            power = opp7_power[freq]
            super_energy.append(power * runtime)
            if (mthread == 1):
                comm_super_list[idx] = comm_super_list[idx] + runtime
        rt.append(runtime)
    list1=[small_list,comm_small_list]
    list2=[big_list,comm_big_list]
    list3=[super_list,comm_super_list]
    #plot_bar(list1,list2,list3)
    #scatter_box_plot(small_energy,big_energy,super_energy)
    #plot_density(small_energy,big_energy,super_energy)


    total_energy_0=np.sum(small_energy)
    total_energy_4 = np.sum(big_energy)
    total_energy_7 = np.sum(super_energy)
    total_energy=total_energy_0+total_energy_4+total_energy_7
    power=total_energy/total_time
    total_rt=np.sum(rt)

    sorted_d = sorted(runtime_rank_dict.items(), key=lambda x: x[1], reverse=True)

    runtime_top_20 = []
    for i, (key, value) in enumerate(sorted_d[:20]):
        runtime_top_20.append([key, value])

    conext=''
    #统计不同簇，不同频点的cpu util均值，方差
    small_util_mean = []
    big_util_mean = []
    super_util_mean = []
    small_util_std = []
    big_util_std = []
    super_util_std = []
    for i in range(16):
        small_util_mean.append(np.mean(small_list_util[i]))
        small_util_std.append(np.std(small_list_util[i]))
        big_util_mean.append(np.mean(big_list_util[i]))
        big_util_std.append(np.std(big_list_util[i]))
    for i in range(19):
        super_util_mean.append(np.mean(super_list_util[i]))
        super_util_std.append(np.std(super_list_util[i]))



    f = open(r'data_trace/{}/{}_{}_{}/runtime_top_20.txt'.format(file_name, app, ii, itemp), 'a')
    print(*runtime_top_20, file=f)
    f.close()

  
    f1 = open(r'data_trace/{}/{}_{}_{}/energy.txt'.format(file_name, app, ii, itemp), 'r')
    context = f1.read()

    f1.close()
    f = open(r'data_trace/{}/{}_{}_{}/energy.txt'.format(file_name, app, ii, itemp), 'a')
    if(context):
        total_result = [total_rt,power, total_energy, total_energy_0, total_energy_4, total_energy_7]
        print(*total_result, file=f)
    else :
        total_result = ['total_rt','power', 'total_energy', 'total_energy_0', 'total_energy_4', 'total_energy_7']
        print(*total_result, file=f)
        total_result = [total_rt,power, total_energy, total_energy_0, total_energy_4, total_energy_7]
        print(*total_result, file=f)
    f.close()




    f1 = open(r'data_trace/{}/{}_{}_{}/runtime.txt'.format(file_name, app, ii, itemp), 'r')
    context = f1.read()
    f1.close()
    f = open(r'data_trace/{}/{}_{}_{}/runtime.txt'.format(file_name, app, ii, itemp), 'a')
    if (context):
        print(*cpu_runtime_list, file=f)
    else:
        total_result = ['cpu0', 'cpu1', 'cpu2', 'cpu3', 'cpu4', 'cpu5', 'cpu6', 'cpu7']
        print(*total_result, file=f)
        total_result = cpu_runtime_list
        print(*total_result, file=f)
    f.close()


    f = open(r'data_trace/{}/{}_{}_{}/runtime_small_freq.txt'.format(file_name, app, ii, itemp), 'a')
    print(*small_list, file=f)
    print(*small_util_mean, file=f)
    print(*small_util_std, file=f)
    f.close()
    f = open(r'data_trace/{}/{}_{}_{}/runtime_big_freq.txt'.format(file_name, app, ii, itemp), 'a')
    print(*big_list, file=f)
    print(*big_util_mean, file=f)
    print(*big_util_std, file=f)
    f.close()
    f = open(r'data_trace/{}/{}_{}_{}/runtime_super_freq.txt'.format(file_name, app, ii, itemp), 'a')
    print(*super_list, file=f)
    print(*super_util_mean, file=f)
    print(*super_util_std, file=f)
    f.close()

    if (main_thread == 1 ) :

        f1 = open(r'data\main_thread\runtime\runtime_{}.txt'.format(app), 'r')
        context = f1.read()
        f1.close()
        f = open(r'data\main_thread\runtime\runtime_{}.txt'.format(app), 'a')
        if (context):
            print(*cpu_runtime_list_mt, file=f)
        else:
            total_result = ['cpu0', 'cpu1', 'cpu2', 'cpu3', 'cpu4', 'cpu5', 'cpu6', 'cpu7']
            print(*total_result, file=f)
            total_result = cpu_runtime_list_mt
            print(*total_result, file=f)
        f.close()

        f = open(r'data\main_thread\small\runtime_freq_{}.txt'.format(app), 'a')
        print(*comm_small_list, file=f)
        f.close()
        f = open(r'data\main_thread\big\runtime_freq_{}.txt'.format(app), 'a')
        print(*comm_big_list, file=f)
        f.close()
        f = open(r'data\main_thread\super\runtime_freq_{}.txt'.format(app), 'a')
        print(*comm_super_list, file=f)
        f.close()






def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf


def migrate_plt():
    migrate_label=['in small cpu','small in cluster','small from big','small from super','in big cpu','big from small','big in cluster','big from super','in super cpu','super from small','super from big']
    migrate_label_0=['no migrate','migrate in cluster','migrate over cluster']
    data1 = pd.read_csv(r'data_process/{}/{}_{}_{}/migrate_file.csv'.format(file_name, app, ii, itemp))
    lenght = len(data1['cpu'])
    total_time = data1['timestamp'][lenght - 1] - data1['timestamp'][0]
    migrate_list=[[],[],[],[]]
    no_migrate_list= [[],[],[],[]]
    migrate_type_list=[]
    migrate_type_list_mt = []
    m_list_pie=[]
    m_list_pie_0=[]
    l_task_util=[]
    for i in tqdm(range(len(data1['cpu']))):
        timestamp=data1['timestamp'][i]
        delta_time=data1['delta_time'][i]
        freq=data1['wake_freq'][i]*1000
        prev_cpu= data1['prev_cpu'][i]
        next_cpu=data1['cpu'][i]
        task_util=data1['task_util'][i]
        pid = data1['pid'][i]
        if(delta_time*freq<500000):
            m_list_pie.append(migrate_label[migrate_type(prev_cpu,next_cpu)-1])

            m_list_pie_0.append(migrate_label_0[migrate_type_0(prev_cpu,next_cpu)])

            migrate_type_list.append(migrate_type(prev_cpu,next_cpu))
            l_task_util.append(task_util)
            #if(pid in main_thread_list ) :
            if (pid == main_thread_pid):
                #print(1)
                migrate_type_list_mt.append(migrate_type(prev_cpu,next_cpu))
            if (next_cpu != prev_cpu):
                # migrate_list.append([timestamp,delta_time])
                migrate_list[0].append(timestamp)
                migrate_list[1].append(delta_time )
                migrate_list[2].append(task_util)
                migrate_list[3].append(delta_time*freq)
            else:
                no_migrate_list[0].append(timestamp)
                no_migrate_list[1].append(delta_time )
                no_migrate_list[2].append(task_util)
                no_migrate_list[3].append(delta_time * freq)

    print(len(migrate_list[0])+len(no_migrate_list[0]))
    print(lenght)
    #migrate_list=[[x for y in migrate_list for x in y]]
    #plot_pie(m_list_pie)
    #plot_pie(m_list_pie_0)
    #plt.scatter(migrate_list[0],migrate_list[2],s=1,c='r')
    #plt.scatter(no_migrate_list[0], no_migrate_list[2],s=1,c='b')
    #plt.show()
    #scatter_box_plot(migrate_list[2],no_migrate_list[2],no_migrate_list[2])
    #dis_plot(migrate_list[1],no_migrate_list[1],no_migrate_list[1])
    #dis_plot(no_migrate_list[2], migrate_list[2], [])
    #dis_plot(migrate_list[1],no_migrate_list[1],no_migrate_list[1])
    #dis_plot(l_task_util,l_task_util,l_task_util)
    print(np.mean(migrate_list[1]))
    print(np.mean(no_migrate_list[1]))
    migrate_time=np.mean(migrate_list[1])-np.mean(no_migrate_list[1])
    migrate_cycle=np.mean(migrate_list[3]) - np.mean(no_migrate_list[3])
    print(migrate_time)
    print(migrate_cycle)
    migrate_count=[0]*11
    for i in migrate_type_list:
        migrate_count[i-1]=migrate_count[i-1]+1
    f = open(r'data_trace/{}/{}_{}_{}/migrate.txt'.format(file_name, app, ii, itemp) , 'a')
    f.close()
    f1 = open(r'data_trace/{}/{}_{}_{}/migrate.txt'.format(file_name, app, ii, itemp) , 'r')
    context = f1.read()
    f1.close()
    f = open(r'data_trace/{}/{}_{}_{}/migrate.txt'.format(file_name, app, ii, itemp) , 'a')
    if (context):

        print(*migrate_count, file=f)
    else:

        print(*migrate_label, file=f)

        print(*migrate_count, file=f)
    f.close()
    migrate_count = [0] * 11
    for i in migrate_type_list_mt:
        migrate_count[i-1]=migrate_count[i-1]+1
    if (main_thread):
        f1 = open(r'data\main_thread\migrate\migrate_{}.txt'.format(app), 'r')
        context = f1.read()
        f1.close()
        f = open(r'data\main_thread\migrate\migrate_{}.txt'.format(app), 'a')
        if (context):

            print(*migrate_count, file=f)
        else:

            print(*migrate_label, file=f)

            print(*migrate_count, file=f)
        f.close()


    return migrate_time,migrate_cycle
def eas_plt():
    #设置迁核label
    migrate_label = ['in small cpu', 'small in cluster', 'small from big', 'small from super', 'in big cpu',
                     'big from small', 'big in cluster', 'big from super', 'in super cpu', 'super from small',
                     'super from big']
    migrate_label_0 = ['no migrate', 'migrate in cluster', 'migrate over cluster']
    #读取数据
    data1 = pd.read_csv(r'data_process/{}/{}_{}_{}/eas_file.csv'.format(file_name, app, ii, itemp))
    #初始化统计列表
    migrate_type_list = []
    migrate_type_list_eas = []
    #用于统计迁核概率
    diff_list = [0] * 22
    #统计eas次数
    eas_count = 0
    #统计总次数
    total_count  = 0
    #遍历处理数据，不同迁核类型进行统计
    for i in tqdm(range(len(data1['prev']))):
        total_count = total_count + 1
        prev_cpu = data1['prev'][i]
        next_cpu = data1['next'][i]
        best_cpu = data1['best'][i]
        select_way = data1['select_way'][i]
        prev_delta =  data1['prev_delta'][i]
        best_delta = data1['best_delta'][i]
        base_energy = data1['base_energy'][i]
        migrate_type_list.append(migrate_type(prev_cpu, next_cpu))
        if(select_way == 2) :
            migrate_type_list_eas.append(migrate_type(prev_cpu, next_cpu))
            eas_count = eas_count + 1
            for j in range(20):
                if(prev_delta-best_delta > 0.006*j*base_energy):
                    next_cpu = best_cpu
                else : next_cpu = prev_cpu
                if (next_cpu != prev_cpu):
                    diff_list[j] = diff_list[j] + 1
    for x in range(20) :
        diff_list[x] = diff_list[x]/eas_count
    diff_list[20] = eas_count
    diff_list[21] = total_count
    migrate_count = [0] * 11
    migrate_count_eas = [0] * 11
    for i in migrate_type_list:
        migrate_count[i - 1] = migrate_count[i - 1] + 1
    for i in migrate_type_list_eas:
        migrate_count_eas[i - 1] = migrate_count_eas[i - 1] + 1
    f = open(r'data_trace/{}/{}_{}_{}/eas_mig.txt'.format(file_name, app, ii, itemp), 'a')
    print(*migrate_count , file=f)
    print(*migrate_count_eas, file=f)
    f.close()
    f = open(r'data_trace/{}/{}_{}_{}/eas_ans.txt'.format(file_name, app, ii, itemp), 'a')
    print(*diff_list, file=f)
    f.close()



def util_plt() :
    data1 = pd.read_csv(r'data_process/{}/{}_{}_{}/cal_lit_freq.csv'.format(file_name, app, ii, itemp))
    data2 = pd.read_csv(r'data_process/{}/{}_{}_{}/cal_big_freq.csv'.format(file_name, app, ii, itemp))
    small_freq_list = [[0] * 16, [0] * 16, [0] * 16, [0] * 16]
    big_freq_list = [[0] * 16, [0] * 16, [0] * 16]
    small_freq_df = pd.DataFrame(columns=['cpu0', 'cpu1', 'cpu2', 'cpu3'])
    big_freq_df = pd.DataFrame(columns=['cpu4', 'cpu5', 'cpu6'])
    for i in tqdm(range(len(data1['cpu0']))):
        freq_0 = data1['cpu0'][i]
        freq_1 = data1['cpu1'][i]
        freq_2 = data1['cpu2'][i]
        freq_3 = data1['cpu3'][i]
        small_freq_list[0][F[0].index(freq_0)] = small_freq_list[0][F[0].index(freq_0)] + 1
        small_freq_list[1][F[0].index(freq_1)] = small_freq_list[0][F[0].index(freq_1)] + 1
        small_freq_list[2][F[0].index(freq_2)] = small_freq_list[0][F[0].index(freq_2)] + 1
        small_freq_list[3][F[0].index(freq_3)] = small_freq_list[0][F[0].index(freq_3)] + 1
    for i in tqdm(range(len(data2['cpu4']))):
        freq_4 = data2['cpu4'][i]
        freq_5 = data2['cpu5'][i]
        freq_6 = data2['cpu6'][i]
        big_freq_list[0][F[1].index(freq_4)] = big_freq_list[0][F[1].index(freq_4)] + 1
        big_freq_list[1][F[1].index(freq_5)] = big_freq_list[0][F[1].index(freq_5)] + 1
        big_freq_list[2][F[1].index(freq_6)] = big_freq_list[0][F[1].index(freq_6)] + 1
    small_freq_df['cpu0'] = small_freq_list[0]
    small_freq_df['cpu1'] = small_freq_list[1]
    small_freq_df['cpu2'] = small_freq_list[2]
    small_freq_df['cpu3'] = small_freq_list[3]
    big_freq_df['cpu4'] = big_freq_list[0]
    big_freq_df['cpu5'] = big_freq_list[1]
    big_freq_df['cpu6'] = big_freq_list[2]
    mg_file_path = r'data_process/{}/{}_{}_{}/small_util_freq.csv'.format(file_name, app, ii, itemp)
    small_freq_df.to_csv(mg_file_path)
    mg_file_path = r'data_process/{}/{}_{}_{}/big_util_freq.csv'.format(file_name, app, ii, itemp)
    big_freq_df.to_csv(mg_file_path)






def task_migrate():
    migrate_label = ['in small cpu', 'small in cluster', 'small from big', 'small from super', 'in big cpu',
                     'big from small', 'big in cluster', 'big from super', 'in super cpu', 'super from small',
                     'super from big']
    data1 = pd.read_csv(r'data_process/{}/{}_{}_{}/task_migrate_file.csv'.format(file_name, app, ii, itemp))
    migrate_type_list = []
    migrate_type_list_mt = []
    for i in tqdm(range(len(data1['orig_cpu']))):
        prev_cpu = data1['orig_cpu'][i]
        next_cpu = data1['dst_cpu'][i]
        migrate_type_list.append(migrate_type(prev_cpu, next_cpu))
        if( int(data1['pid'][i]) == main_thread_pid):
            migrate_type_list_mt.append(migrate_type(prev_cpu, next_cpu))
    migrate_count = [0] * 11
    #print(migrate_type_list)
    for i in migrate_type_list:
        if(i != None):
            migrate_count[i - 1] = migrate_count[i - 1] + 1
    f =  open(r'data_trace/{}/{}_{}_{}/task_migrate.txt'.format(file_name, app, ii, itemp) ,'a')
    f.close()
    f1 =  open(r'data_trace/{}/{}_{}_{}/task_migrate.txt'.format(file_name, app, ii, itemp) ,'r')
    context = f1.read()
    f1.close()
    f =  open(r'data_trace/{}/{}_{}_{}/task_migrate.txt'.format(file_name, app, ii, itemp) ,'a')
    if (context):
        print(*migrate_count, file=f)
    else:
        print(*migrate_label, file=f)
        print(*migrate_count, file=f)
    f.close()
    if (main_thread == 1) :
        migrate_count_mt = [0] * 11
        for i in migrate_type_list_mt:
            migrate_count_mt[i - 1] = migrate_count_mt[i - 1] + 1
        f1 = open(r'data\main_thread\task_migrate\task_migrate_mt_{}.txt'.format(app), 'r')
        context = f1.read()
        f1.close()
        f = open(r'data\main_thread\task_migrate\task_migrate_mt_{}.txt'.format(app), 'a')
        if (context):
            print(*migrate_count_mt, file=f)
        else:
            print(*migrate_label, file=f)
            print(*migrate_count_mt, file=f)
        f.close()


def ddr_read():

    data1 = pd.read_csv(r'data_process/{}/{}_{}_{}/ddr_file.csv'.format(file_name, app, ii, itemp))
    ddr_freq_list = []
    ddr_label = [312, 365, 477 , 585, 643, 982 ,1302, 1467, 1827 ,2216, 5000]

    for i in tqdm(range(len(data1['ddr_freq']))):
      ddr_freq_list.append(data1['ddr_freq'][i])
    ddr_freq = [0] * 11
    for i in ddr_freq_list:
        index = ddr_label.index(i)
        ddr_freq[index] = ddr_freq[index] + 1

    f = open(r'data_trace/{}/{}_{}_{}/ddr_freq.txt'.format(file_name, app, ii, itemp) ,'a')
    print(*ddr_freq, file=f)
    f.close()





def plt_box_cs():
    data1 = pd.read_csv(r'result/cs1.csv')
    data2 = pd.read_csv(r'result/cs2.csv')
    data3 = pd.read_csv(r'result/cs3.csv')

    l1=data1['Current(uA)']
    l2=data2['Current(uA)']
    l3=data3['Current(uA)']

    #print(data_2)
    scatter_box_plot(l1,l2,l3)

def clean_file():
    if( migrate_ans):
        f = open(r'data_trace/{}/{}_{}_{}/migrate.txt'.format(file_name, app, ii, itemp) ,'w')
        f.close()
    if(runtime_ans):
        f = open(r'data_trace/{}/{}_{}_{}/energy.txt'.format(file_name, app, ii, itemp) ,'w')
        f.close()
        f = open(r'data_trace/{}/{}_{}_{}/runtime.txt'.format(file_name, app, ii, itemp) ,'w')
        f.close()
        f = open(r'data_trace/{}/{}_{}_{}/runtime_small_freq.txt'.format(file_name, app, ii, itemp) ,'w')
        f.close()
        f = open(r'data_trace/{}/{}_{}_{}/runtime_big_freq.txt'.format(file_name, app, ii, itemp) ,'w')
        f.close()
        f = open(r'data_trace/{}/{}_{}_{}/runtime_super_freq.txt'.format(file_name, app, ii, itemp) ,'w')
        f.close()
        f = open(r'data_trace/{}/{}_{}_{}/runtime_top_20.txt'.format(file_name, app, ii, itemp) ,'w')
        f.close()

    if( eas_ans ):
        f = open(r'data_trace/{}/{}_{}_{}/eas_mig.txt'.format(file_name, app, ii, itemp) ,'w')
        f.close()
        f = open(r'data_trace/{}/{}_{}_{}/eas_ans.txt'.format(file_name, app, ii, itemp), 'w')
        f.close()


    if (ddr_ans):
        f =  open(r'data_trace/{}/{}_{}_{}/ddr_freq.txt'.format(file_name, app, ii, itemp) ,'w')
        f.close()

    if(task_migrate_ans):
        f =  open(r'data_trace/{}/{}_{}_{}/task_migrate.txt'.format(file_name, app, ii, itemp) ,'w')
        f.close()



    if (main_thread == 1 ) :
        if(runtime_ans):
            f = open(r'data\main_thread\small\runtime_freq_{}.txt'.format(app), 'w')
            f.close()
            f = open(r'data\main_thread\big\runtime_freq_{}.txt'.format(app), 'w')
            f.close()
            f = open(r'data\main_thread\super\runtime_freq_{}.txt'.format(app), 'w')
            f.close()
            f = open(r'data\main_thread\runtime\runtime_{}.txt'.format(app), 'w')
            f.close()

        if(migrate_ans) :
            f = open(r'data\main_thread\migrate\migrate_{}.txt'.format(app), 'w')
            f.close()
        if (task_migrate_ans):
            f = open(r'data\main_thread\task_migrate\task_migrate_mt_{}.txt'.format(app), 'w')
            f.close()



def read_main_thread(i):
    Kernel_Trace_Data = pd.read_table(r'E:\benchmark\pmu\{}\{}\{}\main_thread_pid.txt'.format(app_way,stra,app_path),
                                      header=None,
                                      error_bad_lines=False)  #
    Kernel_Trace_Data_List = Kernel_Trace_Data.values.tolist()
    pid = int(Kernel_Trace_Data_List[i][0])
    print(pid)
    return pid


if __name__ == '__main__':
    app_way = 'dy'
    stra = 'migration_cost'
    app_path = 'test1'
    app = 'dy_migration_cost_1'
    # app_plt(app)
    # plt_box_cs()
    # runtime ans migrate ans 和scm ans分别对应三种文件的分析，即runtime migrate 和scm
    file_name = sys.argv[1]
    app = sys.argv[2]
    ii = int(sys.argv[3])
    itemp = int(sys.argv[4])
    runtime_ans =  int(sys.argv[7])
    migrate_ans = 0
    eas_ans = 0
    util_ans = 0
    ddr_ans = int(sys.argv[5])
    task_migrate_ans =  int(sys.argv[6])
    l_mg_time = []
    l_mg_cycle = []
    on = 0
    main_thread = 0
    # 创建并清空文件内容
    clean_file()
    
    main_thread_pid = -1
    main_thread_list = []
    if (eas_ans):
        eas_plt()
    if(runtime_ans):
        runtime_plt()
    if(migrate_ans) :
        mg_time, mg_cycle = migrate_plt()
    if (task_migrate_ans):
        task_migrate()
    if(ddr_ans):
        ddr_read()
    if(util_ans):
        util_plt()



