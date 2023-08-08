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

F = [
    [300000, 403200, 499200, 595200, 691200, 806400, 902400, 998400, 1094400, 1209600, 1305600, 1401600, 1497600,
     1612800, 1708800, 1804800],
    [710400, 844800, 960000, 1075200, 1209600, 1324800, 1440000, 1555200, 1670400, 1766400, 1881600, 1996800, 2112000,
     2227200, 2342400, 2419200],
    [844800, 960000, 1075200, 1190400, 1305600, 1420800, 1555200, 1670400, 1785600, 1900800, 2035200, 2150400, 2265600,
     2380800, 2496000, 2592000, 2688000, 2764800, 2841600]
]
ddr_label = [312, 365, 477 , 585, 643, 982 ,1302, 1467, 1827 ,2216, 5000]
small_cap = [0] * 16
big_cap = [0] * 16
super_cap = [0] * 19
for i in range(16):
    small_cap[i] = F[0][i] * 325 / F[0][15]
    big_cap[i] = F[1][i] * 828 / F[1][15]
for i in range(19):
    super_cap[i] = F[2][i] * 1024 / F[2][18]
import matplotlib.pyplot as plt  # matplotlib数据可视化神器
def pltscatter_freq(l1,l2,l3,y):
    plt.figure(figsize=(16, 10))
    colors = ['blue','red','orange','green','pink','#008B8B']
    for i in range(len(l1)):

        plt.plot(small_cap, l1[i], marker='o', markerfacecolor='white', linestyle='-',color=colors[i], label='small_{}'.format(i))
    for i in range(len(l2)):

        plt.plot(big_cap, l2[i], marker='x', markerfacecolor='white', linestyle='-',color=colors[i], label='big_{}'.format(i))
    for i in range(len(l3)):
        plt.plot(super_cap, l3[i], marker='o', markerfacecolor='white', linestyle='--',color=colors[i], label='super_{}'.format(i))
    plt.legend()
    plt.xlabel('capacity')
    plt.ylabel(y)
    if( main_thread == 1) :
        path = r'data_plt/{}/{}_{}_{}/main_freq.png'.format(file_name, app, ii, itemp)
    else :
        path = r'data_plt/{}/{}_{}_{}/freq_{}.png'.format(file_name, app, ii, itemp,y)
    plt.savefig(path)
    plt.show()
def pltscatter_ddr_freq(l1):
    plt.figure(figsize=(16, 10))
    colors = ['blue','red','orange','green','pink','#008B8B']
    for i in range(len(ddr_label)):
        ddr_label[i] = int(1e12/ddr_label[i])
    for i in range(len(l1)):

        plt.plot(ddr_label, l1[i], marker='o', markerfacecolor='white', linestyle='-',color=colors[i], label='ddr_{}'.format(i))

    plt.legend()
    plt.xlabel('ddr_freq')
    plt.ylabel('count')

    path =  r'data_plt/{}/{}_{}_{}/ddr.png'.format(file_name, app, ii, itemp)
    plt.savefig(path)
    plt.show()

def pltscatter_eas_rate(l1):
    colors = ['blue', 'red', 'orange', 'green', 'pink','#008B8B']
    x_list = [0] * 20

    for i in range(20):
        x_list[i] = 0.006 * i
    for i in range(len(l1)):
        plt.plot(x_list, l1[i], marker='o', markerfacecolor='white', linestyle='-', color=colors[i],
                 label='ddr_{}'.format(i))

    plt.legend()
    plt.xlabel('eas margin')
    plt.ylabel('migrate rate')
    plt.show()

def pltscatter_energy(l1,l2,l3,y):
    plt.figure(figsize=(16, 10))
    plt.plot(range(len(l1[0])), l1[0], marker='o',markerfacecolor='white', linestyle='-',color='blue',label='small_fix')
    plt.plot(range(len(l1[0])), l2[0], marker='o', markerfacecolor='white',linestyle='-',color='g',label='big_fix')
    plt.plot(range(len(l1[0])), l3[0], marker='o', markerfacecolor='white',linestyle='-',color='r',label='super_fix')
    plt.plot(range(len(l1[0])), l1[1], marker='o', linestyle='--', color='blue', label='small_dyn')
    plt.plot(range(len(l1[0])), l2[1], marker='o', linestyle='--', color='g', label='big_dyn')
    plt.plot(range(len(l1[0])), l3[1], marker='o', linestyle='--', color='r', label='super_dyn')

    plt.legend()
    plt.xlabel('time')
    plt.ylabel(y)

    if (main_thread == 1):
        path = 'E:\\benchmark\pmu\\{}\\{}\\{}\\main_energy.jpg'.format(app_path, stra,pic_save_app)
    else:
        path = 'E:\\benchmark\pmu\\{}\\{}\\{}\\total_energy.jpg'.format(app_path, stra,pic_save_app)
    plt.savefig(path )
    plt.show()

def migrate_bar(y):
    x =range(6)
    # 计算NO和NO2
    y1 = y[0] /(y[1]+y[2]+y[3])
    y2 = y[1] /(y[1]+y[2]+y[3])
    y3 = y[2] /(y[1]+y[2]+y[3])
    y4 = y[3] /(y[1]+y[2]+y[3])


    plt.bar(x, y1, width=0.4, label='no migrate', color='#f9766e', edgecolor='grey', zorder=5)
    plt.bar(x, y2, width=0.4, bottom=y1, label='from small', color='#00bfc4', edgecolor='grey', zorder=5)
    plt.bar(x, y3, width=0.4, bottom=y2, label='from big', color='red', edgecolor='grey', zorder=5)
    plt.bar(x, y4, width=0.4, bottom=y3, label='from super', color='green', edgecolor='grey', zorder=5)
    plt.tick_params(axis='x', length=0)
    plt.xlabel('Site', fontsize=12)
    plt.ylabel('Concentration', fontsize=12)
    plt.ylim(0, 1.01)
    plt.yticks(np.arange(0, 1.2, 0.2), [f'{i}%' for i in range(0, 120, 20)])
    plt.grid(axis='y', alpha=0.5, ls='--')
    # 添加图例，将图例移至图外
    plt.legend(frameon=False, bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    #plt.savefig('bar2.png', dpi=600)
    plt.show()

def read_energy(app):
    Kernel_Trace_Data = pd.read_table(r'data/energy/energy_{}.txt'.format(app),
                                      header=None,
                                      error_bad_lines=False)  #
    prev_energy=[0]*6
    new_energy=[0]*6

    Kernel_Trace_Data_List = Kernel_Trace_Data.values.tolist()
    for i in range(len(Kernel_Trace_Data_List)):
        Kernel_Trace_Data_List[i] = Kernel_Trace_Data_List[i][0].split()

    for i in range(1,len(Kernel_Trace_Data_List)):
        for j in range(len(Kernel_Trace_Data_List[i])):
            Kernel_Trace_Data_List[i][j] = float(Kernel_Trace_Data_List[i][j])
            if (i % 2 ==1):
                prev_energy[j] = prev_energy[j]+Kernel_Trace_Data_List[i][j]
            else:
                new_energy[j] = new_energy[j]+Kernel_Trace_Data_List[i][j]
    for i in range(len(prev_energy)):
        prev_energy[i]=prev_energy[i]/((len(Kernel_Trace_Data_List)-1)/2)
        new_energy[i]=new_energy[i]/((len(Kernel_Trace_Data_List)-1)/2)

    print(prev_energy)
    print(new_energy)

def read_runtime(app):
    Kernel_Trace_Data = pd.read_table(r'data/runtime/runtime_{}.txt'.format(app),
                                      header=None,
                                      error_bad_lines=False)  #
    prev_energy=[0]*8
    new_energy=[0]*8

    Kernel_Trace_Data_List = Kernel_Trace_Data.values.tolist()
    for i in range(len(Kernel_Trace_Data_List)):
        Kernel_Trace_Data_List[i] = Kernel_Trace_Data_List[i][0].split()

    for i in range(1,len(Kernel_Trace_Data_List)):
        for j in range(len(Kernel_Trace_Data_List[i])):
            Kernel_Trace_Data_List[i][j] = float(Kernel_Trace_Data_List[i][j])
            if (i % 2 ==1):
                prev_energy[j] = prev_energy[j]+Kernel_Trace_Data_List[i][j]
            else:
                new_energy[j] = new_energy[j]+Kernel_Trace_Data_List[i][j]
    for i in range(len(prev_energy)):
        prev_energy[i]=prev_energy[i]/((len(Kernel_Trace_Data_List)-1)/2)
        new_energy[i]=new_energy[i]/((len(Kernel_Trace_Data_List)-1)/2)

    print(prev_energy)
    print(new_energy)
def read_freq_runtime(app):

    Kernel_Trace_Data = pd.read_table(r'data_trace/{}/{}_{}_{}/freq_runtime_freq.txt'.format(file_name, app, ii, itemp),
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
            if(i == 0):
                power = little_power_data[0][j]
                small_runtime_list[j] = Kernel_Trace_Data_List[i][j]
            elif(i == 1):
                power = big_power_data[1][j]
                big_runtime_list[j] = Kernel_Trace_Data_List[i][j]
            else:
                power = super_power_data[2][j]
                super_runtime_list[j] = Kernel_Trace_Data_List[i][j]

            power_list[i] = power_list[i] + power * Kernel_Trace_Data_List[i][j]
    power_list[3] = power_list[0] + power_list[1] + power_list[2]
    print(power_list)
    for i in range(16):
        small_runtime_list[i] = small_runtime_list[i]/np.sum(small_runtime_list)
        big_runtime_list[i] = big_runtime_list[i]/np.sum(big_runtime_list)
    for i in range(19):
        super_runtime_list[i] = super_runtime_list[i]/np.sum(super_runtime_list)
    return small_runtime_list,big_runtime_list,super_runtime_list




def read_runtime_freq(app,cluster):
    if(main_thread) :
        Kernel_Trace_Data = pd.read_table(r'data_trace/{}/{}_{}_{}/runtime_{}_freq.txt'.format(file_name, app, ii, itemp,cluster),
                                          header=None,
                                          error_bad_lines=False)  #
    else :
        Kernel_Trace_Data = pd.read_table(r'data_trace/{}/{}_{}_{}/runtime_{}_freq.txt'.format(file_name, app, ii, itemp,cluster),
                                          header=None,
                                          error_bad_lines=False)  #


    power_table = pd.read_table(r'power_table_pmu/table_{}.txt'.format(app_table_path),
                                      header=None,
                                      error_bad_lines=False)  #

    power_table=power_table.values.tolist()

    for i in range(len(power_table)):
        power_table[i] = power_table[i][0].split()
    for i in range(len(power_table)):
        for j in range(len(power_table[i])):
            power_table[i][j]=float(power_table[i][j])
    list_runtime_freq = []
    list_util = []
    list_energy = []
    list_percent = []
    if(cluster == 'super'):
        energy_list = [0] * 19
        runtime_list = [0] * 19
        util_list = [0] * 19
        dyn_power_table = power_table[2]
        opp = super_power_data
        length = 19
        #print(dyn_power_table)
    else :

        energy_list = [0] * 16
        runtime_list = [0] * 16
        util_list = [0] * 16
        length = 16
        if(cluster == 'small') :
            dyn_power_table = power_table[0]
            opp = little_power_data
        else:
            dyn_power_table = power_table[1]
            opp = big_power_data



    Kernel_Trace_Data_List = Kernel_Trace_Data.values.tolist()
    for i in range(len(Kernel_Trace_Data_List)):
        Kernel_Trace_Data_List[i] = Kernel_Trace_Data_List[i][0].split()

    for i in range(0,len(Kernel_Trace_Data_List)):
        energy = 0
        opp_energy = 0
        if(i == 0):
            for j in range(length):
                Kernel_Trace_Data_List[i][j] = float(Kernel_Trace_Data_List[i][j])
                energy_list[j] =  Kernel_Trace_Data_List[i][j] * dyn_power_table[j]
                runtime_list[j] = Kernel_Trace_Data_List[i][j]
            list_energy.append(energy_list)
            rt_list = []
            for rt in runtime_list:
                if (cluster == 'small'):
                    rt = rt / 4
                if (cluster == 'big'):
                    rt = rt / 3
                rt_list.append(rt)
            for rt in runtime_list:
                percent = rt/np.sum(runtime_list)
                list_percent.append(percent)
            list_runtime_freq.append(rt_list)
        if(i == 1):
            for j in range(length):
                Kernel_Trace_Data_List[i][j] = float(Kernel_Trace_Data_List[i][j])

                util_list[j] = Kernel_Trace_Data_List[i][j]

            list_util.append(util_list)




    return list_runtime_freq,list_energy,list_util,list_percent



def compute_total_time():
    Kernel_Trace_Data = pd.read_table(r'data_trace/{}/{}_{}_{}/energy.txt'.format(file_name, app, ii, itemp),
                                      header=None,
                                      error_bad_lines=False)  #
    Kernel_Trace_Data_List = Kernel_Trace_Data.values.tolist()
    for i in range(len(Kernel_Trace_Data_List)):
        Kernel_Trace_Data_List[i] = Kernel_Trace_Data_List[i][0].split()

    for i in range(1, len(Kernel_Trace_Data_List)):
        energy = 0
        for j in range(len(Kernel_Trace_Data_List[i])):
            Kernel_Trace_Data_List[i][j] = float(Kernel_Trace_Data_List[i][j])
    power = Kernel_Trace_Data_List[2][1]
    energy = Kernel_Trace_Data_List[2][2]

    return energy/power

def scatter_plot(data1,data2,migrate_file):
    # 创建画布和子图
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制第一组数据的箱线图和散点图
    ax.boxplot(data1[0], positions=[0], vert=True,boxprops={'color': 'red'})
    # 绘制第二组数据的箱线图和散点图
    ax.boxplot(data2[0], positions=[1], vert=True,boxprops={'color': 'blue'})

    # 绘制第三组数据的箱线图和散点图
    ax.boxplot(data1[1], positions=[2], vert=True,boxprops={'color': 'red'})

    ax.boxplot(data2[1], positions=[3], vert=True,boxprops={'color': 'blue'})

    # 绘制第二组数据的箱线图和散点图
    ax.boxplot(data1[2], positions=[4], vert=True,boxprops={'color': 'red'})
    # 绘制第三组数据的箱线图和散点图
    ax.boxplot(data2[2], positions=[5], vert=True,boxprops={'color': 'blue'})


    # 设置坐标轴和标题
    ax.set_xticks([0,1, 2, 3, 4, 5])
    ax.set_xticklabels(['fix small', 'dyn small', 'fix big', 'dyn big','fix super', 'dyn super'])
    ax.set_ylabel('Count rate')
    ax.set_title('Multiple Box plot with scatter')

    if(main_thread == 1):
        path =r'data_plt/{}/{}_{}_{}/main_{}.png'.format(file_name, app, ii, itemp,migrate_file)

    else:
        path = r'data_plt/{}/{}_{}_{}/{}.png'.format(file_name, app, ii, itemp,migrate_file)
    plt.savefig(path)

    # 显示图形
    plt.show()

def battery_plot(data1):
    # 创建画布和子图
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制第一组数据的箱线图和散点图
    ax.boxplot(data1[0], positions=[0], vert=True,boxprops={'color': 'red'},showmeans=True)
    # 绘制第二组数据的箱线图和散点图
    ax.boxplot(data1[1], positions=[1], vert=True,boxprops={'color': 'blue'},showmeans=True)

    # 绘制第三组数据的箱线图和散点图
    ax.boxplot(data1[2], positions=[2], vert=True,boxprops={'color': 'red'},showmeans=True)

    ax.boxplot(data1[3], positions=[3], vert=True,boxprops={'color': 'blue'},showmeans=True)

    # 绘制第二组数据的箱线图和散点图
    ax.boxplot(data1[4], positions=[4], vert=True,boxprops={'color': 'red'},showmeans=True)
    # 绘制第三组数据的箱线图和散点图
    ax.boxplot(data1[5], positions=[5], vert=True,boxprops={'color': 'blue'},showmeans=True)


    # 设置坐标轴和标题
    ax.set_xticks([0,1, 2, 3, 4, 5])
    ax.set_xticklabels(['0%', '1.5%', '3.0%', '4.5%','6.0%', '7.5%'])
    ax.set_ylabel('power(mw)')
    ax.set_title('Multiple Box plot with scatter')


    path = r'data_plt/{}/{}_{}_{}/bttery.png'.format(file_name, app, ii, itemp)
    plt.savefig(path)

    # 显示图形
    plt.show()

def migrate_ans(migrate_file):
    prev_small=[]
    prev_big=[]
    prev_super =[]
    new_small=[]
    new_big=[]
    new_super = []
    prev_list=[]
    new_list=[]

    prev_mig = []
    new_mig = []
    list_to_hot = []
    if(main_thread):
        Kernel_Trace_Data = pd.read_table(r'data_trace/{}/{}_{}_{}/{}.txt'.format(file_name, app, ii, itemp,migrate_file),
                                          header=None,
                                          error_bad_lines=False)  #
    else :
        Kernel_Trace_Data = pd.read_table(r'data_trace/{}/{}_{}_{}/{}.txt'.format(file_name, app, ii, itemp,migrate_file),
                                          header=None,
                                          error_bad_lines=False)  #

    Kernel_List = Kernel_Trace_Data.values.tolist()
    for i in range(len(Kernel_List)):
        Kernel_List[i] = Kernel_List[i][0].split()

    for i in range(1, len(Kernel_List)):
        energy = 0
        for j in range(len(Kernel_List[i])):
            Kernel_List[i][j] = int(Kernel_List[i][j])
        total_migrate = 0
        for tt in range(len(Kernel_List[i])):
            total_migrate = total_migrate + Kernel_List[i][tt]

        if( i % 2 ==1):
            prev_mig.append(Kernel_List[i])
            prev_small.append((Kernel_List[i][0]+Kernel_List[i][1]+Kernel_List[i][2]+Kernel_List[i][3] )/total_migrate)
            prev_big.append((Kernel_List[i][4] + Kernel_List[i][5] + Kernel_List[i][6] + Kernel_List[i][7]) /total_migrate)
            prev_super.append((Kernel_List[i][8] + Kernel_List[i][9] + Kernel_List[i][10]) / total_migrate)
        if (i % 2 == 0):
            new_mig.append(Kernel_List[i])
            new_small.append((Kernel_List[i][0] + Kernel_List[i][1] + Kernel_List[i][2] + Kernel_List[i][3]) / total_migrate)
            new_big.append((Kernel_List[i][4] + Kernel_List[i][5] + Kernel_List[i][6] + Kernel_List[i][7] )/total_migrate)
            new_super.append((Kernel_List[i][8] + Kernel_List[i][9] + Kernel_List[i][10])/total_migrate)
        list_to_hot.append(Kernel_List[i])

    prev_list=[prev_small,prev_big,prev_super]
    new_list=[new_small,new_big,new_super]
    print(prev_list)
    print(new_list)

    scatter_plot(prev_list,new_list,migrate_file)


    hot_migrate(list_to_hot,migrate_file)

def runtime_rank_ans():
    Kernel_Trace_Data = pd.read_table(r'data_trace/{}/{}_{}_{}/runtime_top_20.txt'.format(file_name, app, ii, itemp),
                                      header=None,
                                      error_bad_lines=False)  #
    Kernel_List = Kernel_Trace_Data.values.tolist()
    for i in range(len(Kernel_List)):
        Kernel_List[i] = Kernel_List[i][0].split(']')

    for i in range(len(Kernel_List)):
        label = []
        x= []
        for j in range(20):
            string = Kernel_List[i][j].split(',')

            x.append(float(string[1]))
            new_label = string[0].replace("'", "")
            new_label = new_label.replace("[", "")
            label.append(new_label)
        idx = np.arange(len(x))
        color = 'blue'
        plt.barh(idx, x, color=color)
        plt.yticks(idx + 0.4, label)
        plt.grid(axis='x')
        plt.xlabel('Runtime(s)')
        plt.ylabel('Task name')
        plt.title('Top 20 runtime task')
        plt.show()



def ddr_freq_ans():
    Kernel_Trace_Data = pd.read_table(r'data_trace/{}/{}_{}_{}/ddr_freq.txt'.format(file_name, app, ii, itemp),
                                      header=None,
                                      error_bad_lines=False)  #
    Kernel_List = Kernel_Trace_Data.values.tolist()
    for i in range(len(Kernel_List)):
        Kernel_List[i] = Kernel_List[i][0].split()

    for i in range(0, len(Kernel_List)):
        ddr_freq = []
        for j in range(len(Kernel_List[i])):
            Kernel_List[i][j] = int(Kernel_List[i][j])
            ddr_freq.append(Kernel_List[i][j])


    return ddr_freq




def util_freq_ans():
    data1 = pd.read_csv(r'data_process/{}/{}_{}_{}/small_util_freq.csv'.format(file_name, app, ii, itemp))
    data2 = pd.read_csv(r'data_process/{}/{}_{}_{}/big_util_freq.csv'.format(file_name, app, ii, itemp))
    small = []
    big = []
    small.append(list(data1['cpu0']))
    small.append(list(data1['cpu1']))
    small.append(list(data1['cpu2']))
    small.append(list(data1['cpu3']))

    big.append(list(data2['cpu4']))
    big.append(list(data2['cpu5']))
    big.append(list(data2['cpu6']))
    return small, big


def eas_ans():

    Kernel_Trace_Data = pd.read_table(r'data_trace/{}/{}_{}_{}/eas_ans.txt'.format(file_name, app, ii, itemp),
                                          header=None,
                                          error_bad_lines=False)  #

    migrate_list = [0] *20


    Kernel_List = Kernel_Trace_Data.values.tolist()
    for i in range(len(Kernel_List)):
        Kernel_List[i] = Kernel_List[i][0].split()

    for i in range(0, len(Kernel_List)):

        for j in range(20):
            migrate_list[j] = float(Kernel_List[i][j])


    return migrate_list



    plt.figure(figsize=(16, 10))
    plt.plot(x_list, prev_rate, marker='o', markerfacecolor='white', linestyle='-', color='blue',
             label='fix power table')
    plt.plot(x_list, new_rate, marker='o', markerfacecolor='white', linestyle='-', color='g', label='dyn power table')
    plt.legend()
    plt.xlabel('eas_margin')
    plt.ylabel('rate')
    plt.savefig(path + 'rate.jpg')

    plt.show()

def battery_read():
    data1 = pd.read_csv(r'data_process/{}/{}_{}_{}/battery_file.csv'.format(file_name, app, ii, itemp))
    power = data1['power']

    return list(power)

def hot_migrate(data_content, migrate_file):
    sns.set_theme(style='darkgrid')  # 图形主题
    # pd.options.display.notebook_repr_html = False  # 表格显示
    params = {
        # 'font.family': 'Times New Roman',
        'font.sans-serif': 'SimHei',  # 用来正常显示中文标签
        'axes.unicode_minus': False,  # 用来正常显示负号
        'font.size': 3,
        'legend.fontsize': 'x-large',
        'figure.figsize': (4, 6),
        'axes.labelsize': 5,
        'axes.titlesize': 11,  # 没有设置title
        'xtick.labelsize': 3,  # 下方也有设置，此处不生效
        'ytick.labelsize': 4,
        'figure.dpi': 300
    }
    sns.set(palette="muted", color_codes=True)  # seaborn样式
    sns.set(font='SimHei', font_scale=1)  # 解决Seaborn中文显示问题
    plt.rcParams.update(params)

    fig = plt.figure()
    fig.subplots_adjust(left=0.2, bottom=0.12, right=0.96, top=0.94, wspace=0.1, hspace=0.1)  # 调整子图间距按百分比

    # 读数据

    '''
    5.390403587, 5.82768415, 5.620938661
    4.238484797, 4.716008071, 4.510990787
    7.470676724, 7.59863963, 15.55063237
    6.216723255, 6.71095276, 9.737460414  
    6.325608073, 6.471562379, 6.527632537
    '''
    # 获取第一行和最后一列的数据，用于作为坐标轴刻度
    row_labels = []
    for i in range(len(data_content)):
        row_labels.append('test_{}'.format(i))

    col_labels = ['small no migrate','migrate in small ','small from big','small from super','big no migrate',
                  'big from small','migrate in big','big from super',
                  'super no migrate','super from small','super from big']

    # 提取数据内容

    # 列表转置

    # 将数据转化为13行8列(reshape)，并生成数据框
    df = pd.DataFrame(data_content, columns=col_labels, index=row_labels)
    # df = df.transpose(True)
    df = df.astype(float)

    sns.heatmap(df,
                annot=True,  # 显示数据
                center=0.5,  # 居中
                fmt='.2f',  # 只显示两位小数
                linewidth=0.2,  # 设置每个框线的宽度
                linecolor='blue',  # 设置间距线的颜色
                # vmin=0, vmax=1,  # 设置数值最小值和最大值
                xticklabels=True, yticklabels=True,  # 显示x轴和y轴
                square=True,  # 每个方格都是正方形
                cbar=True,  # 绘制颜色条
                cmap='hot_r',  # 设置热力图颜色
                )
    plt.title("迁核分布")
    plt.xlabel("迁核类型")
    plt.ylabel("实验组")
    plt.show()  # 显示图片,这个可以方便后面自动关闭
    path = r'data_plt/{}/{}_{}_{}/hot_{}.png'.format(file_name, app, ii, itemp, migrate_file)
    plt.savefig(path)

if __name__ == '__main__':
    app_table_path = 'dy'
    file_name = sys.argv[1]
    app = sys.argv[2]
    ii = int(sys.argv[3])
    itemp = int(sys.argv[4])
    # read_energy(app)
    main_thread = 0
    lenght = 18

    # read_runtime(app)
    #total_time = compute_total_time()
    #print(total_time)
    small =[]
    big = []
    super = []
    #百分比图片
    small_p = []
    big_p = []
    super_p = []
    small_u = []
    big_u = []
    super_u = []
    small_en = []
    big_en = []
    super_en = []
    eas = []
    l_power = []
    ddr_freq_list = []
    for ii in range(6) :
        itemp = ii
        small_t, small_e, small_util, small_pp = read_runtime_freq(app, 'small')
        big_t, big_e, big_util, big_pp = read_runtime_freq(app, 'big')
        super_t, super_e, super_util, super_pp = read_runtime_freq(app, 'super')
        eas_m = eas_ans()
        power = battery_read()
        ddr_freq_list.append(ddr_freq_ans())
        #freq_util_small,freq_util_big = util_freq_ans()
        #pltscatter_freq(freq_util_small, freq_util_big, [[0]*19], 'count')
        l_power.append(power)
        eas.append(eas_m)
        print(small_t[0])
        small .append(small_t[0])
        small_u.append(small_util[0])
        small_en.append(small_e[0])
        big.append(big_t[0])
        big_u.append(big_util[0])
        big_en.append(big_e[0])
        super.append(super_t[0])
        super_u.append(super_util[0])
        super_en.append(super_e[0])
        small_p.append(small_pp)
        big_p.append(big_pp)
        super_p.append(super_pp)
    #print(small[0])
    battery_plot(l_power)
    pltscatter_ddr_freq(ddr_freq_list)
    pltscatter_freq(small, big,super, 'runtime')
    pltscatter_freq(small_p , big_p, super_p, 'percent')
    pltscatter_freq(small_u, big_u, super_u, 'cpu_util')
    pltscatter_freq(small_en, big_en, super_en, 'energy(j)')

    pltscatter_eas_rate(eas)
    #eas_ans()
    #pltscatter_energy(small_e, big_e, super_e, 'energy')

    runtime_rank_ans()
    migrate_ans('migrate')
    migrate_ans('task_migrate')

