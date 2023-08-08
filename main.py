import os
import threading
import time
import time as t
import subprocess
import pandas as pd
from xurui.util_to_freq import get_freq, analyze_freq
from shuheng.get_time import get_task_time
from shuheng.get_time import cpu_plt
def adb_shell(cmd):
    p = subprocess.getstatusoutput(cmd)
    return p
def adb_cat_battery(p_list,temp_list):
    # 电流
    I = int(adb_shell('adb shell "cat /sys/class/power_supply/battery/current_now"')[1])
    # 电压
    V = int(adb_shell('adb shell "cat /sys/class/power_supply/battery/voltage_now"')[1])
    # 功耗
    p = round(I * V / 1e9, 2)
    p_list.append(p)
    #print(p)
    #小核簇温度
    temp0 = int(adb_shell('adb shell "cat /sys/class/thermal/thermal_zone24/temp"')[1])
    #大核簇温度
    temp4 = int(adb_shell('adb shell "cat /sys/class/thermal/thermal_zone30/temp"')[1])
    #超大核温度
    temp7 = int(adb_shell('adb shell "cat /sys/class/thermal/thermal_zone33/temp"')[1])
    temp_list[0].append(temp0)
    temp_list[1].append(temp4)
    temp_list[2].append(temp7)

class SubThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.switch = True
        self.p_list = []
        self.temp_List = [[],[],[]]

    def run(self):
        while self.switch:
            adb_cat_battery(self.p_list,self.temp_List)
            t.sleep(0.5)

    def end(self):
        self.switch = False

    def get_p_list(self):
        return self.p_list
    def get_tmep_list(self):
        return self.temp_List
def getpackname(app):
    pack=""
    if(app=="dy"):
        pack="com.ss.android.ugc.aweme"
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
    if (app == "camera"):
        pack = "com.oneplus.camera"
    return pack
def get_topappthread_pid(str):

    # 获取前台app 包名以下的全部进程 com.ss.android.ugc.aweme
    result_new = subprocess.getstatusoutput('adb shell "ps -eo pid,args,psr|grep {}"'.format(str))[1]
    pid_list = []
    for i in range(len(result_new.split('\n'))):
        if(len(result_new.split('\n')[i-1].split())==3):
            pid_list.append(result_new.split('\n')[i-1].split())

    print(pid_list)
    #返回pid列表
    return pid_list
def find_pid_list(pid_list,pack):
    main_pid=-1
    for i in range(len(pid_list)):
        if pid_list[i][1] == pack :
            main_pid= pid_list[i][0]

    return main_pid
def taskset(cluster,pid):
    core="ff"

    if (cluster == 4):
        core="f0"
    if (cluster == 7):
        core="ff"

    os.system(r"adb shell taskset -p {} {}".format(core, pid))
# ===== 自动化采集数据 =====
def auto_collect_data(file_name, app, ii, itemp):
    print("开启屏幕")
    #subprocess.call('all_bat\\open_screen.bat')

    print("开启应用")
    #benchmark 绑核
    task_pid_list = [10281, 10289]
    for task_pid in task_pid_list:
        taskset(0, task_pid)
    subprocess.call(r'all_bat\\open{}.bat'.format(app))
    os.system('adb shell "echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset"')
    os.system('adb shell "echo 1 > /sys/devices/system/cpu/cpufreq/policy4/stats/reset"')
    os.system('adb shell "echo 1 > /sys/devices/system/cpu/cpufreq/policy7/stats/reset"')

    print("开始trace")
    obj1 = subprocess.Popen(['adb', 'shell'], shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    obj1.stdin.write('cd /sys/kernel/tracing\n'.encode('utf-8'))
    obj1.stdin.write('echo 192000 > buffer_size_kb\n'.encode('utf-8'))
    obj1.stdin.write('echo 0 > tracing_on\n'.encode('utf-8'))
    obj1.stdin.write('echo 0 > trace\n'.encode('utf-8'))
    # 公共
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/sched/sched_stat_runtimes/enable\n".encode('utf-8'))
    # 育肇
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/sched/sched_migrate_task/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/sched/ddr_freq/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/sched/sched_eas/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 1 > renable\n".encode('utf-8'))
    obj1.stdin.write("echo 1 > /sys/kernel/tracing/events/sched/pmu_power_model/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/sched/pmu_cpu_ipc/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/power/sugov_next_freqs/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/power/boost_policy/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 1 > /sys/kernel/tracing/events/sched/sched_cpu_selection/enable\n".encode('utf-8'))



    # 瞿蕊
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/sched/l3_stat/enable\n".encode('utf-8'))
    # 叔衡
    # 叔衡
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/irq/irq_handler_exit/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/irq/irq_handler_entry/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/sched/sched_enq_deq_task/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 1 > /sys/kernel/tracing/events/sched/sched_task_message/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/sched/sched_switch/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/sched/update_msg/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/sde/vsync_arrival/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/thermal/perf_api/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/sched/pmu_data/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 1 > /sys/kernel/tracing/events/power/cpu_idle/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/power/csh_cpu_idle/enable\n".encode('utf-8'))
    # 王许睿
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/power/get_my_util/enable\n".encode('utf-8'))
    # 远博
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/power/cpu_idle/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/sched/sched_wakeup/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/sched/sched_waking/enable\n".encode('utf-8'))
    # 结束
    obj1.stdin.write('echo 1 > tracing_on\n'.encode('utf-8'))
    obj1.stdin.write('exit\n'.encode('utf-8'))  # 重点，一定要执行exit
    obj1.communicate()


    #使用线程统计电池功耗
    thread = SubThread()
    thread.start()
    # time.sleep(60)
    #获取并保存主线程pid
    pack = getpackname(app)
    # pid_list = get_topappthread_pid(pack)  # 获取前台pid
    global pidd
    pidd = 0
    # pidd = find_pid_list(pid_list,pack)
    f2 = open(r'data_trace/{}/{}_{}_{}/main_thread_pid.txt'.format(file_name, app, ii, itemp), 'w')
    surfaceflinger = get_topappthread_pid('surfaceflinger')
    surfaceflinger = find_pid_list(surfaceflinger , 'surfaceflinger')
    print(pidd, file=f2)
    print(surfaceflinger, file=f2)
    f2.close()

    print("执行脚本")
    subprocess.call(r'all_bat\\do{}.bat'.format(app))
    # time.sleep(180)
    # os.system('adb shell "dumpsys gfxinfo  {}" > data_trace/{}/{}_{}_{}/FPS.txt'.format(pack,file_name, app, ii, itemp))

    #结束采集电池功耗
    thread.end()
    power_list = thread.get_p_list()
    temp_list = thread.get_tmep_list()
    print(len(power_list))
    print(len(temp_list[0]))
    print(len(temp_list[1]))
    print(len(temp_list[2]))
    list_lenght = min(len(power_list),len(temp_list[0]),len(temp_list[2]))
    battery_df =  pd.DataFrame(
        {'power': power_list[:list_lenght], 'temp0': temp_list[0][:list_lenght], 'temp4': temp_list[1][:list_lenght],
         'temp7': temp_list[2][:list_lenght]})
    mg_file_path = r'data_process/{}/{}_{}_{}/battery_file.csv'.format(file_name, app, ii, itemp)
    battery_df.to_csv(mg_file_path)

    print("关闭应用")
    subprocess.call('all_bat\\closesolo.bat')
    #subprocess.call('all_bat\\close_screen.bat')

    print("结束trace")

    f1 = open(r'data_trace/{}/{}_{}_{}/freq-running-time.txt'.format(file_name, app, ii, itemp), 'w')
    print("----------------")
    result1 = subprocess.getoutput('adb shell "cat /sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state"')
    print("----------------", file=f1)
    print(result1, file=f1)
    print(result1)

    print("----------------")
    result2 = subprocess.getoutput('adb shell "cat /sys/devices/system/cpu/cpufreq/policy4/stats/time_in_state"')
    print("----------------", file=f1)
    print(result2, file=f1)
    print(result2)
    print("----------------")
    result3 = subprocess.getoutput('adb shell "cat /sys/devices/system/cpu/cpufreq/policy7/stats/time_in_state"')
    print("----------------", file=f1)
    print(result3, file=f1)

    print(result3)
    f1.close()

    obj2 = subprocess.Popen(['adb', 'shell'], shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    obj2.stdin.write('cd /sys/kernel/tracing\n'.encode('utf-8'))
    obj2.stdin.write('echo 0 > tracing_on\n'.encode('utf-8'))
    obj2.stdin.write('exit\n'.encode('utf-8'))  # 重点，一定要执行exit
    obj2.communicate()

    # 王许睿
    #thread1.join()

    os.system(
         f'adb shell "cat /sys/kernel/tracing/trace | grep pmu_power_model" > data_trace/{file_name}/{app}_{ii}_{itemp}/pmu_power_model.txt')
    os.system(
        f'adb shell "cat /sys/kernel/tracing/trace | grep pmu_cpu_ipc" > data_trace/{file_name}/{app}_{ii}_{itemp}/pmu_cpu_ipc.txt')

    os.system(
        r"adb pull /sys/kernel/tracing/trace data_trace/{}/{}_{}_{}/trace.txt".format(file_name, app,
                                                                                                         ii, itemp))

    adb_shell('adb shell "echo 0 > /sys/kernel/tracing/trace"')
    time.sleep(60)
    # adb_shell(r"adb logcat -d -s jitter_data > {}/{}_{}/jitter_data.txt".format(file_name, ii, itemp))
    # adb_shell(r"adb logcat -d -s doframe_data > {}/{}_{}/doframe_data.txt".format(file_name, ii, itemp))


if __name__ == '__main__':
    app = '_tt'
    file_name = 'dynamic_eas_test'
    coef  = 'eas_margin'
    default = 625
    print('===== 数据trace-开始 =====')
    # print("开启应用")
    # subprocess.call(r'all_bat\\open{}.bat'.format(app))
    # print("执行脚本")
    # subprocess.call(r'all_bat\\do{}.bat'.format(app))
    # print("关闭应用")
    # subprocess.call(r'all_bat\closesolo.bat')
    #time.sleep(60)
    for ii in range(0, 6):
        '''
        ii与itemp用处：可以是第ii次测试第itemp轮，也可以是第ii次测试对应参数调节itemp
        如根据itemp调节参数task_need_pid：
            os.system('adb shell "echo itemp > /proc/sys/kernel/task_need_pid\n"')
        '''
        itemp = ii % 12
        if(ii%2 == 0):
            os.system('adb shell "echo {} > /proc/sys/kernel/{}"'.format(625, coef))
            os.system('adb shell "echo {} > /proc/sys/kernel/opp_switch"'.format(0))
        else:
            os.system('adb shell "echo {} > /proc/sys/kernel/{}"'.format(250, coef))
            os.system('adb shell "echo {} > /proc/sys/kernel/opp_switch"'.format(1))

        os.system('adb shell "echo {} > /proc/sys/kernel/opp_type"'.format(2))
        # trace数据所在文件夹举例：trace_data/test/0_1
        print(r'创建本次测试中trace数据文件夹：data_trace/{}/{}_{}_{}'.format(file_name, app, ii, itemp))
        folder_path = r'data_trace/{}/{}_{}_{}'.format(file_name, app, ii, itemp)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        '''
        将处理后的数据放在如：data_process/test/0_1
        '''
        print(r'创建本次测试中数据文件夹：data_process/{}/{}_{}_{}'.format(file_name, app, ii, itemp))
        folder_path = r'data_process/{}/{}_{}_{}'.format(file_name, app, ii, itemp)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        '''
        将处理后的图片放在如：data_plt/test/0_1
        图片命名格式： {单位}_{信息}_{app}.png
            1. 以 CPU 为单位
                如果是一张图展现所有 CPU 信息的：'data_plt/{}/{}_{}/cpu_all_{信息}_{app}.png'
                如果是一张图展现某个 CPU 信息的：'data_plt/{}/{}_{}/cpu_{CPU号}_{信息}_{app}.png'
            2. 以 cluster 为单位
                如果是一张图展现所有 cluster 信息的：'data_plt/{}/{}_{}/cluster_all_{信息}_{app}.png'
                如果是一张图展现某个 cluster 信息的：'data_plt/{}/{}_{}/cluster_{cluster号}_{信息}_{app}.png'
            3. 以 task 为单位
                如果是一张图展现所有 cluster 信息的：'data_plt/{}/{}_{}/task_all_{信息}_{app}.png'
                如果是一张图展现某个 cluster 信息的：'data_plt/{}/{}_{}/task_{task名}_{信息}_{app}.png'
            ……
        '''
        print(r'创建本次测试中图像文件夹：data_plt/{}/{}_{}_{}'.format(file_name, app, ii, itemp))
        folder_path = r'data_plt/{}/{}_{}_{}'.format(file_name, app, ii, itemp)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # trace函数
        auto_collect_data(file_name, app, ii, itemp)
    print('===== 数据trace-结束 =====')

    print('===== 数据trace-开始 =====')

    for ii in range(0, 6):
        # 梁育肇功能 设置4个变量ddr_ans,runtime_ans,migrate_ans,分别表示 ddr 分析，migrate 分析以及runtime 分析
        itemp = ii
        ddr_ans = 0
        migrate_ans = 0
        runtime_ans = 0


        # 叔衡功能
        # print('------')
        # print('功能：获取本次测试中所有任务的 running time 和 runnable time')
        # runtime_list, runnable_list, total_time = get_task_time(file_name, app, ii, itemp)
        # print('runtime_list:', runtime_list)
        # print('runnable_list:', runnable_list)
        # os.system(
        #     r"python yuzhao/data_to_csv.py {} {} {} {} {} {} {}".format(file_name, app, ii, itemp, ddr_ans, migrate_ans, runtime_ans))
        # os.system(r"python yuzhao/plt.py {} {} {} {} {} {} {}".format(file_name, app, ii, itemp, ddr_ans, migrate_ans,
        #                                                               runtime_ans))

        # print('------')
        # print('功能：获取本次测试中 per CPU 的 loading 占比图与 runnable 数据图')
        # cpu_plt(file_name, app, ii, itemp, runtime_list, runnable_list, total_time)
        # 王许睿功能
        #
        # my_path = f"/{file_name}/{app}_{ii}_{itemp}/"
        # get_freq('data_trace' + my_path + 'trace.txt', 'data_process' + my_path)
        #
        # # 将数据从trace转化为csv
        # os.system(r"python yuzhao/data_to_csv.py {} {} {} {} {} {} {}".format(file_name, app, ii, itemp, ddr_ans, migrate_ans,
        #                                                               runtime_ans))
        # os.system(r"python yuzhao/plt.py {} {} {} {} {} {} {}".format(file_name, app, ii, itemp, ddr_ans, migrate_ans,
        #                                                               runtime_ans))
        # 分析txt 文件并绘图
        #   os.system(r"python yuzhao/txt_process2.0.py {} {} {} {} {} {} {}".format(file_name, app, ii, itemp, ddr_ans, migrate_ans, runtime_ans))

        #     r"python util_to_freq.py {} {} ".format('data_trace' + my_path + 'trace.txt', 'data_process' + my_path))

    print('===== 数据处理-结束 =====')

