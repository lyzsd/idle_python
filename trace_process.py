import os
import threading
import time as t
import subprocess
from xurui.util_to_freq import get_freq, analyze_freq
from shuheng.get_time import get_task_time
from shuheng.get_time import cpu_plt
from yuzhao.pmu_power_ans import pmu_read_plt
from yuzhao.txt_plt_2 import read_freq_runtime,battery_read,read_fps,pltscatter_freq,read_sys_freq_runtime
from yuzhao.ipc_compute import ipc_read_plt
from yuzhao.sugov_freq_ans import read_sugov_next_freq,read_cfs_cpu_selection
from yuzhao.boost_policy_ans import read_boost_next_freq
from yuzhao.idle_runtime_ans import idle_runtime_wakeup_trace_analysis,ddr_trace_read,idle_runtime_read
from yuzhao.idle_model import idle_model_process
from yuzhao import Constant
from yuzhao.idle_model_make import idle_model_make,idle_model_test,idle_model_read
def adb_shell(cmd):
    p = subprocess.getstatusoutput(cmd)
    return p


# ===== 自动化采集数据 =====
def auto_collect_data(file_name, app, ii, itemp):
    print("开启屏幕")
    #subprocess.call('all_bat\\open_screen.bat')

    print("开启应用")
    subprocess.call(r'all_bat\\open{}.bat'.format(app))

    print("开始trace")
    obj1 = subprocess.Popen(['adb', 'shell'], shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    obj1.stdin.write('cd /sys/kernel/tracing\n'.encode('utf-8'))
    obj1.stdin.write('echo 96000 > buffer_size_kb\n'.encode('utf-8'))
    obj1.stdin.write('echo 0 > tracing_on\n'.encode('utf-8'))
    # 公共
    obj1.stdin.write("echo 1 > /sys/kernel/tracing/events/sched/sched_stat_runtimes/enable\n".encode('utf-8'))
    # 育肇
    obj1.stdin.write("echo 1 > /sys/kernel/tracing/events/sched/sched_migrate_task/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 1 > /sys/kernel/tracing/events/sched/ddr_freq/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 1 > /sys/kernel/tracing/events/sched/sched_eas/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 1 > /sys/kernel/tracing/events/sched/wakeup_migrate/enable\n".encode('utf-8'))

    # 瞿蕊
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/sched/l3_stat/enable\n".encode('utf-8'))
    # 叔衡
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/thermal/perf_api/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/sched/pmu_data/enable\n".encode('utf-8'))
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/sched/sched_stat_runnable/enable\n".encode('utf-8'))
    # 王许睿
    obj1.stdin.write("echo 0 > /sys/kernel/tracing/events/power/get_my_util/enable\n".encode('utf-8'))
    # 结束
    obj1.stdin.write('echo 1 > tracing_on\n'.encode('utf-8'))
    obj1.stdin.write('exit\n'.encode('utf-8'))  # 重点，一定要执行exit
    obj1.communicate()



    print("执行脚本")
    # time.sleep(60)
    subprocess.call(r'all_bat\\do{}.bat'.format(app))

    print("关闭应用")
    subprocess.call(r'all_bat\closesolo.bat')
    #subprocess.call('all_bat\\close_screen.bat')

    print("结束trace")

    obj2 = subprocess.Popen(['adb', 'shell'], shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    obj2.stdin.write('cd /sys/kernel/tracing\n'.encode('utf-8'))
    obj2.stdin.write('echo 0 > tracing_on\n'.encode('utf-8'))
    obj2.stdin.write('exit\n'.encode('utf-8'))  # 重点，一定要执行exit
    obj2.communicate()

    # 王许睿
    #thread1.join()


    os.system(
        r"adb pull /sys/kernel/tracing/trace data_trace/{}/{}_{}_{}/trace.txt".format(file_name, app,
                                                                                                         ii, itemp))
    adb_shell('adb shell "echo 0 > /sys/kernel/tracing/trace"')

    # adb_shell(r"adb logcat -d -s jitter_data > {}/{}_{}/jitter_data.txt".format(file_name, ii, itemp))
    # adb_shell(r"adb logcat -d -s doframe_data > {}/{}_{}/doframe_data.txt".format(file_name, ii, itemp))


if __name__ == '__main__':
    app = '_tt'
    file_name = 'big_idle_model'
    coef  = 'eas_margin'
    default = 625
    print('===== 数据trace-开始 =====')
    small = []
    big = []
    super = []
    error_list = []

    for ii in range(0, 1):
        # 梁育肇功能 设置4个变量ddr_ans,runtime_ans,migrate_ans,分别表示 ddr 分析，migrate 分析以及runtime 分析
        itemp = ii
        ddr_ans = 0
        migrate_ans = 1
        runtime_ans = 0
        # 将数据从trace转化为csv
        # my_path = f"/{file_name}/{app}_{ii}_{itemp}/"
        # idle_model_make(file_name, app, ii, itemp)
        # error_rate = idle_model_test(file_name, app, ii, itemp)
        # error_list.append(error_rate)
        # idle_model_read(file_name, app, ii, itemp)
        ddr_trace_read(file_name, app, ii, itemp)
        idle_runtime_wakeup_trace_analysis(file_name, app, ii, itemp)
        idle_runtime_read(file_name, app, ii, itemp)
        l, b, s = read_freq_runtime(file_name, app, ii, itemp)
        small.append(l)
        big.append(b)
        super.append(s)
        pltscatter_freq(small, big, super, 'rate')
        # idle_model_process(file_name, app, ii, itemp)

        # # l, b, s = read_boost_next_freq(file_name, app, ii, itemp)
        # small.append(l[1])
        # big.append(b[1])
        # super.append(s[1])
        # pmu_read_plt(file_name, app, ii, itemp)
        #

        # l,b,s = read_sys_freq_runtime(file_name, app, ii, itemp)
        # print(l,b,s)
        # small.append(l)
        # big.append(b)
        # super.append(s)
        # print(battery_read(file_name, app, ii, itemp))
        # read_cfs_cpu_selection(file_name, app, ii, itemp)
        # print(read_fps(file_name, app, ii, itemp))
        # print('------')
        # print('功能：获取本次测试中所有任务的 running time 和 runnable time')
        # runtime_list, runnable_list, total_time = get_task_time(file_name, app, ii, itemp)
        # print('runtime_list:', runtime_list)
        # print('runnable_list:', runnable_list)

        # os.system(r"python yuzhao/data_to_csv.py {} {} {} {} {} {} {}".format(file_name, app, ii, itemp, ddr_ans, migrate_ans,runtime_ans))
        # os.system(r"python yuzhao/plt.py {} {} {} {} {} {} {}".format(file_name, app, ii, itemp, ddr_ans, migrate_ans,runtime_ans))

        # os.system(r"python yuzhao/txt_plt.py {} {} {} {} {} {} {}".format(file_name, app, ii, itemp, ddr_ans, migrate_ans,runtime_ans))
        # 分析txt 文件并绘图
        #   os.system(r"python yuzhao/txt_process2.0.py {} {} {} {} {} {} {}".format(file_name, app, ii, itemp, ddr_ans, migrate_ans, runtime_ans))

        #     r"python util_to_freq.py {} {} ".format('data_trace' + my_path + 'trace.txt', 'data_process' + my_path))
    # pltscatter_freq(small,big,super,'rate')
    print('===== 数据处理-结束 =====')
    print("模型误差率")
    print(error_list)
