#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/workqueue.h>
#include <linux/clk.h>
#include <linux/xmu_kernel.h>
//added by lyz
#include <trace/events/sched.h>
MODULE_DESCRIPTION("A kernel module that creates and schedules a periodic task");
MODULE_AUTHOR("lyz");
MODULE_LICENSE("GPL");
#ifdef LYZ_MOUDLE_PMU
static struct workqueue_struct *my_wq; // the workqueue
static struct delayed_work my_work; // the work item
static int counter = 0; // a counter to keep track of the task execution
static int pmu_inited = 0;
static int over_count = 0;
// the function that will be executed by the work item
#include <linux/kernel.h>                                                      
#include <linux/module.h>                                                      
#include <asm/smp.h>
#include <linux/delay.h>
#include <linux/kthread.h>

#include <linux/perf_event.h>
#include <linux/time64.h>
#include <linux/delay.h>
#include <linux/prefetch.h>
#include <linux/limits.h>
#include <linux/perf/arm_pmu.h>
//#include <linux/thermal_core.h>
#include <linux/thermal.h>

// 定义节点控制

#include <linux/kobject.h>  //zwx debug
struct kobject *fix_l3_kobj; 	 //zwx debug

//#define fix_l3_attr(_name) 


static int l3_level_set = 10;  //255 for dcvs
static int pmu_count = 7;




static struct thermal_zone_device *thermal_dev0_0;
static struct thermal_zone_device *thermal_dev4_0;
static struct thermal_zone_device *thermal_dev7_0;
struct pmu_event_struct
{
        struct perf_event *event; //pmu event 结构体,相当于访问pmu的句柄

        s64 ev_count; //pmu计数增量，即这一次总计数减去上一次总计数，功耗模型用的是pmu的“增量”

        u64 prev_total_count; //pmu上一次总计数

        s64 last_jiffies; //时间间隔
};

struct pmu_event_struct cpufreq_pmu_event[8][7];


/*
pmu初始化：
使用内核api: perf_event_create_kernel_counter()进行pmu的建立和初始化
pmu的初始化依赖于底层pmu的驱动。而有的模块会在pmu驱动之前加载。
也就是说，在开机时，使用perf_event_create_kernel_counter会初始化失败，直到pmu驱动加载成功后才会初始化成功
可以用线程+循环来实现：全部初始化成功后结束线程
也可以用工作队列实现：未全部初始化，便扔进工作队列下一次继续；全部初始化就不再扔进工作队列。

perf_event_create_kernel_counter用于初始化“一个cpu的一个pmu”，并返回struct perf_event的指针。
所以8个cpu核，每个核7个pmu，一共56个struct perf_event指针，用一个数组保存。这个数组名为cpufreq_pmu_even。
*/

static int pmu_init()
{
        struct perf_event_attr *attr = NULL;
        int cpu, cpu_first;
        int all_enable = 0; //所有pmu都成功初始化的标志。初始化失败和0与， 初始化成功和1与，当all_enable最后是1时，说明所有pmu初始化成功。
        //int idx;
		int pmu_index;

        attr = kzalloc(sizeof(struct perf_event_attr), GFP_KERNEL);

        if (!attr)
        {
                printk(KERN_EMERG "CREATE ATTR ERROR\n");
                return -1;
        }

	memset(attr, 0, sizeof(struct perf_event_attr));
        attr->type = PERF_TYPE_RAW;
        attr->size = sizeof(struct perf_event_attr);
        attr->pinned = 1;
	attr->sample_type |= PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_TIME | PERF_SAMPLE_PERIOD | PERF_SAMPLE_CPU | PERF_SAMPLE_ID;
	all_enable = 1;
	do
	{
		for(cpu = 0; cpu < 8; cpu++)
		{
			if(cpu < 4)
			{
				for(pmu_index = 0; pmu_index < pmu_count; pmu_index++)
					{
						switch(pmu_index)
						{
							case 0:
								attr->config = 0x11;
								break;
							case 1:
								attr->config = 0x24;
								break;
							case 2:
								attr->config = 0x14;
								break;
							case 3:
								attr->config = 0x01;
								break;
							case 4:
								attr->config = 0x13;
								break;
							case 5:
								attr->config = 0x26;
								break;
							case 6:
								attr->config = 0x17;
								break;
							
						default:
						printk(KERN_EMERG "INT PMU: CPU%d NOT FOUND", cpu);
						}
						if(!cpufreq_pmu_event[cpu][pmu_index].event)
						{
							cpufreq_pmu_event[cpu][pmu_index].event = perf_event_create_kernel_counter(attr, cpu, NULL, NULL, NULL);
							if(IS_ERR(cpufreq_pmu_event[cpu][pmu_index].event))
							{
								printk(KERN_EMERG "CREATE EVENT(0x%2x) ON CPU%d FAILED\n", attr->config, cpu);
								cpufreq_pmu_event[cpu][pmu_index].event = NULL;
								all_enable &= 0;
							}
							else
							{
								printk(KERN_EMERG "CREATE EVENT(0x%2x) ON CPU%d SUCCESS\n", attr->config, cpu);
								perf_event_enable(cpufreq_pmu_event[cpu][pmu_index].event);
								all_enable &= 1;
							}
						}
						else
							all_enable &= 1;			
					}
			}else
			{
				for(pmu_index = 0; pmu_index < pmu_count; pmu_index++)
						{
							switch(pmu_index)
							{
								case 0:
									attr->config = 0x11;
									break;
								case 1:
									attr->config = 0x24;
									break;
								case 2:
									attr->config = 0x12;
									break;
								case 3:
									attr->config = 0x35;
									break;
								case 4:
									attr->config = 0x04;
									break;
								case 5:
									attr->config = 0x26;
									break;
								case 6:
									attr->config = 0x17;
									break;
								
							default:
							printk(KERN_EMERG "INT PMU: CPU%d NOT FOUND", cpu);
							}
							if(!cpufreq_pmu_event[cpu][pmu_index].event)
							{
								cpufreq_pmu_event[cpu][pmu_index].event = perf_event_create_kernel_counter(attr, cpu, NULL, NULL, NULL);
								if(IS_ERR(cpufreq_pmu_event[cpu][pmu_index].event))
								{
									printk(KERN_EMERG "CREATE EVENT(0x%2x) ON CPU%d FAILED\n", attr->config, cpu);
									cpufreq_pmu_event[cpu][pmu_index].event = NULL;
									all_enable &= 0;
								}
								else
								{
									printk(KERN_EMERG "CREATE EVENT(0x%2x) ON CPU%d SUCCESS\n", attr->config, cpu);
									perf_event_enable(cpufreq_pmu_event[cpu][pmu_index].event);
									all_enable &= 1;
								}
							}
							else
								all_enable &= 1;			
						}
			}
			 
			
		}
	}while(!all_enable);
	
        kfree(attr);
        attr = NULL;
	if(all_enable)
		return 1;
	return 0;
}

static void pmu_reset()
{
int i,j;
for (i=0;i<8;i++){
if(i < 4)
	for (j=0;j<pmu_count;j++){
		perf_event_disable(cpufreq_pmu_event[i][j].event);
		perf_event_release_kernel(cpufreq_pmu_event[i][j].event);
		cpufreq_pmu_event[i][j].event=NULL;
		cpufreq_pmu_event[i][j].ev_count=0;
		cpufreq_pmu_event[i][j].prev_total_count=0;
		}
else
	for (j=0;j<pmu_count;j++){
		perf_event_disable(cpufreq_pmu_event[i][j].event);
		perf_event_release_kernel(cpufreq_pmu_event[i][j].event);
		cpufreq_pmu_event[i][j].event=NULL;
		cpufreq_pmu_event[i][j].ev_count=0;
		cpufreq_pmu_event[i][j].prev_total_count=0;
		}

printk(KERN_EMERG "RELEASE EVENT ON CPU %d SUCCESS\n", i);
		}
pmu_inited = pmu_init();
}

static void pmu_stop()
{
int i,j;
for (i=0;i<8;i++){
if(i < 4)
	for (j=0;j<pmu_count;j++){
		perf_event_disable(cpufreq_pmu_event[i][j].event);
		perf_event_release_kernel(cpufreq_pmu_event[i][j].event);
		cpufreq_pmu_event[i][j].event=NULL;
		cpufreq_pmu_event[i][j].ev_count=0;
		cpufreq_pmu_event[i][j].prev_total_count=0;
		}
else
	for (j=0;j<pmu_count;j++){
		perf_event_disable(cpufreq_pmu_event[i][j].event);
		perf_event_release_kernel(cpufreq_pmu_event[i][j].event);
		cpufreq_pmu_event[i][j].event=NULL;
		cpufreq_pmu_event[i][j].ev_count=0;
		cpufreq_pmu_event[i][j].prev_total_count=0;
		}

printk(KERN_EMERG "RELEASE EVENT ON CPU %d SUCCESS\n", i);
		}

		
}

//fix_l3_attr(l3_level);
static ssize_t l3_level_show(struct kobject *kobj,
			      struct kobj_attribute *attr,
			      char *buf)
{
	return sprintf(buf, "%d\n", l3_level_set);
}

static ssize_t l3_level_store(struct kobject *kobj,
			       struct kobj_attribute *attr,
			       const char *buf, size_t n)
{
	int l3_level = 0;

	if (sscanf(buf, "%d", &l3_level) == 1)
		l3_level_set = l3_level;
	if(l3_level_set == 10000)
		pmu_stop();
	if(l3_level_set == 10001)
		pmu_inited = pmu_init();

	return n;
}

static struct kobj_attribute l3_level_attr = {	
	.attr	= {				
		.name = "pmu_controll",	
		.mode = 0644,			
	},					
	.show	= l3_level_show,			
	.store	= l3_level_store,		
};

static ssize_t pmu_count_show(struct kobject *kobj,
			      struct kobj_attribute *attr,
			      char *buf)
{
	return sprintf(buf, "%d\n", pmu_count);
}
static ssize_t pmu_count_store(struct kobject *kobj,
			       struct kobj_attribute *attr,
			       const char *buf, size_t n)
{
	int pmu_c = 0;

	if (sscanf(buf, "%d", &pmu_c) == 1)
		pmu_count = pmu_c;
	return n;
}

static struct kobj_attribute pmu_count_attr = {	
	.attr	= {				
		.name = "pmu_count",	
		.mode = 0644,			
	},					
	.show	= pmu_count_show,			
	.store	= pmu_count_store,		
};

static struct attribute * g[] = {
	&l3_level_attr.attr,
	&pmu_count_attr.attr,
	NULL,
};

static const struct attribute_group attr_group = {
	.attrs = g,
};

static const struct attribute_group *attr_groups[] = {
	&attr_group,
	NULL,
};
//zwx debug

static int pmu_read()
{
	int cpu, event_idx;
	u64 total, enable, running;
	struct perf_event *event = NULL;
	//perf_event_read_value一次只能读取一个cpu的一个pmu事件，所以要用循环
	int temp0,temp4,temp7;
	thermal_zone_get_temp(thermal_dev0_0, &temp0);
	thermal_zone_get_temp(thermal_dev4_0, &temp4);
	thermal_zone_get_temp(thermal_dev7_0, &temp7);
	for(cpu = 0; cpu < 8; cpu++) //8个cpu
	{
		if(cpu < 4)
		{
		for(event_idx = 0; event_idx < pmu_count; event_idx++) //每个cpu有7个事件
			{
			event = cpufreq_pmu_event[cpu][event_idx].event;
						
			total = perf_event_read_value(event, &enable, &running); //read new value
				//printk(KERN_EMERG "CPU#%d event(0x%2x)   \n",cpu, cpufreq_pmu_event[cpu][event_idx].event->attr.config);
			cpufreq_pmu_event[cpu][event_idx].ev_count = total - cpufreq_pmu_event[cpu][event_idx].prev_total_count; //delta = new - last
			cpufreq_pmu_event[cpu][event_idx].prev_total_count = total; //last = new

		#if(0)
			//这是调试代码
			printk(KERN_EMERG "CPU#%d event(0x%2x) and value (%d,%03d,%03d)\n", cpu,
												cpufreq_pmu_event[cpu][event_idx].event->attr.config,
												cpufreq_pmu_event[cpu][event_idx].ev_count / 1000000,
												cpufreq_pmu_event[cpu][event_idx].ev_count % 1000000 / 1000,
												cpufreq_pmu_event[cpu][event_idx].ev_count % 1000
												);
		#endif

			}
		trace_pmu_power_model(cpu , cpufreq_pmu_event[cpu][0].ev_count,cpufreq_pmu_event[cpu][1].ev_count,cpufreq_pmu_event[cpu][2].ev_count,cpufreq_pmu_event[cpu][3].ev_count,cpufreq_pmu_event[cpu][4].ev_count,cpufreq_pmu_event[cpu][5].ev_count,cpufreq_pmu_event[cpu][6].ev_count,temp0);
		}
		else
		{
		for(event_idx = 0; event_idx < pmu_count; event_idx++) //每个cpu有7个事件
			{
			if(event_idx == 0 && cpufreq_pmu_event[cpu][event_idx].ev_count == 0)
			{	if(over_count < 5)
					{over_count++;
					printk(KERN_EMERG "over count = %d\n",over_count);}
				else{
					over_count = 0;
					return 1;
					}
			}
			event = cpufreq_pmu_event[cpu][event_idx].event;
						
			total = perf_event_read_value(event, &enable, &running); //read new value
				//printk(KERN_EMERG "CPU#%d event(0x%2x)   \n",cpu, cpufreq_pmu_event[cpu][event_idx].event->attr.config);
			cpufreq_pmu_event[cpu][event_idx].ev_count = total - cpufreq_pmu_event[cpu][event_idx].prev_total_count; //delta = new - last
			cpufreq_pmu_event[cpu][event_idx].prev_total_count = total; //last = new

		#if(1)
			//这是调试代码
			printk(KERN_EMERG "CPU#%d event(0x%2x) and value (%d,%03d,%03d) and total = %lu prev_total = %lu\n", cpu,
												cpufreq_pmu_event[cpu][event_idx].event->attr.config,
												cpufreq_pmu_event[cpu][event_idx].ev_count / 1000000,
												cpufreq_pmu_event[cpu][event_idx].ev_count % 1000000 / 1000,
												cpufreq_pmu_event[cpu][event_idx].ev_count % 1000,
												total,
												cpufreq_pmu_event[cpu][event_idx].prev_total_count
												);
		#endif

			}
		if(cpu == 7)
			trace_pmu_power_model(cpu , cpufreq_pmu_event[cpu][0].ev_count,cpufreq_pmu_event[cpu][1].ev_count,cpufreq_pmu_event[cpu][2].ev_count,cpufreq_pmu_event[cpu][3].ev_count,cpufreq_pmu_event[cpu][4].ev_count,cpufreq_pmu_event[cpu][5].ev_count,cpufreq_pmu_event[cpu][6].ev_count,temp7);
		else
			trace_pmu_power_model(cpu , cpufreq_pmu_event[cpu][0].ev_count,cpufreq_pmu_event[cpu][1].ev_count,cpufreq_pmu_event[cpu][2].ev_count,cpufreq_pmu_event[cpu][3].ev_count,cpufreq_pmu_event[cpu][4].ev_count,cpufreq_pmu_event[cpu][5].ev_count,cpufreq_pmu_event[cpu][6].ev_count,temp4);
		}
	
	}
	return 0;
}
static void pmu_init_thermal_dev(){
	thermal_dev0_0 = thermal_zone_get_zone_by_name("cpu-0-0-usr");
	thermal_dev4_0 = thermal_zone_get_zone_by_name("cpu-1-0-usr");
	thermal_dev7_0 = thermal_zone_get_zone_by_name("cpu-1-6-usr");
}
static void my_work_func(struct work_struct *work)
{
   
    // reschedule the work item after 5 ms
    u64 old_rate = 0;
	int cpu;
	int pmu_over = 0;
    //u64 lllc_rate = 0;
    //int lllc_freq = 0;

     if(counter<4000)
	{
		if(counter == 3500)
			pmu_init_thermal_dev();
		printk(KERN_EMERG "pmu initting inited = %d",pmu_inited);
    	printk(KERN_EMERG "Task %d executed\n", ++counter);
	}
    else {
		if(trace_pmu_power_model_enabled())
		{
			pmu_over = pmu_read();
			if(pmu_over)
				pmu_reset();
		}	 
	}
	if(counter > 2950 )
    		queue_delayed_work(my_wq, &my_work, msecs_to_jiffies(l3_level_set));
	else	queue_delayed_work(my_wq, &my_work, msecs_to_jiffies(10));
}


//定义一个模块初始化函数，用于注册定时器
static int __init my_init(void)
{
	int error;  //zwx debug
	fix_l3_kobj = kobject_create_and_add("l3_fix", NULL);
	if (!fix_l3_kobj)
		return -ENOMEM;
	error = sysfs_create_groups(fix_l3_kobj, attr_groups);
	if (error)
		return error;
	     pr_info("Module loaded\n");
	pmu_inited = pmu_init();
	    // create a single-threaded workqueue
	    my_wq = create_singlethread_workqueue("my_wq");
	    if (!my_wq) {
		pr_err("Failed to create workqueue\n");
		return -ENOMEM;
	    }
	    // initialize the work item with the function pointer
	    INIT_DELAYED_WORK(&my_work, my_work_func);
	    // queue the work item after 5 ms
	    queue_delayed_work(my_wq, &my_work, msecs_to_jiffies(5));
	    return 0;
}

//定义一个模块退出函数，用于删除定时器
static void __exit my_exit(void)
{
      pr_info("Module unloaded\n");
    // cancel the work item if it is pending
    cancel_delayed_work(&my_work);
    // destroy the workqueue
    destroy_workqueue(my_wq);
}

module_init(my_init);
module_exit(my_exit);

MODULE_LICENSE("GPL");
#endif

