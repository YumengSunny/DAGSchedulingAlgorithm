# DAGSchedulingAlgorithm

[完整算法仓库](https://github.com/automaticdai/research-dag-scheduling-analysis)

## 说明

- graph.py: 读图类， 用于寻找一个DAG图的关键路线， predecessor, successor等等。 这个类将会在eopa类中被调用
- eopa.py: 算法类，
         最主要的方法是其中的EOPA(self, task_idx, m)，其中task_idx为data文件夹中对应的任务号码， m为对应的处理器核数。最后输出为反应时间， alpha干扰节点以及beta干扰节点
         通过load_task(task_idx)方法，可以查看DAG图以dictionary方式的呈现， task_idx为data文件夹中对应的任务号码。
         这些用法在文件的最末尾出有使用例子。
         此外， 从行1084-1089可以打印这个DAG图

- sched.py: 调度代码， 用于模拟所有节点在不同核数情况下的运行情况， 用以生成调度图

## Installation
下载后安装相应的包之后即可运行
