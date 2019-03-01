# Manually Running TensorFlow on GPUs

TensorFlow can be run on GPUs with the help of SLURM. SLURM is essential here since it allows one to explictily select nodes which have GPU capabilities. 

TensorFlow can be run on GPUs either using a single node (i.e. one node with (*currently*) one GPU), or across several nodes, each having its own GPU. **DISCLAIMER: On certain occasions it might prove that the CUDA drivers installed on the compute nodes differ from that used in the runtime.** To this extent I've seen only *node024* and *node025* actually work without throwing this error. 

## The command lines 

**IMPORTANT:** Always make sure that the options within these scripts, which actually control the experiment context, fit your needs, i.e. change them if you must.

#### Running on a single node

To run the benchark script on **one node (with a single GPU, assumingly)** execute the following on a node:

```
sbatch -w node024 tf_gpu_run_single.job ; squeue | grep dograur
```

Notice that in the above command line we use the option `-w node0xx`, where `xx` should be replaced with some node that has GPU capabilities (and doesn't have driver version issues). The `-w` option demands a particular node indicated by its parameter.

#### Running on multiple nodes

Running a distributed TensorFlow benchmark using a **PARAMETER SERVER** architecture will require that a process is spawned which will act as a PS. This node can usually be a simple compute node without any GPU capabilities. One can spawn such a node using the command line below:

```
python /home/dograur/tutorials/tensorflow/benchmarks/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --data_format=NCHW --batch_size=128 --model=alexnet --local_parameter_device=GPU --device=GPU --optimizer=momentum --variable_update=parameter_server --num_gpus=1 --forward_only=False --print_training_accuracy=True --num_batches=10000 --display_every=5 --benchmark_log_dir=/home/dograur/tutorials/tensorflow/benchmarks/benchmarks/scripts/tf_cnn_benchmarks/logs --data_name=cifar10 --tfprof_file=/home/dograur/tutorials/tensorflow/benchmarks/benchmarks/scripts/tf_cnn_benchmarks/logs/tfprof_logs --ps_hosts=node058:2222 --worker_hosts=node024:2222,node025:2222 --job_name=ps --task_index=0
```

The command line above assumes that *node058* is acting as a PS, and that nodes *node024* and *node025* are acting as GPU based worker nodes. 

We then spawn the two worker nodes, using the command line below: 

```
# For forward or forward+bacward benchmarks
sbatch -w node0xx tf_gpu_run_distributed.job <task_index_id> ; squeue | grep dograur
```

