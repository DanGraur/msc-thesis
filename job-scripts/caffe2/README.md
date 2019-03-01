# Manually Running Caffe2 on GPUs

Caffe2 can be run on GPUs with the help of SLURM. SLURM is essential here since it allows one to explictily select nodes which have GPU capabilities. 

Caffe2 can be run on GPUs either using a single node (i.e. one node with (*currently*) one GPU), or across several nodes, each having its own GPU. **DISCLAIMER: On certain occasions it might prove that the CUDA drivers installed on the compute nodes differ from that using in the runtime.** To this extent I've seen only *node024* and *node025* actually work without throwin this error. 

**IMPORTANT:** The shared model functionality is only a CPU based feature, so it's not possible to employ it when running experiments across several GPU based devices. 

## The command lines 

**IMPORTANT:** Always make sure that the options within these scripts, which actually control the experiment context, fit your needs, i.e. change them if you must.

#### Running on a single node

To run the benchark script on **one node (with a single GPU, assumingly)** execute the following on a node (in the same directory as the benchmark script):

```
sbatch caffe2_gpu_run_single.job ; squeue | grep dograur
```

#### Running on multiple nodes

**IMPORTANT**: Caffe2 will generally use the node with `shard_id=0` as the synchronizer, so this node must be initiated before all the others. This might prove difficult using SLURM, since jobs might not be executed in the same order as they have been submitted, so make sure that the node with `shard_id=0` is running before submitting the others (perhaps add a simple delay between submissions).

To run the benchark script on **multiple nodes (with a single GPU each, assumingly)** execute the following on a node (in the same directory as the benchmark script):

```
# For forward or forward+bacward benchmarks
sbatch caffe2_gpu_run_distributed.job <shard_id> ; squeue | grep dograur

# For time to accuracy benchmarks
sbatch caffe2_gpu_run_distributed_accuracy.job <shard_id> ; squeue | grep dograur
```


