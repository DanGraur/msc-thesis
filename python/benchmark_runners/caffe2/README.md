# Observations on the Caffe2 benchmark runner

## Forward and Forward+Backward execution times

The benchmark runner is currently targeted towards running both forward pass and forward+backward passes and measuring their execution times. It can do this in both a single node context, and multi-node context, and offers a multitude of possible options for running experiments especially in the context of multi-node evaluation.

It should be noted that in the case that **no backward pass is used**, and thus, only forward passes are evaluated, the nodes can run independently, since no updates are actually applied on the parameters. In this case, one should collect the times of each node and average them by the number of nodes.

I have not been able to find a way to limit the number of threads which are being used when running a Caffe2 process on the node. It seems that regardless of which options are used, the process will always spawn in excess of a few hundred threads which will be scheduled on the machine's (virtual) cores.

The following are sample command lines for running the evaluation in **Forward+Backward pass** mode:

```
# Multinode
python resnet_forward_benchmark.py --num_shards=2 --shard_id=<shard_id> --rendezvous_path=/var/scratch/dograur/caffe2_rendezvous/benchmarks/pilot_try  --training_data=/home/dograur/tutorials/caffe2/resnet-venv/data/mnist-train-nchw-lmdb  --testing_data=/home/dograur/tutorials/caffe2/resnet-venv/data/mnist-test-nchw-lmdb --batch_size=32 --use_cpu=True  --backward=True --per_device_optimization=False --epoch_size=64 --warmup_rounds=1 --eval_rounds=1

# Singlenode
python resnet_forward_benchmark.py --training_data=/home/dograur/tutorials/caffe2/resnet-venv/data/mnist-train-nchw-lmdb  --testing_data=/home/dograur/tutorials/caffe2/resnet-venv/data/mnist-test-nchw-lmdb --batch_size=32 --use_cpu=True  --backward=True --per_device_optimization=False --per_layer_eval=True --epoch_size=64 --warmup_rounds=1 --eval_rounds=1
```

The following are sample command lines for running the evaluation in **Forward pass** mode:

```
# Multinode (no synchronization)
python resnet_forward_benchmark.py --num_shards=2 --shard_id=<shard_id> --rendezvous_path=/var/scratch/dograur/caffe2_rendezvous/benchmarks/pilot_try  --training_data=/home/dograur/tutorials/caffe2/resnet-venv/data/mnist-train-nchw-lmdb  --testing_data=/home/dograur/tutorials/caffe2/resnet-venv/data/mnist-test-nchw-lmdb --batch_size=32 --use_cpu=True  --backward=False --epoch_size=64 --warmup_rounds=1 --eval_rounds=1

# Singlenode
python resnet_forward_benchmark.py --training_data=/home/dograur/tutorials/caffe2/resnet-venv/data/mnist-train-nchw-lmdb  --testing_data=/home/dograur/tutorials/caffe2/resnet-venv/data/mnist-test-nchw-lmdb --batch_size=32 --use_cpu=True  --backward=False --per_layer_eval=True --epoch_size=64 --warmup_rounds=1 --eval_rounds=1
```

One should be careful when running this in Forward pass mode only without any synchronization between devices in **MULTINODE** context, as they will just run independently. This cannot be avoided, as this is intrinsic to the setup of the experiment. If you want synchronization, then the backward pass is required as well!

## Time to accuracy

The benchmark can also measure the time it takes a node to each a particular accuracy (Top-1).

A sample command line for executing this for two nodes might be (make sure to replace the **shard_id**'s parameter value with a unique ID between [0, num_shards)):

```
# Multinode:
python resnet_forward_benchmark.py --num_shards=2 --shard_id=<shard_id> --rendezvous_path=/var/scratch/dograur/caffe2_rendezvous/benchmarks/pilot_try  --training_data=/home/dograur/tutorials/caffe2/resnet-venv/data/mnist-train-nchw-lmdb  --testing_data=/home/dograur/tutorials/caffe2/resnet-venv/data/mnist-test-nchw-lmdb --batch_size=128 --use_cpu=True --test_accuracy=True --backward=True --per_device_optimization=False --target_accuracy=0.97 --terminate_on_target=True --epoch_count=5 --base_learning_rate=0.1 --epoch_size=64

# Single Node:
# TODO: requires special implementation of RunEpoch with no prefixes.
```

Some issues might occur here due to blob fetching (such as the learning rate not existing under some given name), **OR THE TOP-(1|5) ACCURACIES CANNOT BE RETRIEVED FROM OTHER PEERS DUE TO NAME PREFIX ISSUES**. To avoid this, it may be best to run the experiments using `--per_device_optimization=False` option. The `--backward=True` option should also always be used, otherwise, the training will have no effect on the parameters, and the accuracy will be 1.0 for both Top-1 and Top-5.