# Observations on the Caffe2 benchmark runner

The benchmark runner is currently targeted towards running both forward pass and forward+backward passes and measuring their execution times. It can do this in both a single node context, and multi-node context, and offers a multitude of possible options for running experiments especially in the context of multi-node evaluation.

It should be noted that in the case that **no backward pass is used**, and thus, only forward passes are evaluated, the nodes can run independently, since no updates are actually applied on the parameters. In this case, one should collect the times of each node and average them by the number of nodes.

I have not been able to find a way to limit the number of threads which are being used when running a Caffe2 process on the node. It seems that regardless of which options are used, the process will always spawn in excess of a few hundred threads which will be scheduled on the machine's (virtual) cores.