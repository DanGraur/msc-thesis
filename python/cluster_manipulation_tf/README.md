# Instructions on how to use the Cluster Spawner for TensorFlow

To run the cluster spawner one must use a command line of the following form:

`./cluster_spawner.py <path_to_application> <list_of_nodes> <other_params_for_the_cluster_spawner>`

Generally, one would also include the `-args <application_parameters>` within the command line, in order to pass essential arguments to the TF application. For example, using the `cifar_10.py` script, one might be able to use the following:
   
`./cluster_spawner.py cifar_10.py node062 node063 -t 0 -pn 1 -pt 1 -wt 1 -args 'cifar-10-batches-py'`

The previous comand line assumes that:

   * We have managed to reserve 2 nodes: `node062` and `node063`
   * We have 1 `PS node`
   * There is no timeout (since -t 0)
   * 1 task per `PS node`
   * 1 task per `Worker node`
   * We pass the `cifar-10-batches-py` positional argument to the TF application (i.e. to `cifar_10.py`) - in this case, this is the directory where the CIFAR 10 data is located - 
