# Instructions on how to use the Cluster Spawner for Caffe2

To run the cluster spawner one must use a command line of the following form:

`./cluster_spawner.py <path_to_application> <list_of_nodes> --app-args='<arguments_to_the_application>' <other_params_for_the_cluster_spawner>`

For example, using the `resnet.py` script, one might be able to use the following:
   
`./cluster_spawner.py resnet.py node062 node063 --timeout=0 --app-args='--train_data=data/mnist-train-nchw-lmdb/ --test_data=data/mnist-test-nchw-lmdb/ --test_epoch_size=1000 --epoch_size=2000 --num_gpus=1 --use_cpu=True --num_epochs=10 --image_size=28 --num_labels=10 --num_shards=%d --shard_id=%d --file_store_path=/var/scratch/dograur/caffe2_rendevous/2n2p'`

The previous comand line assumes that:

    * We have managed to reserve 2 nodes: node062 and node063 
    * We let the cluster spawner infer how many nodes there are (num_shards), and what the shard_id of each node is, hence, we use %d for both those parameters in the command line
    * There is no timeout (since --timeout=0)