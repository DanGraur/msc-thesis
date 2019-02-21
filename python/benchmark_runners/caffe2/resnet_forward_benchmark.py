from argparse import ArgumentParser

from uuid import uuid4

from caffe2.python import core, brew, model_helper, workspace, data_parallel_model
from caffe2.python.models import resnet


def str_to_bool(s):
    return s.lower() in ['yes', 'y', 'true', 't']


def AddParameterUpdate(model):
    # This counts the number if iterations we are making 
    ITER = brew.iter(model, "iter")
    # This adds a learning rate to the model, updated using a simple 'step' policy every 10k steps; gamma is an update parameter
    LR = model.LearningRate(ITER, "LR", base_lr=-1e-8, policy="step", stepsize=10000, gamma=0.999)
    # This is a constant used in the following loop
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Here we are essentially applying the gradients to the weights (using the classical method)
    for param in model.params:
        param_grad = model.param_to_grad[param]
        model.WeightedSum([param, ONE, param_grad, LR], param)

    # This can optionally be used to checkpoint the parameters, however, it might take up computation time
    # model.Checkpoint([ITER] + model.params, [],
    #            db="resnet50_cifar_10_checkpoint_%05d.lmdb",
    #            db_type="lmdb", every=20)


def AddImageInput(model, db_type, db_location, batch_size, image_size, data_type="float", is_test=False):
    reader = model.CreateDB(
        "reader",
        db=db_location,
        db_type=db_type
    )

    data, label = brew.image_input(
        model,
        reader, ["data", "label"],
        batch_size=batch_size,
        output_type="float",
        use_gpu_transform=False,
        use_caffe_datum=True,
        mean_per_channel=None,
        std_per_channel=None,
        # mean_per_channel takes precedence over mean
        mean=128.,
        std=128.,
        scale=256,
        crop=image_size,
        mirror=1,
        is_test=is_test,
    )

    data = model.StopGradient(data, data)

    # The first array is the empty input parameters, to the reader function, the latter are for the output blob labels
    # data_uint8, label = model.TensorProtosDBInput([], ["data_uint8", "label"], batch_size=batch_size, 
    #     db=db_location, db_type=db_type)
    # data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    return data, label


def resnet_eval(args):
    # The training scope of the netwrk
    # train_arg_scope = {
    #     'use_cudnn': False,
    #     'cudnn_exhaustive_search': False,
    #     'training_mode': 1
    # }

    # Define a reader for the network's training data
    # reader = train_model.CreateDB(
    #     "reader",
    #     db=args.train_data,
    #     db_type=args.db_type
    # )

    # Instantiate the model
    # model = model_helper.ModelHelper(name='resnext50', arg_scope=train_arg_scope)
    model = model_helper.ModelHelper(name='resnext50')
    model.Proto().num_workers = 2

    # Feed the input to the model

    # Batch, Channels, Height, Width
    input_shape = [args.batch_size, args.channels, args.height, args.width]
    if args.training_data:
        print("Runnig experiments with user provided data:", args.training_data)
 
        AddImageInput(model, args.db_type, args.training_data, args.batch_size, min(args.height, args.width))
    else:
        # Generate the input data from a Gaussian Distribution 
        model.param_init_net.GaussianFill(
            [],
            "data",
            shape=input_shape,
            mean=0.0,
            std=1.0
        )

        # Generate the labels from a Uniform Distribution
        model.param_init_net.UniformIntFill(
            [],
            "label",
            shape=[args.batch_size, ],
            min=0,
            max=args.num_labels - 1
        )

    # Create the network
    resnet.create_resnet50(model, "data", args.channels, args.num_labels, label="label")

    if not args.backward:
        print('Resnext50: running forward only.')
    else:
        print('Resnext50: running forward-backward.')
        model.AddGradientOperators(["loss"])
        AddParameterUpdate(model)

    # Training will run on CPU for the moment
    # if not arg.cpu:
    #     model.param_init_net.RunAllOnGPU()
    #     model.net.RunAllOnGPU()


    if args.num_shards > 1:
        print("Distributed benchmarking is enabled")
        print("Num shards: {}\nMy shard ID: {}\nRendevous at: {}".format(
            args.num_shards, args.shard_id, args.rendezvous_path))
        # Prepare the required parameters for distribution
        store_handler = "store_handler"
        
        # We'll use the shared file system for rendezvous
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "FileStoreHandlerCreate", 
                [], 
                [store_handler],
                path=args.rendezvous_path,
                prefix=args.run_id,
            )
        )

        rendezvous = dict(
            kv_handler=store_handler,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            engine="GLOO",
            transport=args.distributed_transport,
            interface=args.network_interface,
            exit_nets=None
        )

        # Create parallelized model
        data_parallel_model.Parallelize(
            model,
            # TODO: need to write these two methods
            # input_builder_fun=add_image_input,
            # forward_pass_builder_fun=create_resnext_model_ops,
            # optimizer_builder_fun=add_optimizer,
            # post_sync_builder_fun=add_post_sync_ops,
            devices=(args.gpu_devices if not args.use_cpu else [0]),
            rendezvous=rendezvous,
            optimize_gradient_memory=False,
            cpu_device=args.use_cpu,
            ideep=args.use_ideep,
            shared_model=args.use_cpu,
            combine_spatial_bn=args.use_cpu,
        )

        data_parallel_model.OptimizeGradientMemory(train_model, {}, set(), False)

    # Initialize the model's parameters
    workspace.RunNetOnce(model.param_init_net)
    # Create the network for later execution
    workspace.CreateNet(model.net)
    # Run the network through benchmarks: 10 warmup runs + 100 evaluation runs, with per-layer measurements
    workspace.BenchmarkNet(model.net.Proto().name, args.warmup_rounds, args.eval_rounds, args.per_layer_eval)


def main():
    # TODO: look into how to run this benchmark in a distributed fashion. To a certain extent, this will end the efforts on Caffe2, and scale experiments will be runnable here.
    # TODO: perhaps find a way to change the number of threads which can be run on Caffe2 (is this the mode.Proto().num_workers ? Perhaps).
    parser = ArgumentParser(description="Caffe2 Resnext50 benchmark.")
    parser.add_argument(
        "--run_id",
        type=str,
        default='',
        help="Unique run identifier"
    )
    parser.add_argument(
        "--training_data",
        type=str,
        default='',
        help="The path to the training data"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default='float',
        choices=['float', 'float16'],
        help="Describes the input data's type"
    )
    parser.add_argument(
        "--db_type",
        type=str,
        default='lmdb',
        help="The database type of the input data."
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=10,
        help="The number of labels"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=28,
        help="The height of the image"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=28,
        help="The width of the image"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="The number of channels"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size"
    )
    parser.add_argument(
        "--backward",
        type=str_to_bool,
        default=False,
        help="Perform parameter updates"
    )
    parser.add_argument(
        "--warmup_rounds",
        type=int,
        default=10,
        help="The number of warmup rounds"
    )
    parser.add_argument(
        "--eval_rounds",
        type=int,
        default=50,
        help="The number of evaluation rounds"
    )
    parser.add_argument(
        "--per_layer_eval",
        type=str_to_bool,
        default=False,
        help="Evaluate times on a per layer basis"
    )
    parser.add_argument(
        "--use_cpu",
        type=str_to_bool,
        default=True,
        help="Flag for CPU based computation"
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="The number of compute nodes"
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="The shard ID of this node (0 based index)"
    )
    parser.add_argument(
        "--rendezvous_path",
        type=str,
        default='',
        help="Path to a rendezvous folder"
    )
    parser.add_argument(
        "--distributed_transport",
        type=str,
        default='tcp',
        choices=['tcp', 'ibverbs'],
        help="Protocol for distributed communication"
    )
    parser.add_argument(
        "--network_interface",
        type=str,
        default='',
        help="Network interface for distributed run"
    )
    parser.add_argument(
        "--gpu_devices",
        type=str,
        default=[],
        help="Space separated list of GPU devices (on this node)",
        nargs='*'
    )
    parser.add_argument(
        "--use_ideep",
        type=str_to_bool,
        default=False,
        help="Use Intel's IDEEP"
    )


    args = parser.parse_args()

    # Ensure this run has a unique UUID
    if not args.run_id:
        args.run_id = str(uuid4())

    resnet_eval(args)


if __name__ == '__main__':
    main()
