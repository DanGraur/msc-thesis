import logging
import time

from argparse import ArgumentParser

from uuid import uuid4

from caffe2.python import brew, core, data_parallel_model, dyndep, experiment_util, model_helper, optimizer, timeout_guard, workspace
from caffe2.python.modeling.initializers import Initializer, PseudoFP16Initializer
from caffe2.python.models import resnet


# Load the dynamic libraries required for distributed training / evaluation
dyndep.InitOpsLibrary('@/caffe2/caffe2/distributed:file_store_handler_ops')
dyndep.InitOpsLibrary('@/caffe2/caffe2/distributed:redis_store_handler_ops')


# Set up the logger for the benchmarking script
logging.basicConfig()
log = logging.getLogger("BenchmarkNetLogger")
log.setLevel(logging.INFO)


def str_to_bool(s):
    """
    Convert a string to boolean

    :return: the boolean converted from string
    """
    return s.lower() in ['yes', 'y', 'true', 't']


def AddParameterUpdate(model):
    """
    Add a simple gradient based parameter update with stepwise adaptive learning rate.
    """
    # This counts the number if iterations we are making 
    ITER = brew.iter(model, "iter")
    # Adds learning rate to the model, updated using a simple step policy every 10k steps; gamma is an update parameter
    LR = model.LearningRate(ITER, "LR", base_lr=-1e-8, policy="step", stepsize=10000, gamma=0.999)
    # This is a constant used in the following loop
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Here we are essentially applying the gradients to the weights (using the classical method)
    for param in model.params:
        param_grad = model.param_to_grad[param]
        model.WeightedSum([param, ONE, param_grad, LR], param)


def AddImageInput(model, reader, batch_size, img_size, data_type, mean=128., std=128., scale=256, mirror=1, is_test=False):
    """
    Adds an image input to the model, supplied via a reader.

    :param model: the caffe2 model to which we're adding the image input
    :param reader: the reader which supplies the images from a DB
    :param batch_size: the batch size
    :param img_size: specifies the size of the images (images are considered to be square, and will be cropped)
    :param data_type: the output data type (float or float16)
    :param mean: the global channel mean value (used for normalization purposes)
    :param std: the global channel deviation (used for normalization purposes)
    :param scale: the scale variable (values will be scaled by 1./scale)
    :param mirror: indicates whether or not to mirror the available images for extra robustness (1 = True; 0 = False)
    :param is_test: if True, then the crop will be made at the exact same position in the image
    """
    data, label = brew.image_input(
        model,
        reader, 
        ["data", "label"],
        batch_size=batch_size,
        output_type=data_type,
        use_gpu_transform=True if core.IsGPUDeviceType(model._device_type) else False,
        use_caffe_datum=True,
        # mean_per_channel takes precedence over mean
        mean=float(mean),
        std=float(std),
        scale=scale,
        crop=img_size,
        mirror=mirror,
        is_test=is_test,
    )

    # Avoid updating the data itself
    data = model.StopGradient(data, data)


def AddSyntheticInput(model, data_type, shape, num_labels):
    """
    Add Synthetic data sampled from a multivariate Gaussian Distribution, with 
    labels sampled from a Uniform Categorical Distribution (INT32 based).

    :param model: the model to which the data input is added
    :param data_type: the type of the Gaussian data being generated (float or float16)
    :param shape: A 1x4 vector describing the shape of the generated data. The leading
                  element in the vector shoudl be the batch size
    :param num_labels: the number of possible labels 
    """
    suffix = "_fp16" if data_type == "float16" else ""
    
    model.param_init_net.GaussianFill(
        [],
        ["data" + suffix],
        shape=shape,
    )

    # If we want float on 2 Bytes, then we half the 4 Byte generated data
    if data_type == "float16":
        model.param_init_net.FloatToHalf("data" + suffix, "data")

    # The labels will be 4 Byte integers
    model.param_init_net.UniformIntFill(
        [],
        ["label"],
        # We expect the first parameter of the shape to be the batch count
        shape=[shape[0], ],
        min=0,
        max=num_labels - 1
    )


def RunEpoch(args, epoch, model, total_batch_size, num_shards, expname, explog):
    """
    Run one epoch of the trainer.
    TODO: add checkpointing here.
    """
    # TODO: add loading from checkpoint
    epoch_iters = int(args.epoch_size / total_batch_size / num_shards)
    test_epoch_iters = int(args.test_epoch_size / total_batch_size / num_shards)
    for i in range(epoch_iters):
        # This timeout is required (temporarily) since CUDA-NCCL
        # operators might deadlock when synchronizing between GPUs.
        timeout = 600.0 if i == 0 else 60.0
        with timeout_guard.CompleteInTimeOrDie(timeout):
            t1 = time.time()
            workspace.RunNet(train_model.net.Proto().name)
            t2 = time.time()
            dt = t2 - t1

        fmt = "Finished iteration {}/{} of epoch {} ({:.2f} images/sec)"
        prefix = "{}_{}".format(
            train_model._device_prefix,
            train_model._devices[0])
        accuracy = workspace.FetchBlob(prefix + '/accuracy')
        loss = workspace.FetchBlob(prefix + '/loss')
        train_fmt = "Training loss: {}, accuracy: {}"

    num_images = epoch * epoch_iters * total_batch_size
    prefix = "{}_{}".format(train_model._device_prefix, train_model._devices[0])
    accuracy = workspace.FetchBlob(prefix + '/accuracy')
    loss = workspace.FetchBlob(prefix + '/loss')
    learning_rate = workspace.FetchBlob(
        data_parallel_model.GetLearningRateBlobNames(train_model)[0]
    )
    test_accuracy = -1
    test_accuracy_top5 = -1
    # Removed the test model code which was previously below

    assert loss < 40, "Exploded gradients :("

    # TODO: add checkpointing
    return epoch + 1


def network_eval(args):
    """
    Runs network benchmarking on either a single or multiple nodes
    """
    # Define some parameters for the model instantiation
    if args.use_ideep:
        train_arg_scope = {
            'use_cudnn': False,
            'cudnn_exhaustive_search': False,
            'training_mode': 1
        }
    else:
        train_arg_scope = {
            'order': 'NCHW',
            'use_cudnn': True,
            'cudnn_exhaustive_search': True,
            # 1048576 = 2 ^ 20 (1 MB)
            'ws_nbytes_limit': (args.cudnn_ws_lim * 1048576),
        }
    # Create the model for evaluation        
    evaluation_model = model_helper.ModelHelper(
        name='resnext50', arg_scope=train_arg_scope
    )

    # Compute batch and epoch sizes 
    # Per CPU / GPU batch size
    per_local_device_batch = (args.batch_size // len(args.gpu_devices)) if args.gpu_devices else args.batch_size
    # Total batch size (over all the devices)
    global_batch_size = args.batch_size * args.num_shards
    # Number of epoch iterations
    epoch_iters = args.epoch_size // global_batch_size
    # Adjust the true number of examples per epoch
    args.epoch_size = global_batch_size * epoch_iters

    if args.training_data:
        log.info("Running experiments with user provided data: %s", args.training_data)

        # Create a reader, which can also help distribute data when running on multiple nodes
        reader = evaluation_model.CreateDB(
            "reader",
            db=args.training_data,
            db_type=args.db_type,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
        )
        
        def image_input(model):
            AddImageInput(model, reader, per_local_device_batch, min(args.height, args.width), 
                args.data_type)
    else:
        input_shape = [args.batch_size, args.channels, args.height, args.width]
        log.info("Running experiments with synthetic data w/ shape: %s", input_shape)

        def image_input(model):
            AddSyntheticInput(model, args.data_type, input_shape, args.num_labels)

    # Create the network, and normalize the loss
    def create_model(model, loss_scale):
        initializer = (PseudoFP16Initializer if args.data_type == 'float16' else Initializer)    

        with brew.arg_scope([brew.conv, brew.fc],
                            WeightInitializer=initializer,
                            BiasInitializer=initializer,
                            enable_tensor_core=False,
                            float16_compute=False):
            pred = resnet.create_resnet50(
                model,
                "data",
                num_input_channels=args.channels,
                num_labels=args.num_labels,
                # num_groups=args.resnext_num_groups,
                # num_width_per_group=args.resnext_width_per_group,
                no_bias=True,
                no_loss=True
            )

        # If we're using float on 2B, then inflate to the 4B representation
        if args.data_type == 'float16':
            pred = model.net.HalfToFloat(pred, pred + '_fp32')

        # Compute the softmax probabilities and the loss
        softmax, loss = model.SoftmaxWithLoss([pred, 'label'], ['softmax', 'loss'])

        # Noralize the loss, and compute the top_k accuracies for k \in {1, 5}
        loss = model.Scale(loss, scale=loss_scale)
        brew.accuracy(model, [softmax, "label"], "accuracy", top_k=1)
        brew.accuracy(model, [softmax, "label"], "accuracy_top5", top_k=5)
        return [loss]

    def add_optimizer(model):
        """
        Optimizer function called once for the entire model, as opposed for each 
        CPU / GPU individually. The optimizer will be a stepwise weight decay.

        :return: return the optimizer
        """
        stepsz = int(30 * args.epoch_size / args.batch_size / args.num_shards)

        optimizer.add_weight_decay(model, 1e-4)
        opt = optimizer.build_multi_precision_sgd(
            model,
            0.1,
            momentum=0.9,
            nesterov=1,
            policy="step",
            stepsize=stepsz,
            gamma=0.1
        )
        return opt

    def add_post_sync_ops(model):
        """
        Add ops applied after initial parameter sync.
        """
        for param_info in model.GetOptimizationParamInfo(model.GetParams()):
            if param_info.blob_copy is not None:
                model.param_init_net.HalfToFloat(
                    param_info.blob,
                    param_info.blob_copy[core.DataType.FLOAT]
                )

    if args.num_shards > 1:
        log.info("Distributed benchmarking is enabled")
        log.info("Num shards: %d", args.num_shards)
        log.info("My shard ID: %d", args.shard_id)
        log.info("Rendevous at: %s", args.rendezvous_path)

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

        # Parallelize the model (data parallel)
        data_parallel_model.Parallelize(
            evaluation_model,
            input_builder_fun=image_input,
            forward_pass_builder_fun=create_model,
            optimizer_builder_fun=None if not args.backward else (add_optimizer if not args.per_device_optimization else None),
            param_update_builder_fun=None if not args.backward else (AddParameterUpdate if args.per_device_optimization else None),
            post_sync_builder_fun=add_post_sync_ops if args.post_sync else None,
            devices=(args.gpu_devices if not args.use_cpu else [0]),
            rendezvous=rendezvous,
            # Although this is a parameter of this function, it is 
            # currently not implemented in Caffe2's source code  
            broadcast_computed_params=args.broadcast_params,
            optimize_gradient_memory=args.optimize_gradient_memory,
            dynamic_memory_management=args.dynamic_memory_management,
            max_concurrent_distributed_ops=args.max_distributed_ops,
            num_threads_per_device=args.max_threads,
            use_nccl=args.use_nccl,            
            cpu_device=args.use_cpu,
            ideep=args.use_ideep,
            shared_model=args.shared_model,
            combine_spatial_bn=args.use_cpu,
        )

        if args.backward:
            data_parallel_model.OptimizeGradientMemory(evaluation_model, {}, set(), False)
    else:
        print("Single node benchmarking is enabled")
        image_input(evaluation_model)
        create_model(evaluation_model, 1.0)
        if args.backward:
            AddParameterUpdate(evaluation_model)

        # TODO: I'm not too sure about this method; does it actually initiate a run, or does
        #       it just set some flat to run on GPU: https://caffe2.ai/doxygen-python/html/classcaffe2_1_1python_1_1core_1_1_net.html#af67e059d8f4cc22e7e64ccdd07918681
        if not args.use_cpu:
            evaluation_model.param_init_net.RunAllOnGPU()
            evaluation_model.net.RunAllOnGPU()

    # Initialize the model's parameters
    workspace.RunNetOnce(evaluation_model.param_init_net)
    # Create the network for later execution
    workspace.CreateNet(evaluation_model.net)
    # Run the network through benchmarks
    workspace.BenchmarkNet(evaluation_model.net.Proto().name, args.warmup_rounds, args.eval_rounds, args.per_layer_eval)


def main():
    # TODO: add way to measure time to certain training accuracy, and then stop.
    # TODO: further experiment with threads and number of parallel operations, and see if this changes the number of actual real threads being executed
    # TODO: found the num threads per device option it is in data_parallel_model.Parallelize method, and it's set to 4 by default. There are also options for shared model 
    #       (currently it's only data parallel)
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
        "--epoch_size",
        type=int,
        default=1000,
        help="The epoch size"
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
        default=32,
        help="The height of the image"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=32,
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
        default=32,
        help="The batch size"
    )
    parser.add_argument(
        "--backward",
        type=str_to_bool,
        default=True,
        help="Perform parameter updates"
    )
    parser.add_argument(
        "--per_device_optimization",
        type=str_to_bool,
        default=True,
        help="Perform the backward pass per device"
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
        help="Space separated list of GPU device names (on this node)",
        nargs='*'
    )
    parser.add_argument(
        "--use_ideep",
        type=str_to_bool,
        default=False,
        help="Use Intel's IDEEP"
    )
    parser.add_argument(
        "--broadcast_params",
        type=str_to_bool,
        default=True,
        help="Broadcast computed params"
    )
    parser.add_argument(
        "--optimize_gradient_memory",
        type=str_to_bool,
        default=False,
        help="Optimize gradient memory"
    )
    parser.add_argument(
        "--dynamic_memory_management",
        type=str_to_bool,
        default=False,
        help="Dynamic memory management"
    )
    parser.add_argument(
        "--max_distributed_ops",
        type=int,
        default=16,
        help="Dynamic memory management"
    )
    parser.add_argument(
        "--max_threads",
        type=int,
        default=4,
        help="The maximal number of threads per node"
    )
    parser.add_argument(
        "--use_nccl",
        type=str_to_bool,
        default=False,
        help="Flag for NCCL usage"
    )
    parser.add_argument(
        "--cudnn_ws_lim",
        type=int,
        default=128,
        help="cuDNN workspace limit (in MB)"
    )
    parser.add_argument(
        "--shared_model",
        type=str_to_bool,
        default=True,
        help="Shared model across the nodes / devices"
    )
    parser.add_argument(
        "--post_sync",
        type=str_to_bool,
        default=False,
        help="Add post synchronization operations"
    )
    
    args = parser.parse_args()

    # Ensure that caffe2 is initialized with a log level which does not 
    # dismiss the evaluation results (they are INFO level)
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=INFO'])
    network_eval(args)


if __name__ == '__main__':
    main()
