#!/usr/bin/python3

import subprocess
import os
import sys

from argparse import ArgumentParser
from datetime import datetime
from threading import Thread
from time import time, sleep


def str_to_bool(s):
    """
    Convert a string to boolean. Meant for the argparser.

    :return: the boolean converted from string
    """
    return s.lower() in ['yes', 'y', 'true', 't']


def construct_caffe2_benchmark_script_args(args):
    """
    Constructs a string containing the command line parameters of the Caffe2 benchmark script

    :param args: the obtained object after parsing the input command line, which contains the relevant information
                 for setting up the environment.
    :return: a string containing the command line parameters of the tf_cnn_benchmarks script
    """
    train_data = "''" if not args.training_data else args.training_data
    test_data = "''" if not args.testing_data else args.testing_data
    # Currently, we assume there's only one GPU per node, so, this value will always be '0'
    if args.gpu_mode:
        return "--model_name={} --training_data={} --testing_data={} --epoch_size={} --test_epoch_size={} " \
               "--test_accuracy={} --target_accuracy={} --epoch_count={} --num_labels={} --batch_size={} " \
               "--backward={} --per_device_optimization={} --warmup_rounds={} --eval_rounds={} --use_cpu={} " \
               "--num_shards={} --rendezvous_path={} --gpu_devices={} --shared_model={} --post_sync={} " \
               "--num_labels={} {}".format(args.model_name, train_data, test_data,
                                           args.epoch_size, args.test_epoch_size, args.test_accuracy,
                                           args.target_accuracy, args.epoch_count,
                                           args.num_labels, args.batch_size, args.backward,
                                           args.per_device_optimization, args.warmup_rounds,
                                           args.eval_rounds, not args.gpu_mode, args.node_count, args.rendezvous_path,
                                           0, args.shared_model, args.post_sync, args.num_labels, args.app_args)
    return "--model_name={} --training_data={} --testing_data={} --epoch_size={} --test_epoch_size={} " \
           "--test_accuracy={} --target_accuracy={} --epoch_count={} --num_labels={} --batch_size={} " \
           "--backward={} --per_device_optimization={} --warmup_rounds={} --eval_rounds={} --use_cpu={} " \
           "--num_shards={} --rendezvous_path={} --shared_model={} --post_sync={} --num_labels={} {}" \
           .format(args.model_name, test_data, train_data, args.epoch_size, args.test_epoch_size,
                   args.test_accuracy, args.target_accuracy, args.epoch_count, args.num_labels, args.batch_size,
                   args.backward, args.per_device_optimization, args.warmup_rounds, args.eval_rounds,
                   not args.gpu_mode, args.node_count, args.rendezvous_path, args.shared_model, args.post_sync,
                   args.num_labels, args.app_args)


def spawn_process(idx, node_name, modules, path_extension, venv_path, cd_path, app_path, benchmark_params,
                  base_log_name):
    print("Current idx and CPU node: {}, {}".format(idx, node_name))
    spawned_processes = []

    full_benchmark_params = benchmark_params + (" --shard_id={}".format(idx))
    cl = "ssh %s '%s export PATH=%s && source %s && cd %s && pwd; python %s %s'" % \
         (node_name, modules, path_extension, venv_path, cd_path, app_path, full_benchmark_params)

    print("cl >>", cl)
    print("path >>", cd_path)

    with open(base_log_name + ('_{}.out'.format(idx)), 'w') as fd:
        spawned_processes.append(
            subprocess.Popen(cl, stdout=fd, stderr=fd, shell=True)
        )

    return spawned_processes


def create_subcluster(args, base_log_name):
    """
    This function spawns part of a cluster processes (a subcluster), given the cluster definition, and a key in the
    cluster definition where one can find the required specification of the subcluster, by which it can be created.

    :param args: the obtained object after parsing the input command line, which contains the relevant information
                 for setting up the environment.
    :param base_log_name: a file name, which will be further expanded in order to generate a unique log file for
                          each process
    :return: Popen object, which represent handles to the (local) ssh processes used for spawning the tasks
    """
    assert 0 <= args.main_shard < args.node_count, "Must provide a valid main shard id in the range [0, node_count)"

    process_dict = {}

    cd_path = os.path.dirname(os.path.abspath(__file__))

    # Start creating the command line which loads the relevant modules
    modules_cl = ""
    for module in args.modules:
        modules_cl += "module load %s && " % module

    # Create the command line which extends the PATH environment variable
    path_extend_cl = "$PATH"
    for path in args.path_extensions:
        path_extend_cl = path + ':' + path_extend_cl

    # We'll build the arguments of the tf_cnn_benchmark script
    benchmark_params = construct_caffe2_benchmark_script_args(args)

    # We'll spawn the main shard process first, wait a bit, and then spawn the other shards as well
    # process_dict[args.cpu_nodes[args.main_shard]] = spawn_process(args.main_shard, args.cpu_nodes[args.main_shard],
    #                                                               modules_cl, path_extend_cl, args.caffe_venv, cd_path,
    #                                                               args.app_path, benchmark_params, base_log_name)
    # sleep(20)

    for idx, node_name in enumerate(args.cpu_nodes):
        if args.main_shard != idx:
            process_dict[node_name] = spawn_process(idx, node_name, modules_cl, path_extend_cl, args.caffe_venv,
                                                    cd_path, args.app_path, benchmark_params, base_log_name)

    return process_dict


def wait_for_proc(proc, timeout):
    """
    Wait for a process to terminate, given a timeout value. After the timeout, if the process hasn't already
    terminated, this method will exit.

    :param proc: the process handler (should be a Popen object)
    :param timeout: the timeout value
    :return: None
    """
    try:
        proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def trusty_sleep(n):
    """
    This method ensures the application goes to sleep (no busy waiting) for the
    specified amount of seconds, even if signals wake it up during its sleep.

    :param n: the number of seconds for which the application should sleep
    :return: None
    """
    end_time = time() + n

    while end_time > time():
        sleep(end_time - time())


def force_kill_procs(nodes, owner_name, app_name):
    """
    This function will terminate (by sending a SIGKILL) the processes of a particular
    type which belong to a particular user.

    :param nodes: A list of node aliases / addresses, which should be reachable by ssh
    :param owner_name: the name of the user whose processes will be killed
    :param app_name: the name of the application whose type will be terminated
    :return: A map of the processes killed, of the form (nodename -> [<pid_1>, ..., <pid_n>])
    """
    kill_map = {}

    for node in nodes:
        a = subprocess.Popen(["ssh", node, "ps -u %s | grep %s" % (owner_name, app_name)], stdout=subprocess.PIPE)
        output, _ = a.communicate()
        pids = output.split()[::4]

        kill_command = ';'.join(['kill -9 %s' % pid.decode('utf-8') for pid in pids])
        subprocess.Popen(["ssh", node, kill_command])

        if node in kill_map:
            kill_map[node].extend(pids)
        else:
            kill_map[node] = pids

    return kill_map


def create_cluster(args):
    """
    Create a cluster, by spawning a set of processes in a set of nodes, given a cluster configuration.

    :param args: the obtained object after parsing the input command line, which contains the relevant information
                 for setting up the environment.
    """
    base_log_name = "output-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    if args.gpu_mode:
        print("The GPU only mode is not yet implemented", file=sys.stderr)
        sys.exit(1)
    else:
        all_procs = create_subcluster(args, base_log_name)

    if args.timeout > 0 and args.gpu_mode:
        stop_threads = []

        # Wrap the processes in threads, and then wait (with timeout for the processes to end)
        for node in all_procs:
            for proc in all_procs[node]:
                thread = Thread(target=wait_for_proc, args=(proc, args.timeout))
                thread.start()
                stop_threads.append(thread)

        # Wait for the timeouts to run out or for the threads to naturally terminate
        for thread in stop_threads:
            thread.join()

        # Now kill the processes which may have remained active:
        return force_kill_procs(args.cpu_nodes, args.user, args.app_type)

    return {}


def main():
    parser = ArgumentParser(description="Spawn a cluster for a given Caffe2 application. Minimal usage example:"
                                        "./cluster_spawner_benchmark.py 2 --cpu_nodes node053 node054")
    parser.add_argument("node_count",
                        type=int,
                        help="Number of nodes involved in the benchmark",
                        )
    parser.add_argument("--cpu_nodes",
                        type=str,
                        help="A list of nodes which have CPU capabilities. This list will be used only if the "
                             "evaluation is CPU based",
                        nargs='*'
                        )
    parser.add_argument("--gpu_nodes",
                        type=str,
                        default=['node024', 'node025'],
                        help="A list of nodes which have GPU capabilities. This list will only be used if this is a "
                             "GPU based evaluation.",
                        nargs='*'
                        )
    parser.add_argument("--app_path",
                        type=str,
                        default='/home/dograur/tutorials/caffe2/resnet-benchmark/resnet_forward_benchmark.py',
                        help="The path to the tf benchmark script to be executed.",
                        )
    parser.add_argument("-t", "--timeout",
                        default=2 * 60,
                        type=int,
                        dest="timeout",
                        help="Specifies the timeout of the application (in seconds). If this is 0, there is no timeout."
                             "Beware that you will have to kill any remaining processes yourself.",
                        nargs='?'
                        )
    parser.add_argument("--gpu_mode",
                        type=str_to_bool,
                        default=False,
                        help="Indicates whether the benchmarking should be done in GPU mode or not"
                        )
    parser.add_argument("--main_shard",
                        type=int,
                        default=0,
                        help="Indicates which of the shards is the one responsible with synchronizing the others"
                        )

    # Parameters which are relevant to setting up the experiment context
    parser.add_argument("--modules",
                        default=[
                            'openblas/dynamic/0.2.18',
                            'cuda10.0/blas/10.0.130',
                            'cuda10.0/profiler/10.0.130',
                            'cuda10.0/toolkit/10.0.130',
                            'cuDNN/cuda90rc/7.0'
                        ],
                        type=str,
                        help="Space separated list of module files. E.g. python/2.7.13 "
                             "tensorflow/python2.7/cpu/r1.1.0-py2",
                        nargs='*'
                        )
    parser.add_argument("--path_extensions",
                        default=[
                            '/var/scratch/dograur/ffmpeg/ffmpeg-4.1/prefix/bin'
                        ],
                        type=str,
                        help="Space separated list of paths which should be added (at the front) of the $PATH "
                             "environment variable. E.g. /usr/tmp/asd /var/scratch/usr/mmm",
                        nargs='*'
                        )
    parser.add_argument("--caffe_venv",
                        default='/var/scratch/dograur/caffe2/venv/bin/activate',
                        type=str,
                        help="The path to where the Caffe2 virtual environment is located."
                        )

    # Actual benchmarking and Caffe2 parameters
    parser.add_argument("--data_format",
                        default='NHWC',
                        choices=['NCHW', 'NHWC'],
                        type=str,
                        help="Specifies the data's format",
                        )
    parser.add_argument("--model_name",
                        type=str,
                        default='resnet50',
                        help="The name / type of the model to be executed"
                        )
    parser.add_argument("--training_data",
                        type=str,
                        default='',
                        help="The path to the training data. If left empty, synthetic data will be used."
                        )
    parser.add_argument("--testing_data",
                        type=str,
                        default='',
                        help="The path to the testing data. Required during time to accuracy experiments."
                        )
    parser.add_argument("--backward",
                        type=str_to_bool,
                        default=True,
                        help="Perform parameter updates"
                        )
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="The batch size"
                        )
    parser.add_argument("--epoch_size",
                        type=int,
                        default=512,
                        help="The epoch size"
                        )
    parser.add_argument("--test_epoch_size",
                        type=int,
                        default=512,
                        help="The test epoch size"
                        )
    parser.add_argument("--test_accuracy",
                        type=str_to_bool,
                        default=False,
                        help="Enable testing for accuracy time"
                        )
    parser.add_argument("--target_accuracy",
                        type=float,
                        default=0.9,
                        help="The target test accuracy"
                        )
    parser.add_argument("--epoch_count",
                        type=int,
                        default=1,
                        help="The number of epochs to run"
                        )
    parser.add_argument("--num_labels",
                        type=int,
                        default=10,
                        help="The number of labels"
                        )
    parser.add_argument("--per_device_optimization",
                        type=str_to_bool,
                        default=True,
                        help="Perform the backward pass per device"
                        )
    parser.add_argument("--warmup_rounds",
                        type=int,
                        default=1,
                        help="The number of warmup rounds"
                        )
    parser.add_argument("--eval_rounds",
                        type=int,
                        default=1,
                        help="The number of evaluation rounds"
                        )
    parser.add_argument("--shard_id",
                        type=int,
                        default=0,
                        help="The shard ID of this node (0 based index)"
                        )
    parser.add_argument("--rendezvous_path",
                        type=str,
                        default='/var/scratch/dograur/caffe2_rendezvous/benchmarks/point',
                        help="Path to a rendezvous folder"
                        )
    parser.add_argument("--shared_model",
                        type=str_to_bool,
                        default=True,
                        help="Shared model across the nodes / devices"
                        )
    parser.add_argument("--post_sync",
                        type=str_to_bool,
                        default=False,
                        help="Add post synchronization operations"
                        )
    parser.add_argument("--app_args",
                        default="",
                        type=str,
                        help="Any additional arguments belonging to the Caffe2 application being run.",
                        )

    # Additional parameters required for termianting the benchmark
    parser.add_argument("--app_type",
                        default="resnet_forward_",
                        type=str,
                        help="Specifies which type of application will need to be killed when the timeout runs out",
                        nargs='?'
                        )
    parser.add_argument("--user",
                        default="dograur",
                        type=str,
                        help="Specifies the name of the user which is spawning the processes. This is useful for "
                             "determining which processes to kill when the timer runs out.",
                        nargs='?'
                        )

    args = parser.parse_args()
    create_cluster(args)


if __name__ == '__main__':
    main()
