#!/usr/bin/python3

import os
import sys
import subprocess

from argparse import ArgumentParser
from datetime import datetime, timedelta
from threading import Thread
from time import time, sleep


def str_to_bool(s):
    """
    Convert a string to boolean. Meant for the argparser.

    :return: the boolean converted from string
    """
    return s.lower() in ['yes', 'y', 'true', 't']


class ClusterDefinition(object):
    """
    A class which holds the information relevant for setting up a TensorFlow cluster
    """

    @staticmethod
    def create_nodelist(name, task_count, start_port):
        """
        For a given nodename, create a list of processes, specifying and the port where
        the process' socket will be opened. The list will have the following structure:

        [
            "<name>:<start_port>",
            "<name>:<start_port + 1>",
            ...
            "<name>:<start_port + task_count - 1>"
        ]

        :param name: the nodename
        :param task_count: the number of processes / tasks
        :param start_port: the starting port
        :return:
        """
        return [":".join([name, str(start_port + i)]) for i in range(task_count)]

    def create_cluster_definition(self):
        """
        This method creates a cluster definition, as expected by TensorFlow, i.e.:

        {
            "worker" : [
                "<worker_node_name_1>:<port_1>",
                "<worker_node_name_1>:<port_2>",
                ...
                "<worker_node_name_1>:<port_m>",
                ...
                "<worker_node_name_n>:<port_m>"
            ],
            "ps" : [
                "<ps_node_name_1>:<port_1>",
                "<ps_node_name_1>:<port_2>",
                ...
                "<ps_node_name_1>:<port_r>",
                ...
                "<ps_node_name_k>:<port_r>"
            ]
        }

        :return: return a dictionary containing the cluster definition, which can be used by TensorFlow
        """
        cluster_definition = {}

        # Iterate through the keys
        for role in self.subcluster_def:
            accumulator_list = []

            # Iterate through the tuples, each representing a node
            for entry in self.subcluster_def[role]['nodes']:
                accumulator_list.extend(
                    ClusterDefinition.create_nodelist(entry, self.subcluster_def[role]["tasks_on_node"],
                                                      self.subcluster_def[role]["starting_port"]))

            cluster_definition[role] = accumulator_list

        return cluster_definition

    def partition_nodes(self, args):
        """
        Partitions the available nodes into a PS and Worker set, depending on the chosen architecture, and computation
        mode.

        :param args: the obtained object after parsing the input command line, which contains the relevant information
                     for setting up the environment.
        :return: two lists, one representing the PS nodes and the other representing the Worker nodes
        """
        if args.gpu_mode:
            self.nodes = args.cpu_nodes[:args.ps_nodes] + args.gpu_nodes[:(args.node_count - args.ps_nodes)]
        else:
            self.nodes = args.cpu_nodes[:args.node_count]

        self.cluster_size = len(self.nodes)

        if self.cluster_size != args.node_count:
            print("Incorrect number of resources provided, please make sure the node_count and the number of "
                  "provided nodes match")
            sys.exit(1)

        # TODO: currently only two architecture are supported, and for CPU only (should extend support for other stuff)
        if not args.gpu_mode:
            if args.architecture == 'ps':
                return self.nodes[:args.ps_nodes], self.nodes[args.ps_nodes:]
            elif args.architecture == 'colocated-ps':
                return self.nodes[:args.ps_nodes], self.nodes[:args.node_count]
        else:
            # Currently, only PS is supported for GPU based computation
            return self.nodes[:args.ps_nodes], self.nodes[args.ps_nodes:]

    def __init__(self, args):
        self.nodes = []
        self.cluster_size = 0

        ps_nodes_list, worker_nodes_list = self.partition_nodes(args)

        self.subcluster_def = {
            "ps": {
                "count": len(ps_nodes_list),
                "nodes": ps_nodes_list,
                "tasks_on_node": args.ps_tasks,
                "starting_port": args.ps_port
            },
            "worker": {
                "count": len(worker_nodes_list),
                "nodes": worker_nodes_list,
                "tasks_on_node": args.worker_tasks,
                "starting_port": args.worker_port
            }
        }

        # Cluster structure, as expected by TF
        self.cluster_structure = self.create_cluster_definition()


def construct_tf_benchmark_script_args(args, cluster_def, role):
    """
    Constructs a string containing the command line parameters of the tf_cnn_benchmarks script

    :param cluster_def: the cluster definition object (should be of ClusterDefinition type)
    :param role: a key in the cluster definition object, which points to the structure of the sublcuster
    :param args: the obtained object after parsing the input command line, which contains the relevant information
                 for setting up the environment.
    :return: a string containing the command line parameters of the tf_cnn_benchmarks script
    """
    device = 'CPU' if role == 'ps' or not args.gpu_mode else 'GPU'
    return "--data_format={} --batch_size={} --num_batches={} --data_name={} --model={} --optimizer={} " \
           "--variable_update={} --num_gpus={} --forward_only={} --print_training_accuracy={} " \
           "--display_every={}   --benchmark_log_dir={} --tfprof_file={} --local_parameter_device={} " \
           "--device={} --ps_hosts={} --worker_hosts={} --job_name={} --num_warmup_batches={} {} ".format(
           args.data_format,
           args.batch_size, args.num_batches, args.data_name, args.model, args.optimizer,
           args.variable_update, args.num_gpus, args.forward_only, args.print_training_accuracy,
           args.display_every, args.benchmark_log_dir, args.tfprof_file, device, device,
           ','.join(cluster_def.cluster_structure['ps']), ','.join(cluster_def.cluster_structure['worker']), role,
            args.num_warmup_batches, args.app_args)


def create_subcluster(cluster_def, subcluster_key, base_log_name, args):
    """
    This function spawns part of a cluster processes (a subcluster), given the cluster definition, and a key in the
    cluster definition where one can find the required specification of the subcluster, by which it can be created.
    This subcluster will be CPU based.

    :param cluster_def: the cluster definition object (should be of ClusterDefinition type)
    :param subcluster_key: a key in the cluster definition object, which points to the structure of the sublcuster
    :param base_log_name: a file name, which will be further expanded in order to generate a unique log file for
                          each process
    :param args: the obtained object after parsing the input command line, which contains the relevant information
                 for setting up the environment.
    :return: Popen object, which represent handles to the (local) ssh processes used for spawning the tasks
    """
    assert isinstance(cluster_def, ClusterDefinition), "The passed parameter must be of ClusterDefintion type"

    process_dict = {}

    subcluster_def = cluster_def.subcluster_def[subcluster_key]

    cd_path = os.path.dirname(os.path.abspath(__file__))

    # Start creating the command line which loads the relevant modules
    modules_cl = ""
    for module in args.modules:
        modules_cl += "module load %s && " % module

    # Create the command line which extends the PATH environment variable
    path_extend_cl = "$PATH"
    for path in args.path_extensions:
        path_extend_cl = path + ':' + path_extend_cl

    # Create the command line which extends the PATH environment variable
    pythonpath_extend_cl = "$PYTHONPATH"
    for path in args.pythonpath_extensions:
        pythonpath_extend_cl = pythonpath_extend_cl + ':' + path

    # We'll build the arguments of the tf_cnn_benchmark script
    benchmark_params = construct_tf_benchmark_script_args(args, cluster_def, subcluster_key)

    for idx, node_name in enumerate(subcluster_def['nodes']):
        opened_procs = []

        for local_process_rank in range(subcluster_def['tasks_on_node']):
            full_benchmark_params = benchmark_params + (" --task_index={}".format(
                local_process_rank + idx * subcluster_def['tasks_on_node']))
            cl = "ssh %s 'export PATH=%s && export PYTHONPATH=%s && source %s && cd %s && pwd; %s python %s %s'" % \
                 (node_name, path_extend_cl, pythonpath_extend_cl, args.tf_venv, cd_path, modules_cl, args.app_path,
                  full_benchmark_params)

            print("cl >>", cl)
            print("path >>", cd_path)

            with open(base_log_name + ('_{}.out'.format(idx)), 'w') as fd:
                opened_procs.append(
                    subprocess.Popen(cl, stdout=fd, stderr=fd, shell=True)
                )

        process_dict[node_name] = opened_procs

    return process_dict


def create_gpu_subcluster(cluster_def, subcluster_key, base_log_name, args):
    """
    This function spawns part of a cluster processes (a subcluster), given the cluster definition, and a key in the
    cluster definition where one can find the required specification of the subcluster, by which it can be created.
    This subcluster will be GPU based.

    :param cluster_def: the cluster definition object (should be of ClusterDefinition type)
    :param subcluster_key: a key in the cluster definition object, which points to the structure of the sublcuster
    :param base_log_name: a file name, which will be further expanded in order to generate a unique log file for
                          each process
    :param args: the obtained object after parsing the input command line, which contains the relevant information
                 for setting up the environment.
    :return: Popen object, which represent handles to the (local) ssh processes used for spawning the tasks
    """
    process_dict = {}
    subcluster_def = cluster_def.subcluster_def[subcluster_key]

    # Get the timeout of the batch job
    timeout = str(timedelta(seconds=args.timeout))

    # Construct the argument line for the tff_cnn_benchmarks
    benchmark_params = construct_tf_benchmark_script_args(args, cluster_def, subcluster_key)

    for idx, node_name in enumerate(subcluster_def['nodes']):
        opened_procs = []

        for local_process_rank in range(subcluster_def['tasks_on_node']):
            full_benchmark_params = benchmark_params + (" --task_index={}".format(
                local_process_rank + idx * subcluster_def['tasks_on_node']))

            cl = "sbatch -w {} -t {} {} {} '{}'".format(node_name, timeout, 'tf_gpu_run_distributed.job', timeout,
                                                  full_benchmark_params)
            print("cl >>", cl)

            with open(base_log_name + ('_{}.out'.format(idx)), 'w') as fd:
                opened_procs.append(
                    subprocess.Popen(cl, stdout=fd, stderr=fd, shell=True)
                )

        process_dict[node_name] = opened_procs

    return process_dict


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


def create_cluster(cluster_definition, args):
    """
    Create a cluster, by spawning a set of processes in a set of nodes, given a cluster configuration.

    :param cluster_definition: an object (of the type ClusterDefinition) which is used for configuring the cluster).
    :param args: the obtained object after parsing the input command line, which contains the relevant information
                 for setting up the environment.
    """
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    base_log_name_ps = "output-PS-{}".format(current_time)
    base_log_name_worker = "output-W-{}".format(current_time)

    ps_procs = create_subcluster(cluster_definition, "ps", base_log_name_ps, args)

    if args.gpu_mode:
        worker_procs = create_gpu_subcluster(cluster_definition, "worker", base_log_name_worker, args)
    else:
        worker_procs = create_subcluster(cluster_definition, "worker", base_log_name_worker, args)

    all_procs = ps_procs.copy()
    all_procs.update(worker_procs)

    if args.timeout > 0:
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
        return force_kill_procs(cluster_definition.nodes, args.user, args.app_type)

    return {}


def main():
    # Minimal usage example: ./cluster_spawner_benchmark.py 2 node053 node054  -pn 1 -pt 1 -wt 2
    parser = ArgumentParser(description="Spawn a cluster for a TensorFlow benchmark process. Minimal usage example:"
                                        "./cluster_spawner_benchmark.py 2 node053 node054  -pn 1 -pt 1 -wt 2")
    parser.add_argument("node_count",
                        type=int,
                        help="Number of nodes involved in the benchmark",
                        )
    parser.add_argument("cpu_nodes",
                        type=str,
                        help="A list of nodes which have CPU capabilities. There must be at least one node in the list"
                             "even if evaluation is GPU based. This node will be the PS. If training is CPU based, then"
                             "there must be at least as many nodes in the list as the node count",
                        nargs='+'
                        )
    parser.add_argument("--gpu_nodes",
                        type=str,
                        default=['node024', 'node025'],
                        help="A list of nodes which have GPU capabilities. This list will only be used if this is a "
                             "GPU base evaluation.",
                        nargs='*'
                        )
    parser.add_argument("--app_path",
                        type=str,
                        default='/home/dograur/tutorials/tensorflow/benchmarks/benchmarks/scripts/tf_cnn_benchmarks/'
                                'tf_cnn_benchmarks.py',
                        help="The path to the tf benchmark script to be executed.",
                        )
    parser.add_argument("-t", "--timeout",
                        default=2 * 60,
                        type=int,
                        dest="timeout",
                        help='Specifies the timeout of the application (in seconds). If this is 0, there is no timeout.'
                             'Beware that you will have to kill any remaining processes yourself.',
                        nargs='?'
                        )
    parser.add_argument("--gpu_mode",
                        type=str_to_bool,
                        default=False,
                        help="Indicates whether the benchmarking should be done in GPU mode or not"
                        )
    parser.add_argument("-pn", "--ps_nodes",
                        default=1,
                        type=int,
                        dest="ps_nodes",
                        help="Specifies the number of Parameter Server nodes.",
                        nargs='?'
                        )
    parser.add_argument("-pt", "--ps_tasks",
                        default=4,
                        type=int,
                        dest="ps_tasks",
                        help="Specifies the number of processes per Parameter Server node.",
                        nargs='?'
                        )
    parser.add_argument("-wt", "--worker_tasks",
                        default=4,
                        type=int,
                        dest="worker_tasks",
                        help="Specifies the number of processes per Worker node.",
                        nargs='?'
                        )
    parser.add_argument("--ps_port",
                        default=2222,
                        type=int,
                        help="Specifies the starting port on the PS servers.",
                        )
    parser.add_argument("--worker_port",
                        default=2222,
                        type=int,
                        help="Specifies the starting port on the Worker servers.",
                        )
    parser.add_argument("--architecture",
                        default='colocated-ps',
                        choices=['ps', 'colocated-ps'],
                        type=str,
                        help="Specifies the architecture used by distributed training. For details please see: https://github.com/tensorflow/benchmarks/blob/4536b7ce84aa4cdd246e7f5d389ea87017f0fc66/scripts/tf_cnn_benchmarks/benchmark_cnn.py#L523",
                        )

    # Parameters which are relevant to CPU based training
    parser.add_argument("--modules",
                        default=[
                            'openblas/dynamic/0.2.18',
                            'cuda10.0/blas/10.0.130',
                            'cuda10.0/profiler/10.0.130',
                            'cuda10.0/toolkit/10.0.130',
                            'cuDNN/cuda10.0/7.4',
                        ],
                        type=str,
                        help="Space separated list of module files. E.g. python/2.7.13 "
                             "tensorflow/python2.7/cpu/r1.1.0-py2",
                        nargs='*'
                        )
    parser.add_argument("--path_extensions",
                        default=[
                            '/home/dograur/tutorials/tensorflow/model',
                            '/home/dograur/tutorials/tensorflow/models/slim/',
                            '/var/scratch/dograur/ffmpeg/ffmpeg-4.1/prefix/bin',
                        ],
                        type=str,
                        help="Space separated list of paths which should be added (at the front) of the $PATH "
                             "environment variable. E.g. /usr/tmp/asd /var/scratch/usr/mmm",
                        nargs='*'
                        )
    parser.add_argument("--pythonpath_extensions",
                        default=[
                            '/home/dograur/tutorials/tensorflow/models',
                            '/home/dograur/tutorials/tensorflow/models/slim/',
                        ],
                        type=str,
                        help="Space separated list of paths which should be added (at the end) of the $PYTHONPATH "
                             "environment variable. E.g. /usr/tmp/asd /var/scratch/usr/mmm",
                        nargs='*'
                        )
    parser.add_argument("--tf_venv",
                        default='/var/scratch/dograur/tensorflow/venv_tf/bin/activate',
                        type=str,
                        help="The path to where the TF virtual environment is located."
                        )

    # The following are parameters relevant to the banchmarking script
    parser.add_argument("--data_format",
                        default='NHWC',
                        choices=['NCHW', 'NHWC'],
                        type=str,
                        help="Specifies the data's format",
                        )
    parser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        help="The size of one batch per one device",
                        )
    parser.add_argument("--num_warmup_batches",
                        default=5,
                        type=int,
                        help="The number of warmup batches.",
                        )
    parser.add_argument("--num_batches",
                        default=10,
                        type=int,
                        help="The number of batches to be run and averaged over.",
                        )
    parser.add_argument("--data_name",
                        default='cifar10',
                        choices=['cifar10', 'imagenet'],
                        type=str,
                        help="Specifies the type of synthetic data to be used.",
                        )
    parser.add_argument("--model",
                        default='alexnet',
                        type=str,
                        help="Specifies the NN's model which we are testing",
                        )
    parser.add_argument("--optimizer",
                        default='momentum',
                        type=str,
                        help="Specifies the optimizer to be used during evaluation",
                        )
    parser.add_argument("--variable_update",
                        default='parameter_server',
                        type=str,
                        help="Specifies the distributed learning architecture",
                        )
    parser.add_argument("--num_gpus",
                        default=1,
                        type=int,
                        help="Specifies the number of GPUs / CPUs per device",
                        )
    parser.add_argument("--forward_only",
                        default=False,
                        type=str_to_bool,
                        help="Specifies if parameter updates are computed in addition to the loss",
                        )
    parser.add_argument("--print_training_accuracy",
                        default=True,
                        type=str_to_bool,
                        help="Specifies if the training accuracy should be printed during evaluation",
                        )
    parser.add_argument("--display_every",
                        default=2,
                        type=int,
                        help="Specifies the number of steps between each status print in the evaluation",
                        )
    parser.add_argument("--benchmark_log_dir",
                        default='/home/dograur/tutorials/tensorflow/benchmarks/benchmarks/scripts/'
                                'tf_cnn_benchmarks/logs',
                        type=str,
                        help="Specifies the path to the directory where the logs will be stored",
                        )
    parser.add_argument("--tfprof_file",
                        default='/home/dograur/tutorials/tensorflow/benchmarks/benchmarks/scripts/tf_cnn_benchmarks/'
                                'logs/tfprof_logs',
                        type=str,
                        help="Specifies the path to where the TF profiler logs will be stored.",
                        )
    parser.add_argument("--app_args",
                        default="",
                        type=str,
                        help="Additional arguments which should be added to the tf_cnn_benchmarks command line."
                        )

    # The following are parameters which are used for force killing the resources alive after the timeout has ended

    # It might look like tf_cnn_benchmar has a typo, but due to a character limit, the 'ks' is removed
    parser.add_argument("--app_type",
                        default="tf_cnn_benchmar",
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
    create_cluster(ClusterDefinition(args), args)


if __name__ == '__main__':
    main()
