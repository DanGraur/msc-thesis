#!/usr/bin/python3

import subprocess
import os

from argparse import ArgumentParser
from datetime import datetime
from threading import Thread
from time import time, sleep


class ClusterDefinition(object):
    """
    A class which holds the information relevant for setting up a Caffe2 cluster
    """
    def __init__(self, args):
        # Node counts
        self.nodes = args.cluster_nodes
        self.cluster_size = len(self.nodes)

        # Get the admin stuff parameters
        self.app = args.app
        self.app_args = args.app_arguments
        self.timeout = args.timeout
        self.app_type = args.app_kill_type
        self.user = args.user

        self.modules = args.modules

        self.venv_location = args.caffe_venv
        self.path_extensions = args.path_extensions


def create_subcluster(cluster_def, file_descriptor):
    """
    This function spawns part of a cluster processes (a subcluster), given the cluster definition, and a key in the
    cluster definition where one can find the required specification of the subcluster, by which it can be created.

    :param cluster_def: the cluster definition object (should be of ClusterDefinition type)
    :param file_descriptor: a file descriptor where the output of the processes spawned in the subcluster should
                            be redirected
    :return: Popen object, which represent handles to the (local) ssh processes used for spawning the tasks
    """
    assert isinstance(cluster_def, ClusterDefinition), "The passed parameter must be of ClusterDefinition type"

    process_dict = {}

    cd_path = os.path.dirname(os.path.abspath(__file__))

    # Start creating the command line which loads the relevant modules
    modules_cl = ""
    for module in cluster_def.modules:
        modules_cl += "module load %s && " % module

    # Create the command line which extends the PATH environment variable
    path_extend_cl = "$PATH"
    for path in cluster_def.path_extensions:
        path_extend_cl = path + ':' + path_extend_cl

    for idx, node_name in enumerate(cluster_def.nodes):
        opened_procs = []

        # The following line will change the working dir here; it will load the relevant modules, and run the app
        # might need to be loaded before running the script.
        cl = "ssh %s '%s export PATH=%s && source %s && cd %s && pwd; python %s %s'" % \
             (node_name, modules_cl, path_extend_cl, cluster_def.venv_location, cd_path, cluster_def.app,
              cluster_def.app_args % (cluster_def.cluster_size, idx))

        print("cl >>", cl)
        print("path >>", cd_path)

        opened_procs.append(
            subprocess.Popen(cl, stdout=file_descriptor, stderr=file_descriptor, shell=True)
        )

        process_dict[node_name] = opened_procs

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


def create_cluster(cluster_definition):
    """
    Create a cluster, by spawning a set of processes in a set of nodes, given a cluster configuration.

    :param cluster_definition: an object (of the type ClusterDefinition) which is used for configuring the cluster).
    """
    with open("output-%s.out" % datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), "w") as f:
        all_procs = create_subcluster(cluster_definition, f)

    if cluster_definition.timeout > 0:
        stop_threads = []

        # Wrap the processes in threads, and then wait (with timeout for the processes to end)
        for node in all_procs:
            for proc in all_procs[node]:
                thread = Thread(target=wait_for_proc, args=(proc, cluster_definition.timeout))
                thread.start()
                stop_threads.append(thread)

        # Wait for the timeouts to run out or for the threads to naturally terminate
        for thread in stop_threads:
            thread.join()

        # Now kill the processes which may have remained active:
        return force_kill_procs(cluster_definition.nodes, cluster_definition.user, cluster_definition.app_type)

    return {}


def main():
    parser = ArgumentParser(description="Spawn a cluster for a given Caffe2 application.")
    parser.add_argument("app",
                        type=str,
                        help="The path to the application being executed on the cluster (relative the current location"
                             "of the shell."
                        )
    parser.add_argument("cluster_nodes",
                        type=str,
                        help="The aliases of the nodes which define the cluster.",
                        nargs='+'
                        )
    parser.add_argument("-t", "--timeout",
                        default=2 * 60,
                        type=int,
                        dest="timeout",
                        help="Specifies the timeout of the application (in seconds). If this is 0, there is no timeout."
                             "Beware that you will have to kill any remaining processes yourself.",
                        nargs='?'
                        )
    parser.add_argument("-modules", "--module-files",
                        default=[
                            'openblas/dynamic/0.2.18',
                            'cuda10.0/blas/10.0.130',
                            'cuda10.0/profiler/10.0.130',
                            'cuda10.0/toolkit/10.0.130',
                            'cuDNN/cuda90rc/7.0'
                        ],
                        type=str,
                        dest="modules",
                        help="Space separated list of module files. E.g. python/2.7.13 "
                             "tensorflow/python2.7/cpu/r1.1.0-py2",
                        nargs='*'
                        )
    parser.add_argument("-path_extensions", "--path-extensions",
                        default=[
                            '/var/scratch/dograur/ffmpeg/ffmpeg-4.1/prefix/bin'
                        ],
                        type=str,
                        dest="path_extensions",
                        help="Space separated list of paths which should be added (at the front) of the $PATH "
                             "environment variable. E.g. /usr/tmp/asd /var/scratch/usr/mmm",
                        nargs='*'
                        )
    parser.add_argument("-caffe_venv", "--caffe_venv",
                        default='/var/scratch/dograur/caffe2/venv/bin/activate',
                        type=str,
                        dest="caffe_venv",
                        help="The path to where the Caffe2 virtual environment is located."
                        )
    parser.add_argument("-args", "--app-args",
                        default="",
                        type=str,
                        dest="app_arguments",
                        help="The arguments belonging to the Caffe2 application being run.",
                        required=True
                        )
    parser.add_argument("-apptype", "--app-type",
                        default="python",
                        type=str,
                        dest="app_kill_type",
                        help="Specifies which type of application will need to be killed when the timeout runs out",
                        nargs='?'
                        )
    parser.add_argument("-user", "--user",
                        default="dograur",
                        type=str,
                        dest="user",
                        help="Specifies the name of the user which is spawning the processes. This is useful for "
                             "determining which processes to kill when the timer runs out.",
                        nargs='?'
                        )

    args = parser.parse_args()
    create_cluster(ClusterDefinition(args))


if __name__ == '__main__':
    main()
