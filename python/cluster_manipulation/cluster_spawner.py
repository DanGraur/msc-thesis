#!/usr/bin/python3

import subprocess
import os

from argparse import ArgumentParser
from datetime import datetime
from json import dumps
from threading import Thread
from time import time, sleep


def break_tuple(func):
    """
    Function wrapper. Breaks up a tuple into its component parameters. Tuple can have
    a variable amount of elements.

    :return: the result of the call with the decomposed tuple
    """
    def wrapper(tup):
        return func(*list(tup))
    return wrapper


class ClusterDefinition(object):
    """
    A class which holds the information relevant for setting up a cluster
    """

    @staticmethod
    def create_nodelist(name, task_count, start_port):
        return [":".join([name, str(start_port + i)]) for i in range(task_count)]

    def create_cluster_definition(self):
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

    def __init__(self, args):
        self.nodes = args.cluster_nodes
        self.cluster_size = len(self.nodes)

        self.subcluster_def = {
            "ps": {
                "count": args.ps_nodes,
                "nodes": self.nodes[:args.ps_nodes],
                "tasks_on_node": args.ps_tasks,
                "starting_port": args.ps_port
            },
            "worker": {
                "count": self.cluster_size - args.ps_nodes,
                "nodes": self.nodes[args.ps_nodes:],
                "tasks_on_node": args.worker_tasks,
                "starting_port": args.worker_port
            }
        }

        # Get the admin stuff parameters
        self.app = args.app
        self.app_args = args.app_arguments
        self.timeout = args.timeout
        self.app_type = args.app_kill_type
        self.user = args.user

        self.modules = args.modules

        # Cluster structure, as expected by TF
        self.cluster_structure = self.create_cluster_definition()


def create_subcluster(cluster_def, subcluster_key, file_descriptor):
    assert isinstance(cluster_def, ClusterDefinition), "The passed parameter must be of ClusterDefintion type"

    process_dict = {}

    subcluster_def = cluster_def.subcluster_def[subcluster_key]

    cd_path = os.path.dirname(os.path.abspath(__file__))

    # Start creating the command line which loads the relevant modules
    modules_cl = ""
    for module in cluster_def.modules:
        modules_cl += "module load %s; " % module

    for idx, node_name in enumerate(subcluster_def['nodes']):
        opened_procs = []

        for local_process_rank in range(subcluster_def['tasks_on_node']):
            process_rank = local_process_rank + idx * subcluster_def['tasks_on_node']
            # The following line will change the working dir here; it will load the relevant modules, and run the app
            cl = "ssh %s 'cd %s; pwd; %s python %s %s %s %d %d \"%s\" %s'" % \
                 (node_name, cd_path, modules_cl, cluster_def.app, node_name, subcluster_key, process_rank,
                  cluster_def.subcluster_def['ps']['count'], dumps(cluster_def.cluster_structure).replace('"', '\\"').replace("'", "\\'"),
                  ' '.join(cluster_def.nodes))

            print("cl >>", cl)
            print("path >>", cd_path)
            print("structure >>", dumps(cluster_def.cluster_structure).replace('"', '\\"').replace("'", "\\'"))

            opened_procs.append(
                subprocess.Popen(cl, stdout=file_descriptor, stderr=file_descriptor, shell=True)
            )

        process_dict[node_name] = opened_procs

    return process_dict


def wait_for_proc(proc, timeout):
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
    cluster_definition = ClusterDefinition(args)

    with open("output-%s.out" % datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), "w") as f:
        ps_procs = create_subcluster(cluster_definition, "ps", f)
        worker_procs = create_subcluster(cluster_definition, "worker", f)

    all_procs = ps_procs.copy()
    all_procs.update(worker_procs)

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
    parser = ArgumentParser(description="Spwan a cluster for a given application.")
    parser.add_argument("app",
                        type=str,
                        help="The path to the application being executed on SLURM."
                        )
    parser.add_argument("cluster_nodes",
                        type=str,
                        help="The aliases of the nodes which define the cluster.",
                        nargs='+'
                        )
    parser.add_argument("-pt", "--ps-tasks",
                        default=4,
                        type=int,
                        dest="ps_tasks",
                        help="Specifies the number of processes per Parameter Server node.",
                        nargs='?'
                        )
    parser.add_argument("-t", "--timeout",
                        default=2 * 60,
                        type=int,
                        dest="timeout",
                        help="Specifies the timeout of the application (in seconds). If this is 0, there is no timeout."
                             "Beware that you will have to kill any remaining processes yourself.",
                        nargs='?'
                        )
    parser.add_argument("-wt", "--worker-tasks",
                        default=16,
                        type=int,
                        dest="worker_tasks",
                        help="Specifies the number of processes per Worker node.",
                        nargs='?'
                        )
    parser.add_argument("-pn", "--ps-nodes",
                        default=1,
                        type=int,
                        dest="ps_nodes",
                        help="Specifies the number of Parameter Server nodes.",
                        nargs='?'
                        )
    parser.add_argument("-pport", "--ps-port",
                        default=2222,
                        type=int,
                        dest="ps_port",
                        help="Specifies the starting port on the PS servers.",
                        nargs='?'
                        )
    parser.add_argument("-wport", "--worker-port",
                        default=2222,
                        type=int,
                        dest="worker_port",
                        help="Specifies the starting port on the Worker servers.",
                        nargs='?'
                        )
    parser.add_argument("-modules", "--module-files",
                        default=['tensorflow/python2.7/cpu/r1.1.0-py2'],
                        type=str,
                        dest="modules",
                        help="Space separated list of module files. E.g. python/2.7.13, "
                             "tensorflow/python2.7/cpu/r1.1.0-py2",
                        nargs='*'
                        )
    parser.add_argument("-args", "--app-args",
                        default="",
                        type=str,
                        dest="app_arguments",
                        help="The arguments belonging to the application being run over SLURM.",
                        nargs='*'
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
    create_cluster(args)


if __name__ == '__main__':
    main()
