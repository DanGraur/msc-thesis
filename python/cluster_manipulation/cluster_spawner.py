#!/usr/bin/python3

import subprocess
import os

from datetime import datetime
from argparse import ArgumentParser
from json import dumps
from threading import Thread

# This should be in seconds
TIMEOUT = 5


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
    proc.communicate(timeout=timeout)


def force_kill_procs(process_map):
    for node in process_map:
        for proc in process_map[node]:
            subprocess.Popen(["ssh", node, "kill -9 %d" % proc.pid])


def create_cluster(args):
    cluster_definition = ClusterDefinition(args)

    with open("output-%s.out" % datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), "w") as f:
        ps_procs = create_subcluster(cluster_definition, "ps", f)
        worker_procs = create_subcluster(cluster_definition, "worker", f)

    all_procs = ps_procs.copy()
    all_procs.update(worker_procs)

    stop_threads = []

    # Wrap the processes in threads, and then wait (with timeout for the processes to end)
    for node in all_procs:
        for proc in all_procs[node]:
            thread = Thread(target=wait_for_proc, args=(proc, cluster_definition.timeout))
            thread.start()
            stop_threads.append(thread)

    # We wait for the threads to end, i.e. for the timeouts to run out or for the
    # processes to naturally come to an end
    for thread in stop_threads:
        thread.join()

    # We can now force kill any running processes, since we know they have timed out (if they're still running)
    force_kill_procs(all_procs)

    return all_procs


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
                        help="Specifies the timeout of the application (in seconds).",
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

    args = parser.parse_args()
    create_cluster(args)


if __name__ == '__main__':
    main()
