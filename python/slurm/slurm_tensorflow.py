from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from hostlist.hostlist import expand_hostlist

import os


# The key used for the Worker nodes
WORKER = "worker"
# The key used for the Parameter Server nodes
PARAMETER_SERVER = "ps"


class SlurmConfig(object):

    @staticmethod
    def create_nodelist(name, task_count, start_port):
        return [":".join([name, str(start_port + i)]) for i in range(task_count)]

    @staticmethod
    def create_cluster_definition(serv_work_dict):
        """
        Create a dictionary which is used by TF to define the cluster

        :param serv_work_dict: a dictionary of the following form {("ps"|"worker") :
                               [(node_name, tasks_per_node, start_port)*]}
        :return: the dictionary which defines the structure of the cluster
        """
        cluster_definition = {}

        # Iterate through the keys
        for role_name in serv_work_dict:
            accumulator_list = []

            # Iterate through the tuples, each representing a node
            for entry in serv_work_dict[role_name]:
                accumulator_list.extend(SlurmConfig.create_nodelist(entry[0], entry[1], entry[2]))

            cluster_definition[role_name] = accumulator_list

        return cluster_definition

    def __init__(self, ps_number=1, port_start=22222):
        """
        Create a configuration object for the SLURM cluster

        :param ps_number: the number of Parameter Server nodes
        :param port_start: the starting ports of the for the processes / tasks
        """
        self.nodelist = expand_hostlist(os.environ["SLURM_JOB_NODELIST"])  # This returns the expanded node names
        self.my_nodename = os.environ["SLURMD_NODENAME"]  # This returns my node name
        self.num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES"))  # This returns the number of nodes assigned to this job
        self.on_node_tasks = int(os.getenv("SLURM_NTASKS_PER_NODE"))  # Get the number of processes spawned per machine
        task_id = int(os.getenv("SLURM_PROCID"))  # Get my rank in the amongst spawned processes

        # Some sanity checks
        if len(self.nodelist) != self.num_nodes:
            raise ValueError("Number of slurm nodes {} not equal to {}".format(len(self.nodelist), self.num_nodes))

        if self.my_nodename not in self.nodelist:
            raise ValueError("Nodename({}) not in nodelist({}). This should not happen! ".format(self.my_nodename,
                                                                                                 self.nodelist))

        # Get the nodes which will be Parameter Servers
        ps_nodes = self.nodelist[:ps_number]

        # Get the nodes which will be Workers
        worker_nodes = self.nodelist[ps_number:]

        pre_config = {
            WORKER: [(name, self.on_node_tasks, port_start) for name in worker_nodes],
            PARAMETER_SERVER: [(name, self.on_node_tasks, port_start) for name in ps_nodes]
        }

        self.my_task_type = PARAMETER_SERVER if self.my_nodename in ps_nodes else WORKER
        self.my_task_index = task_id if self.my_nodename in ps_nodes else task_id - self.on_node_tasks * ps_number

        self.cluster_definition = SlurmConfig.create_cluster_definition(pre_config)

    @property
    def task_type(self):
        return self.my_task_type

    @property
    def task_index(self):
        return self.my_task_index

    @property
    def cluster_configuration(self):
        return self.cluster_definition

    @property
    def tasks_per_node(self):
        return self.on_node_tasks

    @property
    def cluster_nodes(self):
        return self.nodelist

    @property
    def nodename(self):
        return self.my_nodename

    @property
    def worker_count(self):
        return len(self.cluster_definition[WORKER])

    @property
    def ps_count(self):
        return len(self.cluster_definition[PARAMETER_SERVER])


if __name__ == '__main__':
    slurm_configuraiton = SlurmConfig()
