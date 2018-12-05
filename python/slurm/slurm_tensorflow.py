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
        self.on_node_ps_tasks = int(os.getenv("PS_TASKS"))  # Get the number of PS processes spawned per node
        self.my_task_index = int(os.getenv("SLURM_PROCID"))  # Get my rank in the amongst spawned processes
        self.on_node_worker_tasks = int(os.getenv("WORKER_TASKS"))  # Get the number of WORKER processes spawned / node

        # Some sanity checks
        if len(self.nodelist) != self.num_nodes:
            raise ValueError("Number of slurm nodes {} not equal to {}".format(len(self.nodelist), self.num_nodes))

        if self.my_nodename not in self.nodelist:
            raise ValueError("Nodename({}) not in nodelist({}). This should not happen! ".format(self.my_nodename,
                                                                                                 self.nodelist))

        # Check to see if the user specified the number of PS nodes in the bash script; if so, replace the current val
        self.ps_number = int(os.environ.get("PS_NODE_COUNT", ps_number))

        # Get the nodes which will be Parameter Servers
        ps_nodes = self.nodelist[:self.ps_number]
        worker_nodes = self.nodelist[self.ps_number:]

        pre_config = {
            WORKER: [(name, self.on_node_worker_tasks, port_start) for name in worker_nodes],
            PARAMETER_SERVER: [(name, self.on_node_ps_tasks, port_start) for name in ps_nodes]
        }

        self.my_task_type = PARAMETER_SERVER if self.my_nodename in ps_nodes else WORKER

        self.cluster_definition = SlurmConfig.create_cluster_definition(pre_config)

    def __str__(self):
        to_write = ['nodelist', 'my_nodename', 'num_nodes', 'on_node_ps_tasks', 'on_node_worker_tasks', 'my_task_type',
                    'my_task_index', 'cluster_definition']
        return ', '.join(['{key}={value}\n'.format(key=key, value=self.__dict__.get(key)) for key in to_write])

    @property
    def task_type(self):
        """
        Return the type of this process (worker or ps)
        """
        return self.my_task_type

    @property
    def task_index(self):
        """
        Return the index in the process type
        """
        return self.my_task_index

    @property
    def cluster_configuration(self):
        """
        Return a dictionary indicating which processes are assigned to which roles
        """
        return self.cluster_definition

    @property
    def worker_tasks_per_node(self):
        """
        Return the number of worker processes there are for a worker node
        """
        return self.on_node_worker_tasks

    @property
    def ps_tasks_per_node(self):
        """
        Return the number of ps processes there are for a ps node
        """
        return self.on_node_ps_tasks

    @property
    def cluster_nodes(self):
        """
        Return the expanded list of nodes in the system
        """
        return self.nodelist

    @property
    def nodename(self):
        """
        Return the name of this node
        """
        return self.my_nodename

    @property
    def worker_processes_count(self):
        """
        Return the total number of worker processes
        """
        return len(self.cluster_definition[WORKER])

    @property
    def ps_processes_count(self):
        """
        Return the total number of ps processes
        """
        return len(self.cluster_definition[PARAMETER_SERVER])

    @property
    def worker_nodes_count(self):
        """
        Return the number of worker nodes
        """
        return self.num_nodes - self.ps_number

    @property
    def ps_nodes_count(self):
        """
        Return the number of ps nodes
        """
        return self.ps_number


if __name__ == '__main__':
    slurm_configuration = SlurmConfig()
    print(slurm_configuration)


