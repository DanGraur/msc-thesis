import os
import sys

from hostlist.hostlist import expand_hostlist

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: python assign_nodes.py <number_of_PS_nodes : int>"
        sys.exit(1)

    # Get the number of ps nodes
    ps_node_count = int(sys.argv[1])

    # Get all the nodes assigned to this batch job
    nodelist = expand_hostlist(os.environ["SLURM_JOB_NODELIST"])

    # These will be executed if we use eval in the outer script
    print("export %s=%s" % ("PS_NODES", ','.join(nodelist[:ps_node_count])))
    print("export %s=%s" % ("WORKER_NODES", ','.join(nodelist[ps_node_count:])))
