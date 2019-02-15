from json import loads


def auto_str(cls):
    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )

    cls.__str__ = __str__
    return cls


@auto_str
class ProcessDefinition(object):

    def __init__(self, args):
        # The name of this node
        self.nodename = args.nodename
        # The role of this node
        self.role = args.role
        # The rank of this process
        self.rank = args.rank
        # The number of PS nodes
        self.ps_number = args.ps_number
        # The cluster definition as expected by TF

        self.cluster_def = loads(args.cluster_def)
        # The nodes assigned for this task
        self.nodes = args.nodes

        # The possible parameters of this application
        self.app_arguments = args.app_arguments

        # Infer the number of PS and Worker tasks
        self.ps_tasks = len(self.cluster_def.get("ps", []))
        self.worker_tasks = len(self.cluster_def.get("worker", []))
