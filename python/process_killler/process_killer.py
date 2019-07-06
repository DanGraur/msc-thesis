#!/usr/bin/python3

import subprocess

from argparse import ArgumentParser


def get_pids(app_name, nodes, user):
    """
    Get the PIDs of the specified application of the specified owner spawned on the specified nodes

    :param app_name: the name of the application
    :param nodes: a list of node aliases / addresses, which should be reachable by ssh
    :param user: the name of the process owner
    :return: a map from nodename to a list of pids
    """
    pid_map = {}

    for node in nodes:
        cl = "ssh {} '{}'".format(node, "ps -u {} | grep {}".format(user, app_name))
        a = subprocess.Popen(cl, stdout=subprocess.PIPE, shell=True)
        output, _ = a.communicate()
        pids = output.split()[::4]

        pid_map[node] = pids

    return pid_map


def kill_processes(app_name, nodes, user):
    """
    This function will terminate (by sending a SIGKILL) the processes of a particular
    type which belong to a particular user.

    :param app_name: the name of the application whose type will be terminated
    :param nodes: A list of node aliases / addresses, which should be reachable by ssh
    :param user: the name of the user whose processes will be killed
    :return: A map of the processes killed, of the form (nodename -> [<pid_1>, ..., <pid_n>])
    """
    pid_map = get_pids(app_name, nodes, user)

    for node in nodes:
        kill_command = ';'.join(['kill -9 %s' % pid.decode('utf-8') for pid in pid_map[node]])
        subprocess.Popen(["ssh", node, kill_command])

    return pid_map


if __name__ == '__main__':
    parser = ArgumentParser(description="Spawn a cluster for a given Caffe2 application. Minimal usage example:"
                                        "./process_killer.py app_name node001 node002")
    parser.add_argument("app_name",
                        type=str,
                        help="The name of the application which needs to be killed",
                        )
    parser.add_argument("nodes",
                        type=str,
                        help="A list of nodes where the processes reside",
                        nargs='+'
                        )
    parser.add_argument("--user",
                        type=str,
                        default='dograur',
                        help="The user to which the processes belong"
                        )
    args = parser.parse_args()

    app_name = args.app_name
    nodes = args.nodes
    user = args.user

    print(kill_processes(app_name, nodes, user))
