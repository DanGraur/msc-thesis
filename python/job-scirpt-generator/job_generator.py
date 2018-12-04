import jinja2

from argparse import ArgumentParser

TEMPLATE_FILE = "tf_job_template.job"


def create_template(template_params):
    template_loader = jinja2.FileSystemLoader(searchpath="./")
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(TEMPLATE_FILE)

    output_text = template.render(
        timeout=template_params.timeout,
        node_count=template_params.node_count,
        ps_nodes=template_params.ps_nodes,
        ps_tasks=template_params.ps_tasks,
        worker_tasks=template_params.worker_tasks,
        modules=template_params.modules,
        application=template_params.app,
        app_arguments=template_params.app_arguments,
        worker_node_count=template_params.node_count - template_params.ps_nodes
    )

    print(output_text)

    # to save the results
    with open(template_params.out_dir + '.job', "wb") as f:
        f.write(output_text)


if __name__ == '__main__':
    parser = ArgumentParser(description="Create a job script.")
    parser.add_argument("app",
                        type=str,
                        help="The path to the application being executed on SLURM."
                        )
    parser.add_argument("out_dir",
                        type=str,
                        help="The path to the output directory."
                        )
    parser.add_argument("-t", "--timeout",
                        dest="timeout",
                        default=None,
                        type=str,
                        help="After this timeframe all processes will be killed. Format: D-HH:MM:SS.",
                        nargs='?'
                        )
    parser.add_argument("-N", "--nodes",
                        default=3,
                        dest="node_count",
                        type=int,
                        help="Specifies the number of nodes to be used for this experiment.",
                        nargs='?'
                        )
    parser.add_argument("-pt", "--ps-tasks",
                        default=4,
                        type=int,
                        dest="ps_tasks",
                        help="Specifies the number of tasks per Parameter Server node.",
                        nargs='?'
                        )
    parser.add_argument("-wt", "--worker-tasks",
                        default=16,
                        type=int,
                        dest="worker_tasks",
                        help="Specifies the number of tasks per Worker node.",
                        nargs='?'
                        )
    parser.add_argument("-pn", "--ps-nodes",
                        default=1,
                        type=int,
                        dest="ps_nodes",
                        help="Specifies the number of Parameter Server nodes.",
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
    create_template(args)
