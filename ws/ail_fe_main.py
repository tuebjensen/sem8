import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from importlib.util import find_spec

from ail_fe_main_scmds import SCmd
from ail_parser import parse_intermixed_args


def main(args: argparse.Namespace):
    if not os.path.isdir(".venv"):
        print("Creating virtual environment")
        subprocess.run(["python3", "-m", "venv", ".venv"])

    if not args.keep_jobs:
        result = subprocess.run(
            ["squeue", "--me", "--nohead", "--format", "%F"],
            capture_output=True,
            text=True,
        )
        jobs = set(result.stdout.split())
        for job in jobs:
            subprocess.run(["scancel", job])

    scmds: list[SCmd] = __import__(args.scmds_from).get_scmds(args)

    for scmd in scmds:
        module_name = scmd.python_module or args.target
        module_spec = find_spec(module_name)
        if module_spec is None:
            raise ImportError(f"Module {args.f} not found")
        module_path = module_spec.origin
        if module_path is None:
            raise ImportError(f"Module {args.f} not found (origin)")

        command = " ".join(
            [
                arg if i == 0 else "'" + arg.replace("'", "'\"'\"'") + "'"
                for (i, arg) in enumerate(
                    [
                        "srun",
                        *scmd.opts,
                        "singularity",
                        "exec",
                        "--nv",
                        "/ceph/container/pytorch/pytorch_24.09.sif",
                        "bash",
                        "ail_slurm_main.sh",
                        module_path,
                        *sys.argv[1:],
                        *scmd.python_args,
                    ]
                )
            ]
        )
        args_for_id = [
            str(datetime.now()),
            *sys.argv[1:],
            *scmd.python_args,
        ]
        id = (
            "'"
            + "-".join(args_for_id)
            .replace("'", "-")
            .replace("/", "-")
            .replace(" ", "-")
            + "'"
        )
        id = re.sub(r"-+", "-", id)
        # id = ""
        print("Running command", command)
        subprocess.run(
            command + " 2>&1 | tee output-" + id + ".log",
            shell=True,
            check=True,
        )


if __name__ == "__main__":
    args, rest = parse_intermixed_args(uninstalled_requirements=True)
    main(args)
