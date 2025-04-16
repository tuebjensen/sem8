import os
import subprocess
import sys

from ail_parser import parse_intermixed_args

if __name__ == "__main__":
    # cut off the first argument, which is the Python file path
    args, rest = parse_intermixed_args(
        sys_args=sys.argv[2:], uninstalled_requirements=True
    )

    if not args.no_install:
        subprocess.call([sys.executable, "-m", "ensurepip"])
        subprocess.call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "-r",
                "requirements.txt",
            ],
            # set the TMPDIR environment variable to a directory in the current
            # directory to avoid storage limit issues on the /tmp directory
            env=(dict(os.environ) | {"TMPDIR": "./pip-cache"}),
        )

    import torch

    python_args = [sys.argv[1]] + rest
    gpu_count = torch.cuda.device_count()
    command = (
        ["torchrun", "--nproc_per_node=" + str(gpu_count)] + python_args
        if args.torchrun
        else ["python3", "-u"] + python_args
    )
    print("Running command: ", command)
    subprocess.run(command)
