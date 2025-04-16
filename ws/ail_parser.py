import argparse
import builtins
from contextlib import nullcontext
from copy import deepcopy
import sys
import traceback
from types import ModuleType
import types
from typing import Any

from ail_fe_main_scmds import SCmd

Parser = argparse.ArgumentParser | argparse._ArgumentGroup


class DummyModule(ModuleType):
    def __getattr__(self, key):
        return DummyModule(name=f"{self.__name__}.{key}")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self

    def __mro_entries__(self, bases):
        return (self, self)

    def __getitem__(self, key):
        return self

    def __init__(self, name: str, doc: str | None = "", *rest) -> None:
        if type(name) is str and (doc is None or type(doc) is str):
            super().__init__(name, doc)

    __all__ = []  # support wildcard imports

    def __or__(self, value: Any, /) -> types.UnionType:
        return None | Any


class DummyImport:
    def __init__(self, dummy_pkgs: list[str], import_err: str) -> None:
        self.dummy_pkgs = [pkg.replace("-", "_") for pkg in dummy_pkgs]
        self.import_err = import_err
        self.realimport = builtins.__import__

    def __enter__(self):
        def tryimport(name, globals={}, locals={}, fromlist=[], level=0):
            pkg_name = name.split(".")[0].replace("-", "_")
            if pkg_name in self.dummy_pkgs:
                return DummyModule(name=name)
            try:
                return self.realimport(name, globals, locals, fromlist, level)
            except ImportError as err:
                print("(!) Tried to import", name, file=sys.stderr)
                print(self.import_err, file=sys.stderr)
                traceback.print_exception(err)
                raise err

        builtins.__import__ = tryimport

    def __exit__(self, exc_type, exc_value, traceback):
        builtins.__import__ = self.realimport


# orchestrate all parsers
def parse_intermixed_args(
    uninstalled_requirements=False,
    sys_args: list[str] = sys.argv[1:],
) -> tuple[argparse.Namespace, list[str]]:
    requirements_path = "./requirements.txt"
    with open(requirements_path, "r") as f:
        requirements = [pkg.split("==")[0] for pkg in f.read().splitlines()]
        requirements = [
            pkg[2:] if pkg.startswith("#:") else pkg for pkg in requirements
        ]
        requirements = [pkg for pkg in requirements if not pkg.startswith("#")]

    parser = argparse.ArgumentParser(
        add_help=False,
        prog="ail_run.sh",
        description="Run commands on the AI-Lab frontend.",
    )

    local_options = parser.add_argument_group("options for local")
    local_options.add_argument(
        "--no_sync",
        "-N",
        action="store_true",
        help="Do not sync the codebase before and after running the job.",
    )

    fe_options = parser.add_argument_group("options for the frontend")
    fe_options.add_argument(
        "--keep_jobs",
        "-k",
        action="store_true",
        help="Do not cancel previous jobs before running the new job.",
    )
    fe_options.add_argument(
        "--scmds_from",
        "-s",
        type=str,
        default="ail_fe_main_scmds",
        help="The Python file to import Slurm commands from.",
    )

    slurm_options = parser.add_argument_group("options for Slurm")
    slurm_options.add_argument(
        "--no_install",
        "-n",
        action="store_true",
        help="Do not install packages from requirements.txt before running the Python command.",
    )
    slurm_options.add_argument(
        "--torchrun",
        "-T",
        action="store_true",
        help="Use torchrun to run the Python command.",
    )

    (args, rest) = parser.parse_known_intermixed_args(sys_args)

    with (
        DummyImport(
            dummy_pkgs=requirements,
            import_err="Make sure to add the package to requirements.txt!",
        )
        if uninstalled_requirements
        else nullcontext()
    ):
        try:
            scmds_module = __import__(args.scmds_from)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"Incorrect --scmds_from: Module {args.scmds_from} does not exist."
            )
        if not hasattr(scmds_module, "get_scmds"):
            raise AttributeError(
                f"Module {args.scmds_from} does not have a get_scmds function."
            )
        if not hasattr(scmds_module, "modify_parser"):
            raise AttributeError(
                f"Module {args.scmds_from} does not have a modify_parser function."
            )
        scmds_module.modify_parser(fe_options)

        (args, rest) = parser.parse_known_intermixed_args(sys_args)

        scmds: list[SCmd] = scmds_module.get_scmds(args)

        slurm_options.add_argument(
            "-t",
            "--target",
            type=str,
            required=any(scmd.python_module is None for scmd in scmds),
            help="The Python target module to run in the Slurm job with torchrun.",
        )
        (args, rest) = parser.parse_known_intermixed_args(sys_args)

        for scmd in scmds:
            py_slurm_module_name = scmd.python_module or args.target
            try:
                py_slurm_module = __import__(py_slurm_module_name)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"Incorrect -t: Module {py_slurm_module_name} does not exist."
                )
            if not hasattr(py_slurm_module, "modify_parser"):
                raise AttributeError(
                    f"Module {py_slurm_module_name} does not have a modify_parser function."
                )
            parser_copy = deepcopy(parser)
            target_options = parser_copy.add_argument_group(
                "options for the target script"
            )
            py_slurm_module.modify_parser(target_options)
            parser_copy.add_argument(
                "--help",
                "-h",
                action="help",
                help="Show this help message and exit.",
            )
            parser_copy.parse_intermixed_args(sys_args + scmd.python_args)

    return (args, rest)
