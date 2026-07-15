from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

KERNELS_DIR = Path(__file__).resolve().parent / "kernels"


def kernel_folders() -> list[str]:
    return sorted(
        path.name
        for path in KERNELS_DIR.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    )


def run_kernel_script(kernel: str, script_name: str, script_args: list[str]) -> None:
    kernel_dir = (KERNELS_DIR / kernel).resolve()
    script_path = kernel_dir / f"{script_name}.py"

    if not kernel_dir.is_dir():
        valid = ", ".join(kernel_folders())
        raise SystemExit(f"Unknown kernel folder: {kernel}. Valid kernels: {valid}")
    if not script_path.is_file():
        raise SystemExit(f"{kernel} has no {script_name}.py")

    sys.path.insert(0, str(kernel_dir))
    old_argv = sys.argv
    sys.argv = [str(script_path), *script_args]
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv
        try:
            sys.path.remove(str(kernel_dir))
        except ValueError:
            pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gemm-metal",
        description="Run MLX/Metal kernel tests and benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List kernel experiment folders")

    for command in ("test", "bench"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("kernel", help="kernel folder under gemm_metal/kernels")
        subparser.add_argument("script_args", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)

    if args.command == "list":
        for kernel in kernel_folders():
            print(kernel)
        return

    run_kernel_script(args.kernel, args.command, args.script_args)


if __name__ == "__main__":
    main()
