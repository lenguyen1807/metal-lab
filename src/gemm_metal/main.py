import argparse
from datetime import datetime
from pathlib import Path

from gemm_metal.bench.common import BenchSpec, benchmark, print_result, selected, write_csv


def specs() -> dict[str, BenchSpec]:
    from gemm_metal.bench.gemm import SPEC as gemm
    from gemm_metal.bench.vadd import SPEC as vadd

    return {spec.op: spec for spec in [vadd, gemm]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gemm-metal")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List ops and kernels")

    validate = sub.add_parser("validate", help="Validate one kernel on the first square case")
    validate.add_argument("op", choices=sorted(specs()))
    validate.add_argument("--kernel")
    validate.add_argument("--seed", type=int, default=0)

    bench = sub.add_parser("bench", help="Benchmark one op")
    bench.add_argument("op", choices=sorted(specs()))
    bench.add_argument("--kernel")
    bench.add_argument("--suite", default="square", choices=["square", "full", "stress", "extreme"])
    bench.add_argument("--warmups", type=int, default=5)
    bench.add_argument("--iterations", type=int, default=20)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--output")

    return parser


def main() -> None:
    all_specs = specs()
    args = build_parser().parse_args()

    if args.command == "list":
        for op, spec in sorted(all_specs.items()):
            print(f"{op}: {', '.join(sorted(spec.kernels))}")
        return

    spec = all_specs[args.op]
    kernel = args.kernel or spec.default_kernel
    if kernel not in spec.kernels:
        valid = ", ".join(sorted(spec.kernels))
        raise SystemExit(f"unknown kernel {kernel!r}; valid: {valid}")

    if args.command == "validate":
        case = spec.cases("square")[0]
        result = benchmark(
            spec=spec,
            kernel_name=kernel,
            case=case,
            warmups=1,
            iterations=1,
            seed=args.seed,
        )
        print_result(result)
        return

    results = []
    for kernel_name in selected(kernel):
        for case in spec.cases(args.suite):
            try:
                result = benchmark(
                    spec=spec,
                    kernel_name=kernel_name,
                    case=case,
                    warmups=args.warmups,
                    iterations=args.iterations,
                    seed=args.seed,
                )
            except ValueError as exc:
                print(f"{spec.op:<5} {kernel_name:>5} {case.name:<18} skipped: {exc}")
                continue
            results.append(result)
            print_result(result)

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("outputs/raw") / f"{spec.op}_{kernel}_{args.suite}_{stamp}.csv"
    else:
        path = Path(args.output)
    write_csv(results, path)
    print(f"wrote {path}")
