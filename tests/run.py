#!/usr/bin/env python3
"""Run upstream cases using CMake-built adapters (never the runner's .cu cache)."""

import argparse
import ctypes
import importlib.util
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = [
    ("1_vector_add", "vector_add"),
    ("21_relu", "relu"),
    ("4_reduction", "reduction"),
    ("5_softmax", "softmax"),
    ("113_layer_normalization", "layer_normalization"),
    ("2_matrix_multiplication", "matrix_multiplication"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT.parent / "build")
    parser.add_argument("--mode", choices=["example", "functional", "performance", "all"], default="all")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup must be nonnegative and repeat must be positive")

    runner_path = ROOT / "runner/scripts/local_test.py"
    if not runner_path.is_file():
        parser.error("initialize test dependencies: git submodule update --init --recursive")
    spec = importlib.util.spec_from_file_location("upstream_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    torch = runner.torch
    if not torch.cuda.is_available():
        parser.error("a CUDA-enabled PyTorch installation and GPU are required")
    sys.path.insert(0, str(runner.challenge_root()))

    for challenge_name, adapter in CASES:
        challenge_dir = runner.resolve_challenge(challenge_name)
        challenge = runner.load_module(challenge_dir / "challenge.py", challenge_name).Challenge("cuda")
        signature = challenge.get_solve_signature()
        outputs = [name for name, (_, direction) in signature.items() if direction in {"out", "inout"}]
        library_path = args.build_dir.resolve() / "integration" / f"test_{adapter}.so"
        if not library_path.is_file():
            parser.error(f"missing {library_path}; run tests/run.sh to build adapters")
        library = ctypes.CDLL(str(library_path))
        function = library.solve
        function.argtypes = [ctype for ctype, _ in signature.values()]
        function.restype = None

        def solve(**kwargs):
            arguments = []
            for name, (ctype, _) in signature.items():
                value = kwargs[name]
                arguments.append(
                    ctypes.cast(ctypes.c_void_p(value.data_ptr()), ctype)
                    if isinstance(value, torch.Tensor) else ctype(value)
                )
            function(*arguments)

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        print(challenge_name, flush=True)
        if args.mode in {"example", "all"}:
            runner.run_case(challenge, solve, challenge.generate_example_test(), outputs, "cuda")
            print("  PASS example")
        if args.mode in {"functional", "all"}:
            cases = challenge.generate_functional_test()
            for case in cases:
                runner.run_case(challenge, solve, case, outputs, "cuda")
            print(f"  PASS {len(cases)} functional cases")
        if args.mode in {"performance", "all"}:
            case = challenge.generate_performance_test()
            runner.run_case(challenge, solve, case, outputs, "cuda")
            times = runner.benchmark(solve, case, outputs, "cuda", args.warmup, args.repeat)
            print(f"  PASS performance correctness | median {statistics.median(times):.3f} ms")


if __name__ == "__main__":
    main()
