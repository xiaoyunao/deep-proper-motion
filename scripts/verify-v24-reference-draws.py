"""Check four parallel draws against signed pre-gauge checkpoints of a real solved graph."""
import argparse
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np

from deep_pm.full_reference_solver import parallel_draws, SchurProblem
from deep_pm.io import digest, save_json
from deep_pm.zero_point import read_checkpoint, write_checkpoint


def run(directory, output, workers):
    binding = directory / "private_numeric_binding.json"
    manifest = json.loads(binding.read_text())
    solved = json.loads((directory / "v22_reference_component_summary.json").read_text())
    if digest(binding) != solved["numeric_binding_sha256"]:
        raise ValueError("Changed archived numeric export")
    for name, sha in manifest["files"].items():
        if digest(directory / name) != sha:
            raise ValueError("Changed archived graph input")
    arrays = [np.load(directory / f"{key}.npy", mmap_mode="r")
              for key in ("ptr", "groups", "precision", "values", "names")]
    ptr, groups, precision, values, names = arrays
    checkpoints = directory / "checkpoints"
    robust = read_checkpoint(checkpoints / "robust.npz")
    archived = [read_checkpoint(checkpoints / f"draw-{i:03d}.npz")["point"] for i in range(5)]
    problem = SchurProblem(ptr, groups, precision, values, robust["weight"], len(names))
    seed = 45116 + solved["component"]
    rng = np.random.default_rng(seed)
    # The untimed first solve prepares the same Numba kernels for both schedules.
    def serial():
        return problem.solve(rng.exponential(size=len(ptr)-1), x0=robust["point"])[0]
    np.testing.assert_array_equal(serial(), archived[0])
    start = monotonic()
    serial_results = [serial() for _ in range(4)]
    serial_seconds = monotonic() - start
    trial = output.parent / ("private_draw_check_" + os.environ["SLURM_JOB_ID"])
    trial.mkdir()
    write_checkpoint(trial / "draw-000.npz", point=archived[0])
    start = monotonic()
    values, _ = parallel_draws(problem, robust["point"], 5, seed, trial, workers)
    parallel_seconds = monotonic() - start
    for i, value in enumerate(serial_results, 1):
        np.testing.assert_array_equal(value, archived[i])
        np.testing.assert_array_equal(values[i], archived[i])
        if not value.tobytes() == values[i].tobytes() == archived[i].tobytes():
            raise ValueError("Bootstrap draw bytes differ from the archived estimator")
    save_json(output, dict(archived_numeric_sha256=digest(binding), component=solved["component"],
              measurements=len(groups), references=len(ptr)-1, groups=len(names), compared_draws=4,
              workers=workers, byte_equal_values=True, serial_seconds=serial_seconds,
              parallel_seconds=parallel_seconds, slurm_job=os.environ["SLURM_JOB_ID"],
              scope="Draws 1–4 before gauge; identical final robust weights, reference rows and RNG states. "
                    "Parallel time includes worker creation, response checksums and checkpoint writes; "
                    "serial timing excludes checkpoint I/O. Component export and IRLS are outside both times."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1 < args.workers <= int(os.getenv("SLURM_CPUS_PER_TASK", 1)):
        parser.error("Use an allocated Slurm CPU node")
    run(args.directory, args.output, args.workers)
