from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .fd_reference import elliptic_fd_metrics, parabolic_fd_metrics
from .problems import ELLIPTIC_CONSTANT_CASES, PARABOLIC_CASES, get_elliptic_problem, get_parabolic_problem


def make_fd_tables(output_dir: str | Path, grids: Iterable[int] = (32, 64, 128), times: Iterable[float] = (0.5, 1.0, 2.5, 5.0), tau: float = 0.1) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in ELLIPTIC_CONSTANT_CASES:
        for n in grids:
            problem = get_elliptic_problem(case, n)
            metrics = elliptic_fd_metrics(problem)
            rows.append({"problem": case, "N": n, **metrics})
    pd.DataFrame(rows).to_csv(output_dir / "table3_fd_elliptic.csv", index=False)

    rows = []
    for case in PARABOLIC_CASES:
        for n in grids:
            problem = get_parabolic_problem(case, n)
            for t in times:
                metrics = parabolic_fd_metrics(problem, final_time=float(t), tau=tau)
                rows.append({"problem": case, "N": n, "t": float(t), **metrics})
    pd.DataFrame(rows).to_csv(output_dir / "table7_fd_parabolic.csv", index=False)
