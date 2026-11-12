import numpy as np

from mlkan.data.generate import generate_burgers, generate_elliptic
from mlkan.data.loading import validate_dataset
from mlkan.data.solvers import laplacian_dirichlet


def test_group_split_has_no_leakage():
    splits, metadata = generate_burgers(
        n=32, n_groups=12, seed=1, final_time=0.02, save_dt=0.01
    )
    assert not validate_dataset(splits, metadata)
    groups = [set(v["group_id"].tolist()) for v in splits.values()]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])


def test_elliptic_target_satisfies_saved_discrete_source():
    splits, metadata = generate_elliptic("trig_n4", n=16, n_groups=12, seed=2)
    source_channel = metadata["input_channels"]["source"]
    for values in splits.values():
        for x, y in zip(values["x"], values["y"]):
            residual = -laplacian_dirichlet(y[..., 0], metadata["h"])[1:-1, 1:-1] - x[1:-1, 1:-1, source_channel]
            assert np.max(np.abs(residual)) < 1.0e-4
