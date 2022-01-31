# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys
from pathlib import Path
from typing import Any

from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.test_utils.test_utils import (
    TSweepRunner,
    chdir_plugin_root,
    run_process,
    run_python_script,
)
from omegaconf import DictConfig, OmegaConf
from pytest import mark

from hydra_plugins.hydra_orion_sweeper import _impl
from hydra_plugins.hydra_orion_sweeper.orion_sweeper import OrionSweeper

chdir_plugin_root()


def test_discovery() -> None:
    assert OrionSweeper.__name__ in [
        x.__name__ for x in Plugins.instance().discover(Sweeper)
    ]


def test_orion_space():
    config = dict(
        a0='uniform(0, 1)',
        a1='uniform(0, 1, discrete=True)',
        a2='uniform(0, 1, precision=2)',

        b0='loguniform(1, 2)',
        b1='loguniform(1, 2, discrete=True)',
        b2='loguniform(1, 2, precision=2)',

        c0='normal(0, 1)',
        c1='normal(0, 1, discrete=True)',
        c2='normal(0, 1, precision=True)',

        d0='choices(["a", "b"])',

        e0='fidelity(10, 100)',
        e1='fidelity(10, 100, base=3)',
    )

    space = _impl.create_orion_space(config)
    assert config == space.configuration

def test_orion_overrides():
    overrides = [
        # Overrides
        "choice_1=1,2",
        "choice_2=range(1, 8)",

        "uniform_1=interval(0, 1)",
        "uniform_2=int(interval(0, 1))",
        "uniform_3=tag(log, interval(0, 1))",

        # Regular argument
        "bar=4:8",
    ]

    space, name = _impl.space_from_overrides(overrides)


def test_launched_jobs(hydra_sweep_runner: TSweepRunner) -> None:
    budget = 8
    sweep = hydra_sweep_runner(
        calling_file=None,
        calling_module="hydra.test_utils.a_module",
        config_path="configs",
        config_name="compose.yaml",
        task_function=None,
        overrides=[
            "hydra/sweeper=orion",
            "hydra/launcher=basic",
            f"hydra.sweeper.orion.max_trials={budget}",  # small budget to test fast
            "+hydra.sweeper.orion.algorithms.name=random",
            "+hydra.sweeper.orion.storage.host=test.db",
            "+hydra.sweeper.orion.storage.type=pickledb",
            "hydra.sweeper.worker.n_workers=3",

            "foo=1,2",
            "bar=4:8",
        ],
    )

    with sweep:
        assert sweep.returns is None


@mark.parametrize("with_commandline", (True, False))
def test_orion_example(with_commandline: bool, tmpdir: Path) -> None:
    budget = 32 if with_commandline else 1  # make a full test only once (faster)

    cmd = [
        "example/my_app.py",
        "-m",
        "hydra.sweep.dir=" + str(tmpdir),
        "hydra.job.chdir=True",
        f"hydra.sweeper.optim.max_trials={budget}",  # small budget to test fast
        f"hydra.sweeper.optim.num_workers={min(8, budget)}",
    ]

    if with_commandline:
        cmd += [
            "db=mnist,cifar",
            "batch_size=4,8,12,16",
            "lr=tag(log, interval(0.001, 1.0))",
            "dropout=interval(0,1)",
        ]

    run_python_script(cmd)

    returns = OmegaConf.load(f"{tmpdir}/optimization_results.yaml")

    assert isinstance(returns, DictConfig)
    assert returns.name == "orion"
    assert len(returns) == 3

    best_parameters = returns.best_evaluated_params
    assert not best_parameters.dropout.is_integer()

    if budget > 1:
        assert best_parameters.batch_size == 4  # this argument should be easy to find

    # check that all job folders are created
    last_job = max(int(fp.name) for fp in Path(tmpdir).iterdir() if fp.name.isdigit())
    assert last_job == budget - 1


# @mark.parametrize("max_failure_rate", (0.5, 1.0))
# def test_failure_rate(max_failure_rate: float, tmpdir: Path) -> None:
#     cmd = [
#         sys.executable,
#         "example/my_app.py",
#         "-m",
#         f"hydra.sweep.dir={tmpdir}",
#         "hydra.sweeper.optim.budget=2",  # small budget to test fast
#         "hydra.sweeper.optim.num_workers=2",
#         f"hydra.sweeper.optim.max_failure_rate={max_failure_rate}",
#         "error=true",
#     ]
#     out, err = run_process(cmd, print_error=False, raise_exception=False)
#     assert "Returning infinity for failed experiment" in out
#     error_string = "RuntimeError: cfg.error is True"
#     if max_failure_rate < 1.0:
#         assert error_string in err
#     else:
#         assert error_string not in err
