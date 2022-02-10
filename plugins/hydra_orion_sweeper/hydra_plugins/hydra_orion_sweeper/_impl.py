# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy
from dataclasses import asdict
import logging
import math
from typing import (
    Any,
    Dict,
    List,
    MutableMapping,
    MutableSequence,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)

from hydra.core import utils
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import (
    ChoiceSweep,
    IntervalSweep,
    Override,
    Transformer,
)
from hydra.core.plugins import Plugins
from hydra.plugins.launcher import Launcher
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, ListConfig, OmegaConf

from orion.core.utils.flatten import flatten
from orion.client import create_experiment
from orion.client.experiment import ExperimentClient
from orion.core.worker.trial import Trial
from orion.algo.space import Space, Dimension
from orion.core.io.space_builder import DimensionBuilder, SpaceBuilder
from orion.core.utils.exceptions import (
    CompletedExperiment,
    ReservationRaceCondition,
    WaitingForTrials,
)

from .config import OrionClientConf, WorkerConf, AlgorithmConf, StorageConf

log = logging.getLogger(__name__)


def as_overrides(trial, additional):
    """Returns the trial arguments as hydra overrides"""
    kwargs = deepcopy(additional)
    kwargs.update(flatten(trial.params))
    return tuple(f"{k}={v}" for k, v in kwargs.items())


class SpaceParser:
    """Generate an Orion space from parameters and overrides"""

    def __init__(self) -> None:
        self.space = dict()
        self.overrides = dict()
        self.arguments = dict()

    def space(self) -> Space:
        configuration = deepcopy(self.space)
        configuration.update(self.overrides)
        return SpaceBuilder().build(configuration), self.arguments

    def add_from_parametrization(self, parametrization: Optional[DictConfig]) -> None:
        for k, v in parametrization.items():
            dim = self.parse_parametrization(k, v)
            self.space[dim.name] = dim.get_prior_string()

    def add_from_overrides(self, arguments: List[str]) -> None:
        parser = OverridesParser.create()
        parsed = parser.parse_overrides(arguments)

        for override in parsed:
            dim = self.process_overrides(override)
            self.overrides[dim.name] = dim.get_prior_string()

        self.parse_orion_overrides(arguments)

    def parse_orion_overrides(self, arguments: List[str]) -> None:
        for arg in arguments:
            # name='loguniform(0, 1)'
            name_prior = arg.replace("'", "").replace('"', "").split("=")

            if len(name_prior) != 2:
                continue

            if name_prior in self.overrides:
                continue

            name, prior = name_prior
            self.overrides[name] = prior

    def process_overrides(self, override: Override) -> Optional[Dimension]:
        values = override.value()
        name = override.key_or_group

        if not override.is_sweep_override():
            self.arguments[name] = values
            return None

        elif override.is_choice_sweep():
            return DimensionBuilder(name).choices(*values.list)

        elif override.is_range_sweep():
            choices = [v for v in range(values.start, values.stop, values.step)]
            return DimensionBuilder(name).choices(*choices)

        elif override.is_interval_sweep():
            discrete = type(values.start) is int

            return DimensionBuilder(name).uniform(
                values.start, values.end, discrete=discrete
            )

    def parse_dimension(name, dim) -> Dimension:
        return DimensionBuilder(name).build(dim)


class OrionSweeperImpl(Sweeper):
    def __init__(
        self,
        orion: OrionClientConf,
        worker: WorkerConf,
        algorithm: AlgorithmConf,
        storage: StorageConf,
        parametrization: Optional[DictConfig],
    ):
        self.orion_config = orion
        self.worker_config = worker
        self.algo_config = algorithm
        self.storage_config = storage

        self.launcher: Optional[Launcher] = None
        self.hydra_context: Optional[HydraContext] = None
        self.job_results = None

        self.job_idx: Optional[int] = None

        self.space_parser = SpaceParser()
        self.space_parser.add_from_parametrization(parametrization)

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.job_idx = 0
        self.config = config
        self.hydra_context = hydra_context

        # Maybe we could have an option to replace the launcher
        # by our own here, we can save the task function
        # and use it with workon
        # which would make our research more efficient
        self.launcher = Plugins.instance().instantiate_launcher(
            hydra_context=hydra_context, task_function=task_function, config=config
        )

    def suggest_trials(self, count) -> List[Trial]:
        """Suggest a bunch of trials to be dispatched to the workers"""
        trials = []

        for _ in range(count):
            try:
                trial = self.client.suggest(pool_size=count)
                trials.append(trial)

            # non critical errors
            except WaitingForTrials:
                break

            except ReservationRaceCondition:
                break

            except CompletedExperiment:
                break

        return trials

    def new_experiment(self, arguments) -> ExperimentClient:
        """Initialize orion client from the config and the arguments"""

        self.space_parser.add_from_overrides(arguments)
        space, arguments = self.space_parser.space()

        print()
        print(space)
        print(arguments)
        print(self.client_config)
        print()

        return create_experiment(
            name=self.orion_config.name,
            version=self.orion_config.version,
            space=space,
            algorithms={self.algo_config.name: self.algo_config.config},
            strategy=None,
            max_trials=self.worker_config.max_trials,
            max_broken=self.worker_config.max_broken,
            storage=self.storage_config,
            branching=self.orion_config.branching,
            max_idle_time=None,
            heartbeat=None,
            working_dir=None,
            debug=self.orion_config.debug,
            executor=None,
        )

    def sweep(self, arguments: List[str]) -> None:
        """Execute the optimization process"""

        assert self.config is not None
        assert self.launcher is not None
        assert self.job_idx is not None

        self.client = self.new_experiment(arguments)
        failures = []

        while not self.client.is_done:
            trials = self.suggest_trials(self.worker_config.n_workers)

            overrides = list(as_overrides(t, dict()) for t in trials)

            self.validate_batch_is_legal(overrides)
            returns = self.launcher.launch(overrides, initial_job_idx=self.job_idx)

            self.job_idx += len(returns)

            for trial, result in zip(trials, returns):
                if result.status == utils.JobStatus.COMPLETED:
                    value = result.return_value

                    objective = dict(name="objective", type="objective", value=value)

                    self.client.observe(trial, [objective])

                elif result.status == utils.JobStatus.FAILED:
                    # We probably got an exception
                    self.client.release(trial, status="broken")
                    failures.append(result)

                elif result.status == utils.JobStatus.UNKNOWN:
                    self.client.release(trial, status="interrupted")

            if len(failures) > self.worker_config.max_broken:
                # make the `Future` raise the exception it received
                failures[-1].return_value

        self.show_results()

    def show_results(self) -> None:
        """Retrieve the optimization stats and show which config was the best"""
        results = self.client.stats

        best_params = self.client.get_trial(uid=results.best_trials_id).params

        results = asdict(results)
        results["best_params"] = best_params
        results["start_time"] = str(results["start_time"])
        results["finish_time"] = str(results["finish_time"])
        results["duration"] = str(results["duration"])

        OmegaConf.save(
            OmegaConf.create(results),
            f"{self.config.hydra.sweep.dir}/optimization_results.yaml",
        )

        log.info(
            "Best parameters: %s", " ".join(f"{x}={y}" for x, y in best_params.items())
        )