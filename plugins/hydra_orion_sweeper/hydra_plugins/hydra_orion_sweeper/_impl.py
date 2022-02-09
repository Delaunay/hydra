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

from .config import OrionClientConf, WorkerConf

log = logging.getLogger(__name__)


from orion.core.utils.flatten import flatten
from orion.client import create_experiment
from orion.client.experiment import ExperimentClient
from orior.core.worker.trial import Trial
from orion.algo.space import Space
from orion.core.io.space_builder import DimensionBuilder, SpaceBuilder
from orion.core.utils.exceptions import (
    CompletedExperiment,
    ReservationRaceCondition,
    WaitingForTrials,
)


def make_dimension(name, method, **kwargs):
    lower = kwargs.get('lower', 0)
    upper = kwargs.get('upper', 1)
    discrete = kwargs.get('discrete', False)
    default_value = kwargs.get('default_value', None)
    precision = kwargs.get('precision', 4)
    shape = kwargs.get('shape', 1)
    base = kwargs.get('base', 1)
    choices = kwargs.get('choices', 1)

    builder = DimensionBuilder()
    builder.name = name

    if method == 'choices' and isinstance(choices, MutableMapping):
        return builder.choices(**choices)

    if method == 'choices' and isinstance(choices, MutableSequence):
        return builder.choices(*choices)

    if method == 'fidelity':
        return builder.fidelity(lower, upper, base=base)

    method = getattr(builder, method)
    return method(
        lower,
        upper,
        discrete=discrete,
        default_value=default_value,
        precision=precision,
        shape=shape
    )


def create_orion_override(name, override: Override) -> Any:
    val = override.value()
    if not override.is_sweep_override():
        return val

    if override.is_choice_sweep():
        assert isinstance(val, ChoiceSweep)

        vals = [x for x in override.sweep_iterator(transformer=Transformer.encode)]

        return make_dimension(name, 'choices', *vals)


def space_from_arguments(arguments: List[str]):
    arguments.sort()
    remains = []
    configure = OrderedDict()

    for arg in arguments:
        # name='loguniform(0, 1)'
        name_prior = (arg
            .replace("'", "")
            .replace("\"", "")
            .split("=")
        )

        if len(name_prior) != 2:
            remains.append(arg)
            continue

        name, prior = name_prior
        configure[name] = prior

    print(configure)
    builder = SpaceBuilder()
    return builder.build(configure), remains


def space_from_nevergrad_overrides(arguments: List[str]):
    """Generate an Orion space from a list of arguments"""
    space = Space()
    parser = OverridesParser.create()
    parsed = parser.parse_overrides(arguments)
    arguments = dict()

    for override in parsed:
        builder = DimensionBuilder()
        builder.name = override.key_or_group
        values = override.value()

        if not override.is_sweep_override():
            arguments[builder.name] = values

        elif override.is_choice_sweep():
            dim = builder.choices(values.list)
            space.register(dim)

        elif override.is_range_sweep():
            choices = [v for v in range(values.start, values.stop, values.step)]
            dim = builder.choices(*choices)
            space.register(dim)

        elif override.is_interval_sweep():
            discrete = type(values.start) is int

            if 'log' in values.tags:
                method = builder.loguniform
            else:
                method = builder.uniform

            dim = method(values.start, values.end, discrete=discrete)
            space.register(dim)

    return space, arguments


def as_overrides(trial, additional):
    """Returns the trial arguments as hydra overrides"""
    kwargs = deepcopy(additional)
    kwargs.update(flatten(trial.params))
    return  tuple(f"{k}={v}" for k, v in kwargs.items())


class OrionSweeperImpl(Sweeper):
    def __init__(
        self,
        client: OrionClientConf,
        worker: WorkerConf,
        parametrization: Optional[DictConfig]
    ):
        self.client_config = client
        self.worker_config = worker

        self.config: Optional[DictConfig] = None
        self.launcher: Optional[Launcher] = None
        self.hydra_context: Optional[HydraContext] = None
        self.job_results = None

        self.space: Dict[str, Any] = parametrization
        self.job_idx: Optional[int] = None

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
        self.launcher = Plugins.instance().instantiate_launcher(
            hydra_context=hydra_context,
            task_function=task_function,
            config=config
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
        space, arguments = space_from_arguments(arguments)

        client_config = OmegaConf.to_container(self.client_config)

        if 'algorithms' in client_config:
            algorithms_config = client_config.pop('algorithms')
            algorithms_name = algorithms_config.pop('name', None)
            client_config['algorithms'] = {
                algorithms_name: algorithms_config
            }

        print()
        print(space)
        print(arguments)
        print(self.client_config)
        print()

        return create_experiment(
            space=space,
            **client_config
        )

    def sweep(self, arguments: List[str]) -> None:
        """Execute the optimization process"""

        assert self.config is not None
        assert self.launcher is not None
        assert self.job_idx is not None

        self.client = self.new_experiment(arguments)

        while not self.client.is_done:
            trials = self.suggest_trials(self.worker_config.n_workers)

            overrides = list(
                as_overrides(t, dict())  for t in trials
            )

            returns = self.launcher.launch(overrides, initial_job_idx=self.job_idx)

            for trial, result in zip(trials, returns):
                if result.status == utils.JobStatus.COMPLETED:
                    value = result.return_value

                    objective = dict(name="objective", type="objective", value=value)

                    self.client.observe(trial, [objective])

                elif result.status == utils.JobStatus.FAILED:
                    # We probably got an exception
                    self.client.release(trial, status="broken")

                elif result.status == utils.JobStatus.UNKNOWN:
                    # Assume unkown is because something weird happened
                    self.client.release(trial, status="interrupted")

        self.show_results()

    def show_results(self) -> None:
        """Retrieve the optimization stats and show which config was the best"""
        results = self.client.stats

        best_params = self.client.get_trial(uid=results.best_trials_id).params

        results = asdict(results)
        results['best_params'] = best_params
        results['start_time'] = str(results['start_time'])
        results['finish_time'] = str(results['finish_time'])
        results['duration'] = str(results['duration'])

        OmegaConf.save(
            OmegaConf.create(results),
            f"{self.config.hydra.sweep.dir}/optimization_results.yaml",
        )

        log.info(
            "Best parameters: %s", " ".join(f"{x}={y}" for x, y in best_params.items())
        )
