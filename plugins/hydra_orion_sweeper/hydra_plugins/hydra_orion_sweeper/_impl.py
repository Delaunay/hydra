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

from .config import OrionClientConf

log = logging.getLogger(__name__)


from orion.client import create_experiment
from orion.algo.space import Space
from orion.core.io.space_builder import DimensionBuilder


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


def create_orion_space(parametrization: Optional[DictConfig]) -> Any:
    space = Space()

    for x, y in parametrization.items():
        if isinstance(y, MutableMapping) and 'name' in y:
            dim = make_dimension(x, y['name'], **y)
            try:
                space.register(dim)
            except ValueError as exc:
                log.error("Duplicate name for %s", x)

    return space

def workon_wrapper(*args, launcher=None, initial_job_idx=None, **kwargs):
    print(args, kwargs)


class OrionSweeperImpl(Sweeper):
    def __init__(
        self,
        optim: OrionClientConf,
        parametrization: Optional[DictConfig]
    ):
        self.client_config = optim

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
            hydra_context=hydra_context, task_function=task_function, config=config
        )

    def overrides(self, arguments: List[str]):
        # Override the parametrization from commandline
        params = deepcopy(self.parametrization)

        parser = OverridesParser.create()
        parsed = parser.parse_overrides(arguments)

        for override in parsed:
            name = override.get_key_element()
            value = create_orion_override(name, override)
            super(Space, params).__setitem__(name, value)

        return params

    def sweep(self, arguments: List[str]) -> None:
        assert self.config is not None
        assert self.launcher is not None
        assert self.job_idx is not None

        self.client = create_experiment(**asdict(self.client_config))

        additional_arguments = dict(
            initial_job_idx=self.job_idx,
            launcher=self.launcher.launch
        )

        self.client.workon(
            workon_wrapper,
            n_workers=None,
            pool_size=0,
            reservation_timeout=None,
            max_trials=None,
            max_trials_per_worker=None,
            max_broken=None,
            trial_arg=arguments,
            on_error=None,
            **additional_arguments,
        )

        results_to_serialize = self.client.stats
        best_params = self.client.get_trial(
            uid=results_to_serialize['best_trials_id ']
        ).params
        results_to_serialize['best_params'] = best_params

        OmegaConf.save(
            OmegaConf.create(results_to_serialize),
            f"{self.config.hydra.sweep.dir}/optimization_results.yaml",
        )

        log.info(
            "Best parameters: %s", " ".join(f"{x}={y}" for x, y in best_params.items())
        )
