# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

from hydra.core.config_store import ConfigStore


@dataclass
class DatabaseConfig:
    # Database backend (pickleddb, MongoDB)
    type: str = 'pickleddb'

    # Name of the database
    name: str = 'orion'

    # host or path to database
    host: str = 'orion.pkl'

    # Default port to MongoDB database
    port: int = 27017


@dataclass
class WorkerConfig:
    # Total number of worker to spawn
    n_workers: int = 1

    # Batch size number of samples to generate in parallel
    pool_size: int = 0

    # Worker backend
    executor: str = "joblib"

    # Worker specific configuration
    executor_configuration: Dict[str, Any] = field(default_factory=dict)

    # Heartbeat (Trials dies after not responding)
    heartbeat: int = 120

    # Trials return this return code when interrupted
    interrupt_signal_code: int = 130

    # Number of broken trials before the worker stops
    max_broken: int = 10

    # Max number of trials for each workers
    max_trials: int = 1000000000

    user_script_config: str = "config"



@dataclass
class ExperimentConfig:
    # Tries to stay close to Nevergrad.OptimConf

    optimizer: str = "NGOpt"

    # Optimizer arguments
    options: Dict[str, Any] = field(default_factory=dict)

    # optimization seed, for reproducibility
    seed: Optional[int] = None

    # Number of failed trials/function evaluation before stopping
    max_broken: int = 3

    # Max number of trials/function evaluation
    max_trials: int = 1000000000


@dataclass
class Prior:
    """Representation of all the options to define
    a scalar.
    """

    # uniform,loguniform,normal,lognormal,choices,fidelity
    name: str

    # lower bound if any
    lower: Optional[float] = None

    # upper bound if any
    upper: Optional[float] = None

    discrete: bool = False

    default_value: Optional[float] = None

    precision: Optional[int] = 4

    shape: Optional[int] = None

    base: Optional[int] = 2

    choises: List[Any] = []


@dataclass
class OrionSweeperConf:
    _target_: str = (
        "hydra_plugins.hydra_orion_sweeper.orion_sweeper.OrionSweeper"
    )

    # Configuration of the optimizer
    experiment: ExperimentConfig = ExperimentConfig()

    # Configuration of the workers
    worker: WorkerConfig = WorkerConfig()

    # Configuration of the Storage
    storage: DatabaseConfig = DatabaseConfig()

    # default parametrization of the search space
    # can be specified:
    # - as a string, like commandline arguments
    # - as a list, for categorical variables
    # - as a full scalar specification
    parametrization: Dict[str, Prior] = field(default_factory=dict)


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="orion",
    node=OrionSweeperConf,
    provider="orion",
)
