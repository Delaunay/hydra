# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Union

from hydra.core.config_store import ConfigStore


@dataclass
class AlgorithmsConf:
    hyperband: Optional[Dict[str, Any]] = None
    random: Optional[Dict[str, Any]] = None
    asha: Optional[Dict[str, Any]] = None
    evolution_es: Optional[Dict[str, Any]] = None
    tpe: Optional[Dict[str, Any]] = None


@dataclass
class OrionClientConf:
    name: Optional[str] = None
    version: Optional[str] = None
    algorithms: Dict[str, Any] = field(default_factory=dict)
    strategy: Optional[str] = None
    max_trials: Optional[int] = None
    max_broken: Optional[int] = None
    storage: Dict[str, Any] = field(default_factory=dict)
    branching: Optional[str] = None
    max_idle_time: Optional[str] = None
    heartbeat: Optional[int] = None
    working_dir: Optional[str] = None
    debug: Optional[str] = False
    executor: Optional[str] = None


@dataclass
class WorkerConf:
    n_workers: int = 1
    pool_size: Optional[int] = None
    reservation_timeout: int = 120
    max_trials: int = 10000000
    max_trials_per_worker: int = 1000000
    max_broken: int = 3


@dataclass
class OrionSweeperConf:
    _target_: str = (
        "hydra_plugins.hydra_orion_sweeper.orion_sweeper.OrionSweeper"
    )

    orion: OrionClientConf = OrionClientConf()

    worker: WorkerConf = WorkerConf()

    # default parametrization of the search space
    parametrization: Dict[str, Any] = field(default_factory=dict)


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="orion",
    node=OrionSweeperConf,
    provider="orion",
)
