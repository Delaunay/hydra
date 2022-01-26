# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

from hydra.core.config_store import ConfigStore


@dataclass
class OrionClientConf:
    name: Optional[str] = None
    version: Optional[str] = None,
    algorithms: Optional[str] = None
    strategy: Optional[str] = None
    max_trials: Optional[str] = None
    max_broken: Optional[str] = None
    storage: Optional[str] = None
    branching: Optional[str] = None
    max_idle_time: Optional[str] = None
    heartbeat: Optional[str] = None
    working_dir: Optional[str] = None
    debug: Optional[str] = False
    executor: Optional[str] = None


@dataclass
class OrionSweeperConf:
    _target_: str = (
        "hydra_plugins.hydra_orion_sweeper.orion_sweeper.OrionSweeper"
    )

    optim: OrionClientConf = OrionClientConf()

    # default parametrization of the search space
    parametrization: Dict[str, Any] = field(default_factory=dict)


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="orion",
    node=OrionSweeperConf,
    provider="orion",
)
