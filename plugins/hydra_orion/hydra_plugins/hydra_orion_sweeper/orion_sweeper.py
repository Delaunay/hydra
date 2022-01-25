# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional

from hydra import TaskFunction
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext
from omegaconf import DictConfig

from .config import ExperimentConfig, WorkerConfig, DatabaseConfig


class OrionSweeper(Sweeper):
    """Class to interface with Nevergrad"""

    def __init__(self,
        experiment: ExperimentConfig,
        worker: WorkerConfig,
        storage: DatabaseConfig,
        parametrization: Optional[DictConfig]
    ):
        from ._impl import OrionSweeperImpl

        self.sweeper = OrionSweeperImpl(experiment, worker, storage, parametrization)

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        return self.sweeper.setup(
            hydra_context=hydra_context, task_function=task_function, config=config
        )

    def sweep(self, arguments: List[str]) -> None:
        return self.sweeper.sweep(arguments)
