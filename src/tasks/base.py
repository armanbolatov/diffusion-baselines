"""DiffusionTask abstract base class.

Each task implements:
- build_model(args)  -> nn.Module
- build_data(args)   -> (train_iter, eval_iter)  (iters yielding batches)
- loss_step(model, batch, step)  -> (loss tensor, dict-of-scalars for logging)
- eval_step(model, eval_iter)    -> dict-of-scalars
- (optional) sample(model, args, iter, out_dir)
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Tuple


class DiffusionTask(ABC):

    @abstractmethod
    def build_model(self, args) -> Any: ...

    @abstractmethod
    def build_data(self, args) -> Tuple[Iterator, Iterator]: ...

    @abstractmethod
    def loss_step(self, model, batch, step: int) -> Tuple[Any, Dict[str, float]]: ...

    @abstractmethod
    def eval_step(self, model, eval_iter, n_batches: int) -> Dict[str, float]: ...

    def sample(self, model, args, iter_num: int, out_dir) -> None:
        # default: no-op
        return None
