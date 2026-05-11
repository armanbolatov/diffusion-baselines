"""Entry point. Dispatches by --task and --optimizer."""
import json
import sys

import torch

from .config.base import get_args
from .logger.jsonl import JsonlLogger  # only for --dry_run
from .optim import build_optimizer
from .tasks import get_task
from .trainer import Trainer
from .utils.exp_name import resolve_out_dir
from .utils.seed import set_seed


def main(argv=None) -> int:
    args = get_args(argv)
    set_seed(args.seed)
    out_dir = resolve_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return 0

    # device
    device = torch.device(args.device)

    # task
    task = get_task(args)

    # model
    model = task.build_model(args).to(device)
    if args.compile:
        model = torch.compile(model)

    # data
    train_iter, eval_iter = task.build_data(args)

    # optimizer
    optimizer, opt_summary = build_optimizer(args, model)

    # train
    trainer = Trainer(args, task, model, optimizer, opt_summary,
                      train_iter, eval_iter, out_dir)
    trainer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
