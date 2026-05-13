"""Task-agnostic training loop.

The trainer owns: iter counter, LR schedule, eval cadence, checkpointing,
logging. Everything diffusion-task-specific lives in the task object.
"""
import math
import time
from pathlib import Path
from typing import Optional

import torch

from .logger.jsonl import JsonlLogger
from .optim.schedule import build_schedule
from .utils.checkpoint import load_ckpt, save_ckpt


class Trainer:
    def __init__(self, args, task, model, optimizer, opt_summary,
                 train_iter, eval_iter, out_dir):
        self.args = args
        self.task = task
        self.model = model
        self.optimizer = optimizer
        self.opt_summary = opt_summary
        self.train_iter = train_iter
        self.eval_iter = eval_iter
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # logging
        self.jsonl = JsonlLogger(self.out_dir)
        self.wandb_logger = None
        if args.wandb:
            from .logger.wandb_logger import WandbLogger
            self.wandb_logger = WandbLogger(args, exp_name=self.out_dir.name)

        # schedule
        self._lr_fn = build_schedule(args.schedule)
        # Schedule-Free has its own LR control: we still expose a fixed lr (no decay) to schedulefree.
        self._is_schedulefree = hasattr(optimizer, "train") and hasattr(optimizer, "eval")

        # state
        self.iter_num = 0
        self.ckpt_path = self.out_dir / "ckpt.pt"
        # Auto-resume only when --resume is set. Otherwise sweep agents
        # could silently load a sibling-config's checkpoint and skip
        # training entirely (the unique-hash exp_name should prevent this
        # too, but belt-and-suspenders).
        if getattr(args, "resume", False):
            self.iter_num = load_ckpt(self.ckpt_path, model, optimizer, device=args.device)

    def _log(self, event: str, **fields):
        self.jsonl.log(event, **fields)
        if self.wandb_logger is not None:
            self.wandb_logger.log(event, **fields)

    def _set_lr(self) -> float:
        if self._is_schedulefree:
            # schedulefree manages its own LR; just return the constant value.
            return self.args.lr
        lr = self._lr_fn(self.iter_num, self.args.lr, self.args.warmup_iters,
                        self.args.max_iters, self.args.min_lr)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def run(self):
        args = self.args
        model = self.model
        optimizer = self.optimizer

        # config + params header
        self._log("config", args=vars(args))
        self._log("params", **self.opt_summary)

        # ensure schedule-free is in 'train' mode
        if self._is_schedulefree:
            optimizer.train()

        t_start = time.time()
        t_prev = t_start
        # reset cuda memory stats so peak_memory_mb is for the current run only
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        nonfin = 0
        diverged = False

        while self.iter_num < args.max_iters:
            lr = self._set_lr()

            # one optimizer step
            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = self.task.build_data(args)[0]
                batch = next(self.train_iter)

            optimizer.zero_grad(set_to_none=True)
            loss, log_dict = self.task.loss_step(model, batch, self.iter_num)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            l = float(loss.detach())
            if not (l == l and abs(l) < 1e9):
                nonfin += 1
                if nonfin >= 50:
                    diverged = True
                    break
            else:
                nonfin = 0

            if self.iter_num % args.log_interval == 0:
                opt_metrics = getattr(optimizer, "last_metrics", {}) or {}
                now = time.time()
                dt_iter = (now - t_prev) / max(1, args.log_interval)
                t_prev = now
                peak_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                           if torch.cuda.is_available() else 0.0)
                self._log("iter", iter=self.iter_num, loss=l, lr=lr,
                          dt_iter_s=dt_iter, peak_mem_mb=peak_mb,
                          opt_metrics=opt_metrics, **log_dict)

            self.iter_num += 1

            # eval at interval (skip iter 0)
            if self.iter_num % args.eval_interval == 0:
                self._eval_and_checkpoint(t_start)

        # final eval
        self._eval_and_checkpoint(t_start, final=True)
        if diverged:
            self._log("diverged", iter=self.iter_num)

        self.jsonl.close()
        if self.wandb_logger is not None:
            self.wandb_logger.close()

    def _eval_and_checkpoint(self, t_start, final: bool = False):
        model = self.model
        opt = self.optimizer
        if self._is_schedulefree:
            opt.eval()
        model.eval()
        with torch.no_grad():
            metrics = self.task.eval_step(model, self.eval_iter, self.args.eval_batches)
        model.train()
        if self._is_schedulefree:
            opt.train()

        event = "final_eval" if final else "eval"
        self._log(event, iter=self.iter_num,
                  elapsed_s=time.time() - t_start, **metrics)
        save_ckpt(self.ckpt_path, model, opt, self.iter_num)
