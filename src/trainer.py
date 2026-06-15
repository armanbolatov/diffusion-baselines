"""Task-agnostic training loop."""
import time
from pathlib import Path

import torch

from .optim.schedule import build_schedule
from .logger.jsonl import JsonlLogger
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

        self.jsonl = JsonlLogger(self.out_dir)
        self.wandb_logger = None
        if args.wandb:
            from .logger.wandb_logger import WandbLogger
            self.wandb_logger = WandbLogger(args, exp_name=self.out_dir.name)

        self._lr_fn = build_schedule(args.schedule)
        # Schedule-Free maintains its own running-average weights and ignores the LR schedule.
        self._is_schedulefree = hasattr(optimizer, "train") and hasattr(optimizer, "eval")

        self.ema_decay = float(getattr(args, "ema_decay", 0.0))
        self.ema_state = ({k: v.detach().clone() for k, v in model.state_dict().items()}
                          if self.ema_decay > 0 else None)

        self.iter_num = 0
        self.ckpt_path = self.out_dir / "ckpt.pt"
        # Captured once when training crosses into the WSD decay region so a
        # later run can lengthen the schedule without an LR discontinuity.
        self.stable_ckpt_path = self.out_dir / "ckpt_stable.pt"
        self._stable_saved = self.stable_ckpt_path.exists()
        if getattr(args, "resume", False):
            self.iter_num, ema = load_ckpt(self.ckpt_path, model, optimizer, device=args.device)
            if ema is not None and self.ema_state is not None:
                self.ema_state = ema

    def _log(self, event: str, **fields):
        self.jsonl.log(event, **fields)
        if self.wandb_logger is not None:
            self.wandb_logger.log(event, **fields)

    def _set_lr(self) -> float:
        if self._is_schedulefree:
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

        self._log("config", args=vars(args))
        self._log("params", **self.opt_summary)

        if self._is_schedulefree:
            optimizer.train()

        t_start = time.time()
        t_prev = t_start
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        nonfin = 0
        diverged = False

        while self.iter_num < args.max_iters:
            lr = self._set_lr()

            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = self.task.build_data(args)[0]
                batch = next(self.train_iter)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=torch.cuda.is_available()):
                loss, log_dict = self.task.loss_step(model, batch, self.iter_num)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if self.ema_state is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        self.ema_state[k].mul_(self.ema_decay).add_(v, alpha=1 - self.ema_decay)

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

            if self.iter_num % args.eval_interval == 0:
                last = self._eval_and_checkpoint(t_start)
                kill_iter = getattr(args, "kill_at_iter", 0)
                kill_thresh = getattr(args, "kill_if_above", 0.0)
                if (kill_iter > 0 and kill_thresh > 0
                        and self.iter_num >= kill_iter and last is not None
                        and last > kill_thresh):
                    self._log("early_killed", iter=self.iter_num,
                              metric=last, threshold=kill_thresh)
                    break

        if self.iter_num % args.eval_interval != 0:
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
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=torch.cuda.is_available()):
            metrics = self.task.eval_step(model, self.eval_iter, self.args.eval_batches)
        model.train()
        if self._is_schedulefree:
            opt.train()

        event = "final_eval" if final else "eval"
        self._log(event, iter=self.iter_num,
                  elapsed_s=time.time() - t_start, **metrics)
        if getattr(self.args, "save_ckpt", False):
            # Schedule-Free needs eval-mode weights snapshotted so the saved ckpt is
            # usable for sampling without reconstructing the optimizer state.
            if self._is_schedulefree:
                opt.eval()
                save_ckpt(self.ckpt_path, model, opt, self.iter_num,
                          ema_state=self.ema_state)
                opt.train()
            else:
                save_ckpt(self.ckpt_path, model, opt, self.iter_num,
                          ema_state=self.ema_state)
            if (not self._stable_saved and self.args.schedule == "wsd"
                    and self.iter_num >= int(self.args.max_iters * 0.8)):
                save_ckpt(self.stable_ckpt_path, model, opt, self.iter_num,
                          ema_state=self.ema_state)
                self._stable_saved = True
                self._log("stable_ckpt_saved", iter=self.iter_num)
        return metrics.get("eval_bpc", metrics.get("eval_loss"))
