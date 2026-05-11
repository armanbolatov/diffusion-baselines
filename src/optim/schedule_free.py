"""Schedule-Free AdamW wrapper.

The schedulefree library's optimizer has special semantics:
- Call `opt.train()` before training steps (default state after construction).
- Call `opt.eval()` before evaluation / loss computation in eval mode.
- Call `opt.train()` again afterwards.

Our Trainer detects these methods via hasattr and toggles them automatically.
"""
from schedulefree import AdamWScheduleFree as _AdamWScheduleFree


class ScheduleFreeAdamW(_AdamWScheduleFree):
    def __init__(self, params, lr=2.5e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, warmup_steps=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, warmup_steps=warmup_steps)
        self.last_metrics: dict = {}
