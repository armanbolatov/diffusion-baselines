"""Generate the Bayesian sweep YAMLs (one per optimizer per task)."""
from pathlib import Path

import yaml

OUT = Path("configs/sweeps")
OUT.mkdir(parents=True, exist_ok=True)

WD_GRID         = [0.01, 0.1, 0.3]
GRAD_CLIP_GRID  = [0, 1.0]
MOMENTUM_GRID   = [0.9, 0.95, 0.99]
BETA1_GRID      = [0.9, 0.95]
BETA2_GRID      = [0.99, 0.999]
GAMMA_GRID      = [0.0, 0.025, 0.1]
PRECOND_GRID    = [10, 50]

OPTIMIZERS = {
    "adamw": {
        "lr":     [1e-4, 3e-4, 1e-3, 3e-3],
        "extras": {"beta1": BETA1_GRID, "beta2": BETA2_GRID},
    },
    "lion": {
        "lr":     [1e-5, 3e-5, 1e-4, 3e-4],
        "extras": {"beta1": BETA1_GRID, "beta2": BETA2_GRID},
    },
    "muon": {
        "lr":     [1e-5, 3e-5, 1e-4, 3e-4],
        "extras": {"momentum": MOMENTUM_GRID},
        "fixed":  {"ns_steps": 5},
    },
    "schedule_free": {
        "lr":              [1e-4, 3e-4, 1e-3, 3e-3],
        "extras":          {"beta1": BETA1_GRID, "beta2": BETA2_GRID},
        "schedule_override": "constant",
        "warmup_override":   0,
    },
    "soap": {
        "lr":     [3e-4, 1e-3, 3e-3],
        "extras": {"beta1": BETA1_GRID, "beta2": BETA2_GRID,
                   "precondition_frequency": PRECOND_GRID},
        "fixed":  {"shampoo_beta": 0.95},
    },
    "mars": {
        "lr":     [1e-4, 3e-4, 1e-3, 3e-3],
        "extras": {"beta1": BETA1_GRID, "beta2": BETA2_GRID, "gamma": GAMMA_GRID},
    },
    "mars_m": {
        "lr":     [1e-4, 3e-4, 1e-3, 3e-3],
        "extras": {"momentum": MOMENTUM_GRID, "gamma": GAMMA_GRID},
        "fixed":  {"ns_steps": 5},
    },
}

TASKS = {
    "text8": {
        "task":      "text_text8",
        "diffusion": "mdlm",
        "model_keys": {
            "n_layer": 8, "n_head": 8, "d_model": 256, "block_size": 256,
            "mdlm_eps": 1e-3, "dropout": 0.0,
        },
    },
    "celeba": {
        "task":      "image_celeba",
        "diffusion": "ddpm",
        "model_keys": {
            "num_diffusion_steps": 1000, "beta_schedule": "linear",
            "dropout": 0.0,
        },
    },
}

COMMON = {
    "warmup_iters":  1000,
    "schedule":      "wsd",
    "batch_size":    256,
    "eps":           1e-8,
    "seed":          0,
}
TEXT_MAX_ITERS  = 10000
IMAGE_MAX_ITERS = 500
RUN_CAP = 100

TEXT_KILL_AT_ITER  = 3000
IMAGE_KILL_AT_ITER = 200
TEXT_KILL_IF_ABOVE  = 3.5
IMAGE_KILL_IF_ABOVE = 0.5


def build_yaml(task_name: str, opt_name: str) -> dict:
    t = TASKS[task_name]
    o = OPTIMIZERS[opt_name]
    is_image = t["task"].startswith("image_")

    params = {}

    locked = {**COMMON,
              "task":      t["task"],
              "diffusion": t["diffusion"],
              "optimizer": opt_name,
              "max_iters":     IMAGE_MAX_ITERS if is_image else TEXT_MAX_ITERS,
              "warmup_iters":  100             if is_image else COMMON["warmup_iters"],
              "eval_interval": 100             if is_image else 500}
    for k, v in locked.items():
        if opt_name == "schedule_free" and k == "schedule":
            params[k] = {"value": o.get("schedule_override", "constant")}
        elif opt_name == "schedule_free" and k == "warmup_iters":
            params[k] = {"value": o.get("warmup_override", 0)}
        else:
            params[k] = {"value": v}

    for k, v in t["model_keys"].items():
        params[k] = {"value": v}

    params["exp_root"]      = {"value": f"exps_tuning/{task_name}/{opt_name}"}
    params["kill_at_iter"]  = {"value": IMAGE_KILL_AT_ITER if is_image else TEXT_KILL_AT_ITER}
    params["kill_if_above"] = {
        "value": IMAGE_KILL_IF_ABOVE if is_image else TEXT_KILL_IF_ABOVE
    }

    for k, v in o.get("fixed", {}).items():
        params[k] = {"value": v}

    params["lr"]           = {"values": o["lr"]}
    params["weight_decay"] = {"values": WD_GRID}
    for k, v in o["extras"].items():
        params[k] = {"values": v}
    params["grad_clip"]    = {"values": GRAD_CLIP_GRID}

    return {
        "program": "src/wandb_entrypoint.py",
        "method":  "bayes",
        "run_cap": RUN_CAP,
        "metric":  {"name": "eval_bpc" if t["task"] == "text_text8" else "eval_loss",
                    "goal": "minimize"},
        "parameters": params,
    }


def main() -> None:
    n = 0
    for task in TASKS:
        for opt in OPTIMIZERS:
            doc = build_yaml(task, opt)
            path = OUT / f"{task}_{opt}.yaml"
            with path.open("w") as f:
                yaml.dump(doc, f, sort_keys=False, default_flow_style=False)
            n += 1
    print(f"wrote {n} YAMLs to {OUT}/")


if __name__ == "__main__":
    main()
