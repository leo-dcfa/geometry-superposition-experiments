"""Training loop for the toy superposition models.

One config -> one run -> one JSON-serializable result dict. The R2 monitor is
wired in unconditionally: there is no way to train a curved arm here without
producing its saturation record, because an unmonitored curved run cannot
enter a hypothesis test anyway.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from csc.interp.capacity import (
    dead_unit_fraction,
    probe_metrics,
    prototype_geometry,
    response_scale_report,
)
from csc.models.toy import ToySuperposition, parameter_count
from csc.spaces import make_space
from csc.training.data import importance_spectrum, sample_batch, weighted_mse
from csc.training.monitor import SaturationMonitor
from csc.training.seeding import data_generator, seed_everything


@dataclass
class ToyConfig:
    # geometry
    arm: str = "euclidean"  # euclidean | curved | clamped | normalized | product
    kappa: float = 0.0
    dim: int = 2
    space_kwargs: dict = field(default_factory=dict)
    # task
    n_features: int = 16
    sparsity: float = 0.9  # P(feature off)
    importance_decay: float = 0.9
    # model
    # norm_affine, not rbf: the 00c parity fixture measured rbf collapsing to
    # the all-zero solution at weight_decay=0 and recovering perfectly at
    # 0.01, i.e. the optimizer rather than the geometry selects its basin.
    # norm_affine and softmax were stable across both settings and both
    # curvature signs. See CSC_RESULTS/phase00/00c_dead_unit_parity.json.
    head: str = "norm_affine"
    encoder_init_scale: float = 1.0  # gain on the encoder weight init (00a knob)
    proto_init_scale: float = 0.2
    # optimization
    steps: int = 4_000
    batch_size: int = 512
    lr: float = 1e-2
    weight_decay: float = 0.0
    eval_every: int = 200
    seed: int = 0
    device: str = "cpu"
    # evaluation
    probe_value: float = 0.8
    probe_tol: float = 0.2

    def to_json(self) -> dict:
        return asdict(self)


def build_model(cfg: ToyConfig) -> ToySuperposition:
    space = make_space(cfg.arm, cfg.dim, cfg.kappa, **cfg.space_kwargs)
    model = ToySuperposition(
        space,
        cfg.n_features,
        head=cfg.head,
        proto_init_scale=cfg.proto_init_scale,
    )
    if cfg.encoder_init_scale != 1.0:
        with torch.no_grad():
            model.encoder.weight.mul_(cfg.encoder_init_scale)
            model.encoder.bias.mul_(cfg.encoder_init_scale)
    return model.to(cfg.device)


@dataclass
class ToyRun:
    """A finished run: a JSON-safe summary plus the live model for figures.

    Kept as two fields rather than one dict with a smuggled ``_model`` key, so
    that ``json.dump(run.summary)`` cannot fail at write time — which, for a
    study whose rule R5 says a claim without a committed artifact does not
    count, is a failure worth making structurally impossible.
    """

    summary: dict
    model: ToySuperposition


def train_toy(cfg: ToyConfig, progress: bool = False) -> ToyRun:
    seed_everything(cfg.seed)
    model = build_model(cfg)
    gen = data_generator(cfg.seed, device=cfg.device)
    importance = importance_spectrum(cfg.n_features, cfg.importance_decay, device=cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    monitor = SaturationMonitor(eval_every=cfg.eval_every)

    losses: list[dict] = []
    for step in range(cfg.steps + 1):
        batch = sample_batch(cfg.batch_size, cfg.n_features, cfg.sparsity, gen, cfg.device)
        loss = weighted_mse(model(batch), batch, importance)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        if step % cfg.eval_every == 0:
            monitor.record(step, model.geometry_report(batch))
            losses.append({"step": step, "loss": float(loss.detach())})
            if progress:
                print(f"  step {step:>6} loss {float(loss.detach()):.5f}", flush=True)

    eval_batch = sample_batch(4096, cfg.n_features, cfg.sparsity, gen, cfg.device)
    with torch.no_grad():
        final_loss = float(weighted_mse(model(eval_batch), eval_batch, importance))

    # probes are evaluated inside the eval batch, not alone: the primary head
    # normalizes by batch mean distance, so an isolated probe batch measures a
    # different readout than the one that was trained (see probe_metrics).
    probes = probe_metrics(model, value=cfg.probe_value, tol=cfg.probe_tol, context=eval_batch)
    recovered_mask = probes.pop("recovered_mask")

    summary = {
        "config": cfg.to_json(),
        "n_parameters": parameter_count(model),
        "final_loss": final_loss,
        "loss_curve": losses,
        "probes": probes,
        "recovered_mask": recovered_mask.tolist(),
        "prototype_geometry": prototype_geometry(model),
        "dead_unit_fraction": dead_unit_fraction(model, eval_batch),
        "response_scale": response_scale_report(model, eval_batch),
        "saturation": monitor.summary(),
        "saturation_records": monitor.records,
    }
    return ToyRun(summary=summary, model=model)
