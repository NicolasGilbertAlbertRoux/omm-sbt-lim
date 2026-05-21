#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Latent-memory calibration scan — Latent Memory Calibration Scan
Goal:
  Test whether latent memory M is calibrable as a stable mixture of:
    Observable O, latent X, Phi, K, Arbiter A
  under coherent / destroyed / shuffled / wrong / adversarial memory controls.

Interpretation:
  - Stable coefficients => latent memory is vector-calibrable.
  - Functional recovery without stable coefficients => memory is constraint-calibrable.
  - Collapse under controls => genuine memory necessity.
"""

from __future__ import annotations

import os
import json
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path


SEED = 4343
OUTDIR = Path("results/generated/latent_memory_calibration_scan")

N_IDENTITIES = 8
N_VARIANTS = 8
D_OBS = 32
D_LAT = 32
N_TRIALS = 131072

NOISES = [0.02, 0.05, 0.10]
MEMORY_CAPS = [0.25, 0.50, 0.75, 1.00]
ARBITER_SPEEDS = [0.5, 1.0, 2.0, 4.0, 8.0]

RIDGE = 1e-4


def unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(unit(a) * unit(b), axis=-1)


def stable_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass
class World:
    O: np.ndarray
    X: np.ndarray
    Phi: np.ndarray
    K: np.ndarray
    A: np.ndarray
    M_true: np.ndarray
    ids: np.ndarray


def make_world(rng: np.random.Generator, noise: float, memory_cap: float, speed: float) -> World:
    n = N_IDENTITIES * N_VARIANTS
    ids = np.repeat(np.arange(N_IDENTITIES), N_VARIANTS)

    base_O = unit(rng.normal(size=(N_IDENTITIES, D_OBS)))
    base_X = unit(rng.normal(size=(N_IDENTITIES, D_LAT)))
    base_Phi = unit(rng.normal(size=(N_IDENTITIES, D_LAT)))
    base_K = unit(rng.normal(size=(N_IDENTITIES, D_LAT)))

    O = unit(base_O[ids] + noise * rng.normal(size=(n, D_OBS)))
    X = unit(base_X[ids] + noise * rng.normal(size=(n, D_LAT)))
    Phi = unit(base_Phi[ids] + noise * rng.normal(size=(n, D_LAT)))
    K = unit(base_K[ids] + noise * rng.normal(size=(n, D_LAT)))

    # Arbiter saturates with speed: unbounded tendency, bounded by memory organization.
    speed_gain = speed / (1.0 + speed)
    A = unit(speed_gain * (0.45 * X + 0.35 * Phi + 0.20 * K) + noise * rng.normal(size=(n, D_LAT)))

    # True memory: partly compressive / cap-bounded.
    cap = memory_cap
    M_true = unit(
        cap * (0.42 * X + 0.34 * Phi + 0.24 * K)
        + (1.0 - cap) * (0.60 * A + 0.40 * X)
        + noise * rng.normal(size=(n, D_LAT))
    )

    return World(O=O, X=X, Phi=Phi, K=K, A=A, M_true=M_true, ids=ids)


def condition_memory(world: World, rng: np.random.Generator, condition: str) -> np.ndarray:
    M = world.M_true.copy()

    if condition == "coherent_memory":
        return M

    if condition == "destroyed_memory":
        return unit(rng.normal(size=M.shape))

    if condition == "shuffled_memory":
        p = rng.permutation(len(M))
        return M[p]

    if condition == "wrong_memory":
        shifted_ids = (world.ids + 1) % N_IDENTITIES
        M_wrong = np.zeros_like(M)
        for i, sid in enumerate(shifted_ids):
            pool = np.where(world.ids == sid)[0]
            M_wrong[i] = M[rng.choice(pool)]
        return unit(M_wrong)

    if condition == "adversarial_memory":
        return unit(-M + 0.05 * rng.normal(size=M.shape))

    if condition == "memory_swap":
        swapped = M.copy()
        for identity in range(0, N_IDENTITIES, 2):
            a = np.where(world.ids == identity)[0]
            b = np.where(world.ids == (identity + 1) % N_IDENTITIES)[0]
            swapped[a], swapped[b] = M[b].copy(), M[a].copy()
        return unit(swapped)

    raise ValueError(condition)


def ridge_fit_coefficients(target: np.ndarray, components: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Fit target vector as linear combination of component vectors.
    Returns mean coefficients over all samples.
    """
    names = list(components.keys())
    coeffs = []

    for i in range(len(target)):
        C = np.stack([components[name][i] for name in names], axis=1)  # d × p
        y = target[i]
        lhs = C.T @ C + RIDGE * np.eye(len(names))
        rhs = C.T @ y
        beta = np.linalg.solve(lhs, rhs)
        coeffs.append(beta)

    coeffs = np.asarray(coeffs)
    return {name: float(coeffs[:, j].mean()) for j, name in enumerate(names)} | {
        f"{name}_std": float(coeffs[:, j].std()) for j, name in enumerate(names)
    }


def reconstruct(coeffs: dict[str, float], components: dict[str, np.ndarray]) -> np.ndarray:
    y = np.zeros_like(next(iter(components.values())))
    for name, comp in components.items():
        y += coeffs[name] * comp
    return unit(y)


def evaluate_recovery(query: np.ndarray, candidates: np.ndarray, ids: np.ndarray, target_idx: np.ndarray) -> dict:
    sims = query @ candidates.T
    predicted = sims.argmax(axis=1)
    recovered = predicted == target_idx

    nearest_non_target = []
    dominance = []
    for i, t in enumerate(target_idx):
        row = sims[i].copy()
        target_sim = row[t]
        row[t] = -np.inf
        nn = np.max(row)
        nearest_non_target.append(nn)
        dominance.append(target_sim - nn)

    return {
        "target_similarity": float(np.mean(sims[np.arange(len(target_idx)), target_idx])),
        "nearest_non_target": float(np.mean(nearest_non_target)),
        "dominance": float(np.mean(dominance)),
        "recovered": float(np.mean(recovered)),
        "competitor_won": float(np.mean(~recovered)),
    }


def run() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = stable_rng(SEED)
    rows = []

    conditions = [
        "coherent_memory",
        "destroyed_memory",
        "shuffled_memory",
        "wrong_memory",
        "adversarial_memory",
        "memory_swap",
    ]

    modes = [
        "calibrated_full",
        "calibrated_no_observable",
        "calibrated_no_arbiter",
        "memory_only",
        "arbiter_only",
        "observable_only",
        "speed_only_no_memory",
        "null_memory",
        "shuffled_label_control",
    ]

    for noise in NOISES:
        for cap in MEMORY_CAPS:
            for speed in ARBITER_SPEEDS:
                world = make_world(rng, noise=noise, memory_cap=cap, speed=speed)
                n = len(world.ids)

                trial_idx = rng.integers(0, n, size=N_TRIALS // (len(NOISES) * len(MEMORY_CAPS) * len(ARBITER_SPEEDS)))
                candidate_idx = np.arange(n)

                for condition in conditions:
                    M = condition_memory(world, rng, condition)

                    components_full = {
                        "X": world.X,
                        "Phi": world.Phi,
                        "K": world.K,
                        "A": world.A,
                    }

                    coeffs_full = ridge_fit_coefficients(M, components_full)
                    M_cal = reconstruct(coeffs_full, components_full)

                    components_no_A = {
                        "X": world.X,
                        "Phi": world.Phi,
                        "K": world.K,
                    }
                    coeffs_no_A = ridge_fit_coefficients(M, components_no_A)
                    M_no_A = reconstruct(coeffs_no_A, components_no_A)

                    components_no_obs = components_full
                    M_no_obs = M_cal

                    queries = {
                        "calibrated_full": M_cal[trial_idx],
                        "calibrated_no_observable": M_no_obs[trial_idx],
                        "calibrated_no_arbiter": M_no_A[trial_idx],
                        "memory_only": M[trial_idx],
                        "arbiter_only": world.A[trial_idx],
                        "observable_only": unit(world.O[trial_idx] @ rng.normal(size=(D_OBS, D_LAT))),
                        "speed_only_no_memory": unit(speed * world.A[trial_idx]),
                        "null_memory": unit(rng.normal(size=(len(trial_idx), D_LAT))),
                        "shuffled_label_control": M_cal[rng.permutation(trial_idx)],
                    }

                    candidates = M
                    for mode in modes:
                        ev = evaluate_recovery(
                            query=unit(queries[mode]),
                            candidates=unit(candidates),
                            ids=world.ids,
                            target_idx=trial_idx,
                        )

                        row = {
                            "condition": condition,
                            "mode": mode,
                            "noise": noise,
                            "memory_cap": cap,
                            "arbiter_speed": speed,
                            "memory_true_similarity": float(np.mean(cosine(M, world.M_true))),
                            "memory_observable_similarity": float(np.mean(cosine(M, world.X))),
                            "memory_arbiter_similarity": float(np.mean(cosine(M, world.A))),
                            "count": len(trial_idx),
                        }
                        row.update(ev)

                        if mode.startswith("calibrated"):
                            for k, v in coeffs_full.items():
                                row[f"coef_{k}"] = v

                        rows.append(row)

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["condition", "mode"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["condition", "recovered", "dominance"], ascending=[True, False, False])
    )

    summary_mode = (
        df.groupby("mode", as_index=False)
        .mean(numeric_only=True)
        .sort_values(["recovered", "dominance"], ascending=False)
    )

    coherent = summary[
        (summary.condition == "coherent_memory") &
        (summary.mode == "calibrated_full")
    ].iloc[0]

    destroyed = summary[
        (summary.condition == "destroyed_memory") &
        (summary.mode == "calibrated_full")
    ].iloc[0]

    wrong = summary[
        (summary.condition == "wrong_memory") &
        (summary.mode == "calibrated_full")
    ].iloc[0]

    shuffled = summary[
        (summary.condition == "shuffled_memory") &
        (summary.mode == "calibrated_full")
    ].iloc[0]

    adversarial = summary[
        (summary.condition == "adversarial_memory") &
        (summary.mode == "calibrated_full")
    ].iloc[0]

    no_arb = summary[
        (summary.condition == "coherent_memory") &
        (summary.mode == "calibrated_no_arbiter")
    ].iloc[0]

    memory_only = summary[
        (summary.condition == "coherent_memory") &
        (summary.mode == "memory_only")
    ].iloc[0]

    interpretation = {
        "coherent_calibrated_recovery": float(coherent.recovered),
        "coherent_memory_only_recovery": float(memory_only.recovered),
        "coherent_no_arbiter_recovery": float(no_arb.recovered),
        "destroyed_calibrated_recovery": float(destroyed.recovered),
        "wrong_calibrated_recovery": float(wrong.recovered),
        "shuffled_calibrated_recovery": float(shuffled.recovered),
        "adversarial_calibrated_recovery": float(adversarial.recovered),
        "gain_over_destroyed": float(coherent.recovered - destroyed.recovered),
        "gain_over_wrong": float(coherent.recovered - wrong.recovered),
        "gain_over_shuffled": float(coherent.recovered - shuffled.recovered),
        "gain_over_adversarial": float(coherent.recovered - adversarial.recovered),
        "arbiter_necessity_gain": float(coherent.recovered - no_arb.recovered),
        "memory_vector_calibrable": bool(
            coherent.recovered > 0.90
            and destroyed.recovered < 0.35
            and wrong.recovered < 0.35
            and adversarial.recovered < 0.20
        ),
        "arbiter_contributes_to_calibration": bool(coherent.recovered - no_arb.recovered > 0.05),
    }

    return summary, summary_mode, interpretation


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    summary, summary_mode, interpretation = run()

    print("\n=== SUMMARY BY CONDITION × MODE ===")
    print(summary.to_string(index=False))

    print("\n=== SUMMARY BY MODE ===")
    print(summary_mode.to_string(index=False))

    print("\n=== INTERPRETATION ===")
    for k, v in interpretation.items():
        if isinstance(v, bool):
            print(f"{k:40s} {v}")
        else:
            print(f"{k:40s} {v:.4f}")

    if interpretation["memory_vector_calibrable"]:
        print("\n[RESULT] Latent memory appears vector-calibrable under strict controls.")
    elif interpretation["coherent_calibrated_recovery"] > 0.90:
        print("\n[RESULT] Latent memory is functionally calibrable, but vector calibration remains partially confounded.")
    else:
        print("\n[RESULT] Latent memory calibration remains weak, ambiguous, or non-vectorial.")

    summary.to_csv(OUTDIR / "summary_condition_mode.csv", index=False)
    summary_mode.to_csv(OUTDIR / "summary_mode.csv", index=False)
    with open(OUTDIR / "interpretation.json", "w", encoding="utf-8") as f:
        json.dump(interpretation, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] wrote results to {OUTDIR}")
    print("[DONE] Latent-memory calibration scan complete")


if __name__ == "__main__":
    main()