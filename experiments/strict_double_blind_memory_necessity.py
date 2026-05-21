#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict double-blind memory-necessity benchmark.

Double-blind observable-alias memory-necessity anti-cheat.

Purpose
-------
This benchmark explicitly forbids the observable channel from carrying
within-alias identity information:

  • all identities inside the same alias group have the exact same observable vector;
  • the candidate set is the alias group itself;
  • observable_only must fall to chance = 1 / ALIAS_SIZE;
  • speed_only_no_memory must also stay near chance;
  • coherent memory should recover the target;
  • wrong / swapped / adversarial / destroyed / shuffled memory should collapse;
  • shuffled_label_control must score near chance and is counted as false positive
    when it recovers the true target while evaluated against a shuffled target.

No template/target vector is used by the tested modes. The exact template is only
kept as an upper-bound sanity check.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------
SEED = 424242
OUTDIR = Path("results/generated/strict_double_blind_memory_necessity")

N_GROUPS = 64
ALIAS_SIZE = 8
N_IDENTITIES = N_GROUPS * ALIAS_SIZE
OBS_DIM = 48
MEM_DIM = 96

N_REPEATS = 32

ARBITER_SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
MEMORY_CAPS = [0.25, 0.50, 0.75, 1.00]
MEMORY_LEAKS = [0.0, 0.10]

CHANCE = 1.0 / ALIAS_SIZE
CHANCE_TOL = 0.035
SUCCESS_FLOOR = 0.92
COLLAPSE_CEIL = 0.25


# -----------------------------
# Utilities
# -----------------------------
def normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def stable_soft_clip(x: np.ndarray, cap: float) -> np.ndarray:
    cap = float(max(cap, 1e-6))
    return np.tanh(x / cap) * cap


def speed_transform(x: np.ndarray, speed: float) -> np.ndarray:
    y = np.tanh(float(speed) * x)
    return normalize(y)


def choose_with_random_tie(scores: np.ndarray, rng: np.random.Generator) -> int:
    m = np.max(scores)
    tied = np.flatnonzero(np.isclose(scores, m, atol=1e-12, rtol=1e-12))
    return int(rng.choice(tied))


def fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{sec:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{sec:04.1f}s"


def print_progress(done: int, total: int, start: float) -> None:
    elapsed = time.time() - start
    pct = 100.0 * done / max(total, 1)
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else float("nan")

    print(
        f"[progress] {done:,}/{total:,} "
        f"({pct:5.1f}%) | elapsed={fmt_seconds(elapsed)} "
        f"| eta={fmt_seconds(remaining) if math.isfinite(remaining) else 'n/a'}",
        flush=True,
    )


@dataclass(frozen=True)
class World:
    observable_by_group: np.ndarray
    observable_by_id: np.ndarray
    memory_by_id: np.ndarray
    group_of_id: np.ndarray
    ids_by_group: np.ndarray


def make_world(rng: np.random.Generator) -> World:
    observable_by_group = normalize(rng.normal(size=(N_GROUPS, OBS_DIM)))

    ids_by_group = np.arange(N_IDENTITIES).reshape(N_GROUPS, ALIAS_SIZE)
    group_of_id = np.repeat(np.arange(N_GROUPS), ALIAS_SIZE)

    observable_by_id = observable_by_group[group_of_id].copy()
    memory_by_id = normalize(rng.normal(size=(N_IDENTITIES, MEM_DIM)))

    return World(
        observable_by_group=observable_by_group,
        observable_by_id=observable_by_id,
        memory_by_id=memory_by_id,
        group_of_id=group_of_id,
        ids_by_group=ids_by_group,
    )


def memory_query_for_condition(
    condition: str,
    target_id: int,
    candidate_ids: np.ndarray,
    world: World,
    rng: np.random.Generator,
) -> np.ndarray:
    if condition == "coherent_memory":
        q = world.memory_by_id[target_id]

    elif condition == "destroyed_memory":
        q = normalize(rng.normal(size=MEM_DIM))

    elif condition == "wrong_memory":
        wrong_pool = candidate_ids[candidate_ids != target_id]
        wrong_id = int(rng.choice(wrong_pool))
        q = world.memory_by_id[wrong_id]

    elif condition == "adversarial_memory":
        target_mem = world.memory_by_id[target_id]
        comp = candidate_ids[candidate_ids != target_id]
        sims = world.memory_by_id[comp] @ target_mem
        adv_id = int(comp[np.argmin(sims)])
        q = -world.memory_by_id[target_id] + 0.75 * world.memory_by_id[adv_id]
        q = normalize(q)

    elif condition == "shuffled_memory":
        q = world.memory_by_id[int(rng.integers(0, N_IDENTITIES))]

    elif condition == "memory_swap":
        g = world.group_of_id[target_id]
        other_groups = np.flatnonzero(np.arange(N_GROUPS) != g)
        other_g = int(rng.choice(other_groups))
        other_id = int(rng.choice(world.ids_by_group[other_g]))
        q = world.memory_by_id[other_id]

    else:
        raise ValueError(f"Unknown memory condition: {condition}")

    return normalize(q)


def score_mode(
    mode: str,
    obs_q: np.ndarray,
    mem_q: np.ndarray,
    target_id: int,
    candidate_ids: np.ndarray,
    world: World,
    speed: float,
    cap: float,
    leak: float,
    rng: np.random.Generator,
) -> np.ndarray:
    obs_c = world.observable_by_id[candidate_ids]
    mem_c = world.memory_by_id[candidate_ids]

    obs_scores = obs_c @ obs_q
    mem_scores = mem_c @ mem_q

    bounded_q = normalize(stable_soft_clip(mem_q, cap))
    bounded_c = normalize(stable_soft_clip(mem_c, cap), axis=1)
    bounded_scores = bounded_c @ bounded_q

    arb_q = speed_transform(bounded_q, speed)
    arb_c = normalize(np.vstack([speed_transform(v, speed) for v in bounded_c]), axis=1)
    arb_scores = arb_c @ arb_q

    if mode == "exact_template_upper_bound":
        scores = np.full(len(candidate_ids), -1.0)
        scores[np.where(candidate_ids == target_id)[0][0]] = 1.0
        return scores

    if mode == "observable_only":
        return obs_scores

    if mode == "null_memory":
        return obs_scores

    if mode == "latent_memory_only":
        return mem_scores

    if mode == "arbiter_only":
        return arb_scores

    if mode == "speed_only_no_memory":
        gate = math.tanh(speed) * (1.0 + leak)
        return gate * obs_scores

    if mode == "memory_bounded_arbiter":
        return 0.20 * obs_scores + 0.45 * bounded_scores + 0.35 * arb_scores

    if mode == "overdriven_arbiter":
        over_q = speed_transform(mem_q, speed * 4.0)
        over_c = normalize(
            np.vstack([speed_transform(v, speed * 4.0) for v in mem_c]),
            axis=1,
        )
        return 0.20 * obs_scores + 0.80 * (over_c @ over_q)

    if mode == "shuffled_label_control":
        return 0.20 * obs_scores + 0.45 * bounded_scores + 0.35 * arb_scores

    raise ValueError(f"Unknown mode: {mode}")


def run(progress: bool = True, progress_every: int = 100_000) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    rng = np.random.default_rng(SEED)
    world = make_world(rng)

    conditions = [
        "coherent_memory",
        "destroyed_memory",
        "wrong_memory",
        "adversarial_memory",
        "shuffled_memory",
        "memory_swap",
    ]

    modes = [
        "exact_template_upper_bound",
        "observable_only",
        "null_memory",
        "speed_only_no_memory",
        "latent_memory_only",
        "arbiter_only",
        "overdriven_arbiter",
        "memory_bounded_arbiter",
        "shuffled_label_control",
    ]

    total_jobs = (
        N_REPEATS
        * N_GROUPS
        * ALIAS_SIZE
        * len(conditions)
        * len(ARBITER_SPEEDS)
        * len(MEMORY_CAPS)
        * len(MEMORY_LEAKS)
        * len(modes)
    )

    rows: List[dict] = []
    done = 0
    start = time.time()

    if progress:
        print(f"[start] strict double-blind benchmark | total scored jobs={total_jobs:,}", flush=True)

    for repeat in range(N_REPEATS):
        if progress:
            print(f"[repeat] {repeat + 1}/{N_REPEATS}", flush=True)

        for g in range(N_GROUPS):
            candidate_ids = world.ids_by_group[g]

            for target_id in candidate_ids:
                obs_q = world.observable_by_id[target_id]

                for condition in conditions:
                    mem_q = memory_query_for_condition(
                        condition,
                        int(target_id),
                        candidate_ids,
                        world,
                        rng,
                    )

                    shuffled_target_id = int(rng.choice(candidate_ids))

                    for speed in ARBITER_SPEEDS:
                        for cap in MEMORY_CAPS:
                            for leak in MEMORY_LEAKS:
                                for mode in modes:
                                    scores = score_mode(
                                        mode=mode,
                                        obs_q=obs_q,
                                        mem_q=mem_q,
                                        target_id=int(target_id),
                                        candidate_ids=candidate_ids,
                                        world=world,
                                        speed=speed,
                                        cap=cap,
                                        leak=leak,
                                        rng=rng,
                                    )

                                    pred_local = choose_with_random_tie(scores, rng)
                                    pred_id = int(candidate_ids[pred_local])

                                    eval_target = (
                                        shuffled_target_id
                                        if mode == "shuffled_label_control"
                                        else int(target_id)
                                    )

                                    recovered = float(pred_id == eval_target)
                                    false_positive = float(
                                        mode == "shuffled_label_control"
                                        and pred_id == int(target_id)
                                    )

                                    target_local = int(np.where(candidate_ids == eval_target)[0][0])
                                    target_score = float(scores[target_local])
                                    non_target = np.delete(scores, target_local)
                                    nearest_non_target = float(np.max(non_target))
                                    dominance = target_score - nearest_non_target

                                    true_target_local = int(np.where(candidate_ids == int(target_id))[0][0])
                                    true_target_score = float(scores[true_target_local])

                                    observable_similarity = float(
                                        (world.observable_by_id[candidate_ids] @ obs_q).mean()
                                    )
                                    memory_similarity = float(world.memory_by_id[eval_target] @ mem_q)

                                    rows.append(
                                        {
                                            "repeat": repeat,
                                            "group": g,
                                            "target_id": int(target_id),
                                            "eval_target_id": int(eval_target),
                                            "pred_id": pred_id,
                                            "condition": condition,
                                            "mode": mode,
                                            "arbiter_speed": speed,
                                            "memory_cap": cap,
                                            "memory_leak": leak,
                                            "target_similarity": target_score,
                                            "true_target_similarity": true_target_score,
                                            "nearest_non_target": nearest_non_target,
                                            "dominance": dominance,
                                            "observable_similarity": observable_similarity,
                                            "memory_similarity": memory_similarity,
                                            "recovered": recovered,
                                            "competitor_won": float((pred_id != eval_target) and (dominance < 0)),
                                            "false_positive": false_positive,
                                            "chance": CHANCE,
                                        }
                                    )

                                    done += 1
                                    if progress and (
                                        done == 1
                                        or done % progress_every == 0
                                        or done == total_jobs
                                    ):
                                        print_progress(done, total_jobs, start)

    raw = pd.DataFrame(rows)

    summary_condition_mode = (
        raw.groupby(["condition", "mode"], as_index=False)
        .agg(
            target_similarity=("target_similarity", "mean"),
            true_target_similarity=("true_target_similarity", "mean"),
            nearest_non_target=("nearest_non_target", "mean"),
            dominance=("dominance", "mean"),
            recovered=("recovered", "mean"),
            competitor_won=("competitor_won", "mean"),
            false_positive=("false_positive", "mean"),
            observable_similarity=("observable_similarity", "mean"),
            memory_similarity=("memory_similarity", "mean"),
            count=("recovered", "size"),
        )
        .sort_values(["condition", "recovered", "dominance"], ascending=[True, False, False])
    )

    summary_mode = (
        raw.groupby("mode", as_index=False)
        .agg(
            recovered=("recovered", "mean"),
            dominance=("dominance", "mean"),
            false_positive=("false_positive", "mean"),
            count=("recovered", "size"),
        )
        .sort_values("recovered", ascending=False)
    )

    def rec(condition: str, mode: str) -> float:
        m = summary_condition_mode[
            (summary_condition_mode["condition"] == condition)
            & (summary_condition_mode["mode"] == mode)
        ]
        return float(m["recovered"].iloc[0])

    interpretation: Dict[str, float | bool] = {
        "chance": CHANCE,
        "observable_only_recovery": rec("coherent_memory", "observable_only"),
        "speed_only_no_memory_recovery": rec("coherent_memory", "speed_only_no_memory"),
        "coherent_memory_bounded_recovery": rec("coherent_memory", "memory_bounded_arbiter"),
        "coherent_memory_only_recovery": rec("coherent_memory", "latent_memory_only"),
        "destroyed_memory_recovery": rec("destroyed_memory", "memory_bounded_arbiter"),
        "wrong_memory_recovery": rec("wrong_memory", "memory_bounded_arbiter"),
        "adversarial_memory_recovery": rec("adversarial_memory", "memory_bounded_arbiter"),
        "shuffled_memory_recovery": rec("shuffled_memory", "memory_bounded_arbiter"),
        "memory_swap_recovery": rec("memory_swap", "memory_bounded_arbiter"),
        "shuffled_label_recovery": rec("coherent_memory", "shuffled_label_control"),
        "shuffled_label_false_positive": float(
            summary_condition_mode[
                (summary_condition_mode["condition"] == "coherent_memory")
                & (summary_condition_mode["mode"] == "shuffled_label_control")
            ]["false_positive"].iloc[0]
        ),
    }

    interpretation["anti_cheat_observable_pass"] = (
        interpretation["observable_only_recovery"] <= CHANCE + CHANCE_TOL
    )
    interpretation["anti_cheat_speed_pass"] = (
        interpretation["speed_only_no_memory_recovery"] <= CHANCE + CHANCE_TOL
    )
    interpretation["coherent_memory_success"] = (
        interpretation["coherent_memory_bounded_recovery"] >= SUCCESS_FLOOR
    )
    interpretation["wrong_memory_collapse"] = (
        interpretation["wrong_memory_recovery"] <= COLLAPSE_CEIL
    )
    interpretation["destroyed_memory_collapse"] = (
        interpretation["destroyed_memory_recovery"] <= COLLAPSE_CEIL
    )
    interpretation["shuffled_label_pass"] = (
        interpretation["shuffled_label_recovery"] <= CHANCE + CHANCE_TOL
    )

    interpretation["strict_memory_necessity_supported"] = bool(
        interpretation["anti_cheat_observable_pass"]
        and interpretation["anti_cheat_speed_pass"]
        and interpretation["coherent_memory_success"]
        and interpretation["wrong_memory_collapse"]
        and interpretation["destroyed_memory_collapse"]
        and interpretation["shuffled_label_pass"]
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUTDIR / "raw_results.csv", index=False)
    summary_condition_mode.to_csv(OUTDIR / "summary_by_condition_mode.csv", index=False)
    summary_mode.to_csv(OUTDIR / "summary_by_mode.csv", index=False)

    with open(OUTDIR / "interpretation.json", "w", encoding="utf-8") as f:
        json.dump(interpretation, f, indent=2, ensure_ascii=False)

    if progress:
        print(f"[write] results written to {OUTDIR}", flush=True)

    return summary_condition_mode, summary_mode, interpretation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict double-blind observable-alias memory-necessity benchmark."
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress messages.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100_000,
        help="Print progress every N scored jobs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_condition_mode, summary_mode, interpretation = run(
        progress=not args.no_progress,
        progress_every=max(1, args.progress_every),
    )

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 240)

    print("\n=== SUMMARY BY CONDITION × MODE ===")
    print(summary_condition_mode.to_string(index=False))

    print("\n=== SUMMARY BY MODE ===")
    print(summary_mode.to_string(index=False))

    print("\n=== INTERPRETATION ===")
    for k, v in interpretation.items():
        if isinstance(v, float):
            print(f"{k:40s} {v:.4f}")
        else:
            print(f"{k:40s} {v}")

    if interpretation["strict_memory_necessity_supported"]:
        print("\n[RESULT] Strict memory necessity is supported under true observable aliasing and anti-cheat controls.")
    else:
        print("\n[RESULT] Strict memory necessity is NOT yet decisive; inspect failed anti-cheat flags above.")

    print(f"\n[OK] wrote results to {OUTDIR}")
    print("[DONE] Strict double-blind memory-necessity benchmark complete")


if __name__ == "__main__":
    main()# -----------------------------
SEED = 424242
OUTDIR = Path("results/generated/strict_double_blind_memory_necessity")

N_GROUPS = 64
ALIAS_SIZE = 8
N_IDENTITIES = N_GROUPS * ALIAS_SIZE
OBS_DIM = 48
MEM_DIM = 96

# Repeats are cheap; increase if you want tighter confidence intervals.
N_REPEATS = 32

ARBITER_SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
MEMORY_CAPS = [0.25, 0.50, 0.75, 1.00]
MEMORY_LEAKS = [0.0, 0.10]

# Conservative tolerance around chance for anti-cheat verdicts.
CHANCE = 1.0 / ALIAS_SIZE
CHANCE_TOL = 0.035
SUCCESS_FLOOR = 0.92
COLLAPSE_CEIL = 0.25


# -----------------------------
# Utilities
# -----------------------------
def normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(b @ a)


def stable_soft_clip(x: np.ndarray, cap: float) -> np.ndarray:
    """Memory constitution bounds expression without changing dimensionality."""
    cap = float(max(cap, 1e-6))
    return np.tanh(x / cap) * cap


def speed_transform(x: np.ndarray, speed: float) -> np.ndarray:
    """Unbounded-speed-like sharpening, then renormalization.

    Higher speed makes the arbiter expression more extreme, but it still cannot
    create identity information without the memory vector.
    """
    y = np.tanh(float(speed) * x)
    return normalize(y)


def choose_with_random_tie(scores: np.ndarray, rng: np.random.Generator) -> int:
    m = np.max(scores)
    tied = np.flatnonzero(np.isclose(scores, m, atol=1e-12, rtol=1e-12))
    return int(rng.choice(tied))


@dataclass(frozen=True)
class World:
    observable_by_group: np.ndarray      # [N_GROUPS, OBS_DIM]
    observable_by_id: np.ndarray         # [N_IDENTITIES, OBS_DIM], exact alias repeats
    memory_by_id: np.ndarray             # [N_IDENTITIES, MEM_DIM]
    group_of_id: np.ndarray              # [N_IDENTITIES]
    ids_by_group: np.ndarray             # [N_GROUPS, ALIAS_SIZE]


def make_world(rng: np.random.Generator) -> World:
    observable_by_group = normalize(rng.normal(size=(N_GROUPS, OBS_DIM)))

    ids_by_group = np.arange(N_IDENTITIES).reshape(N_GROUPS, ALIAS_SIZE)
    group_of_id = np.repeat(np.arange(N_GROUPS), ALIAS_SIZE)

    # Strict observable alias: all identities in a group share EXACTLY the same vector.
    observable_by_id = observable_by_group[group_of_id].copy()

    # Identity-carrying latent/memory basis.
    # Not perfectly orthogonal, but high-dimensional enough for reliable coherent recovery.
    memory_by_id = normalize(rng.normal(size=(N_IDENTITIES, MEM_DIM)))

    return World(
        observable_by_group=observable_by_group,
        observable_by_id=observable_by_id,
        memory_by_id=memory_by_id,
        group_of_id=group_of_id,
        ids_by_group=ids_by_group,
    )


def memory_query_for_condition(
    condition: str,
    target_id: int,
    candidate_ids: np.ndarray,
    world: World,
    rng: np.random.Generator,
) -> np.ndarray:
    if condition == "coherent_memory":
        q = world.memory_by_id[target_id]
    elif condition == "destroyed_memory":
        q = normalize(rng.normal(size=MEM_DIM))
    elif condition == "wrong_memory":
        wrong_pool = candidate_ids[candidate_ids != target_id]
        wrong_id = int(rng.choice(wrong_pool))
        q = world.memory_by_id[wrong_id]
    elif condition == "adversarial_memory":
        # Pick the strongest same-alias competitor against target memory.
        target_mem = world.memory_by_id[target_id]
        comp = candidate_ids[candidate_ids != target_id]
        sims = world.memory_by_id[comp] @ target_mem
        adv_id = int(comp[np.argmin(sims)])  # maximally anti-target among local candidates
        q = -world.memory_by_id[target_id] + 0.75 * world.memory_by_id[adv_id]
        q = normalize(q)
    elif condition == "shuffled_memory":
        q = world.memory_by_id[int(rng.integers(0, N_IDENTITIES))]
    elif condition == "memory_swap":
        # Memory from another alias group: should not identify the local target.
        g = world.group_of_id[target_id]
        other_groups = np.flatnonzero(np.arange(N_GROUPS) != g)
        other_g = int(rng.choice(other_groups))
        other_id = int(rng.choice(world.ids_by_group[other_g]))
        q = world.memory_by_id[other_id]
    else:
        raise ValueError(f"Unknown memory condition: {condition}")
    return normalize(q)


def score_mode(
    mode: str,
    obs_q: np.ndarray,
    mem_q: np.ndarray,
    target_id: int,
    candidate_ids: np.ndarray,
    world: World,
    speed: float,
    cap: float,
    leak: float,
    rng: np.random.Generator,
) -> np.ndarray:
    obs_c = world.observable_by_id[candidate_ids]
    mem_c = world.memory_by_id[candidate_ids]

    obs_scores = obs_c @ obs_q
    mem_scores = mem_c @ mem_q

    bounded_q = normalize(stable_soft_clip(mem_q, cap))
    bounded_c = normalize(stable_soft_clip(mem_c, cap), axis=1)
    bounded_scores = bounded_c @ bounded_q

    arb_q = speed_transform(bounded_q, speed)
    arb_c = normalize(np.vstack([speed_transform(v, speed) for v in bounded_c]), axis=1)
    arb_scores = arb_c @ arb_q

    if mode == "exact_template_upper_bound":
        scores = np.full(len(candidate_ids), -1.0)
        scores[np.where(candidate_ids == target_id)[0][0]] = 1.0
        return scores

    if mode == "observable_only":
        return obs_scores

    if mode == "null_memory":
        return obs_scores  # explicit null: must equal observable-only under strict aliasing

    if mode == "latent_memory_only":
        return mem_scores

    if mode == "arbiter_only":
        return arb_scores

    if mode == "speed_only_no_memory":
        # Uses speed and observable gates, but no identity memory. Under exact aliases,
        # this must remain at chance.
        gate = math.tanh(speed) * (1.0 + leak)
        return gate * obs_scores

    if mode == "memory_bounded_arbiter":
        # Observable is allowed as context but carries no within-alias identity.
        # Memory/arbiter must do the work.
        return 0.20 * obs_scores + 0.45 * bounded_scores + 0.35 * arb_scores

    if mode == "overdriven_arbiter":
        # Deliberately less memory-constitutional: should be worse/less stable.
        over_q = speed_transform(mem_q, speed * 4.0)
        over_c = normalize(np.vstack([speed_transform(v, speed * 4.0) for v in mem_c]), axis=1)
        return 0.20 * obs_scores + 0.80 * (over_c @ over_q)

    if mode == "shuffled_label_control":
        # Same score family as memory_bounded_arbiter; evaluation target is shuffled later.
        return 0.20 * obs_scores + 0.45 * bounded_scores + 0.35 * arb_scores

    raise ValueError(f"Unknown mode: {mode}")


def run() -> Tuple[pd.DataFrame, Dict[str, float]]:
    rng = np.random.default_rng(SEED)
    world = make_world(rng)

    conditions = [
        "coherent_memory",
        "destroyed_memory",
        "wrong_memory",
        "adversarial_memory",
        "shuffled_memory",
        "memory_swap",
    ]
    modes = [
        "exact_template_upper_bound",
        "observable_only",
        "null_memory",
        "speed_only_no_memory",
        "latent_memory_only",
        "arbiter_only",
        "overdriven_arbiter",
        "memory_bounded_arbiter",
        "shuffled_label_control",
    ]

    rows: List[dict] = []

    for repeat in range(N_REPEATS):
        for g in range(N_GROUPS):
            candidate_ids = world.ids_by_group[g]
            for target_id in candidate_ids:
                obs_q = world.observable_by_id[target_id]

                for condition in conditions:
                    mem_q = memory_query_for_condition(condition, int(target_id), candidate_ids, world, rng)

                    # Shuffled-label target is deliberately unrelated but same alias group.
                    shuffled_target_id = int(rng.choice(candidate_ids))

                    for speed in ARBITER_SPEEDS:
                        for cap in MEMORY_CAPS:
                            for leak in MEMORY_LEAKS:
                                for mode in modes:
                                    scores = score_mode(
                                        mode=mode,
                                        obs_q=obs_q,
                                        mem_q=mem_q,
                                        target_id=int(target_id),
                                        candidate_ids=candidate_ids,
                                        world=world,
                                        speed=speed,
                                        cap=cap,
                                        leak=leak,
                                        rng=rng,
                                    )

                                    pred_local = choose_with_random_tie(scores, rng)
                                    pred_id = int(candidate_ids[pred_local])

                                    eval_target = shuffled_target_id if mode == "shuffled_label_control" else int(target_id)
                                    recovered = float(pred_id == eval_target)
                                    false_positive = float(mode == "shuffled_label_control" and pred_id == int(target_id))

                                    target_local = int(np.where(candidate_ids == eval_target)[0][0])
                                    target_score = float(scores[target_local])
                                    non_target = np.delete(scores, target_local)
                                    nearest_non_target = float(np.max(non_target))
                                    dominance = target_score - nearest_non_target

                                    true_target_local = int(np.where(candidate_ids == int(target_id))[0][0])
                                    true_target_score = float(scores[true_target_local])

                                    rows.append({
                                        "repeat": repeat,
                                        "group": g,
                                        "target_id": int(target_id),
                                        "eval_target_id": int(eval_target),
                                        "pred_id": pred_id,
                                        "condition": condition,
                                        "mode": mode,
                                        "arbiter_speed": speed,
                                        "memory_cap": cap,
                                        "memory_leak": leak,
                                        "target_similarity": target_score,
                                        "true_target_similarity": true_target_score,
                                        "nearest_non_target": nearest_non_target,
                                        "dominance": dominance,
                                        "observable_similarity": float((world.observable_by_id[candidate_ids] @ obs_q).mean()),
                                        "memory_similarity": float(world.memory_by_id[eval_target] @ mem_q),
                                        "recovered": recovered,
                                        "competitor_won": float((pred_id != eval_target) and (dominance < 0)),
                                        "false_positive": false_positive,
                                        "chance": CHANCE,
                                    })

    raw = pd.DataFrame(rows)

    summary_condition_mode = (
        raw.groupby(["condition", "mode"], as_index=False)
        .agg(
            target_similarity=("target_similarity", "mean"),
            true_target_similarity=("true_target_similarity", "mean"),
            nearest_non_target=("nearest_non_target", "mean"),
            dominance=("dominance", "mean"),
            recovered=("recovered", "mean"),
            competitor_won=("competitor_won", "mean"),
            false_positive=("false_positive", "mean"),
            observable_similarity=("observable_similarity", "mean"),
            memory_similarity=("memory_similarity", "mean"),
            count=("recovered", "size"),
        )
        .sort_values(["condition", "recovered", "dominance"], ascending=[True, False, False])
    )

    summary_mode = (
        raw.groupby("mode", as_index=False)
        .agg(
            recovered=("recovered", "mean"),
            dominance=("dominance", "mean"),
            false_positive=("false_positive", "mean"),
            count=("recovered", "size"),
        )
        .sort_values("recovered", ascending=False)
    )

    def rec(condition: str, mode: str) -> float:
        m = summary_condition_mode[
            (summary_condition_mode["condition"] == condition)
            & (summary_condition_mode["mode"] == mode)
        ]
        return float(m["recovered"].iloc[0])

    interpretation = {
        "chance": CHANCE,
        "observable_only_recovery": rec("coherent_memory", "observable_only"),
        "speed_only_no_memory_recovery": rec("coherent_memory", "speed_only_no_memory"),
        "coherent_memory_bounded_recovery": rec("coherent_memory", "memory_bounded_arbiter"),
        "coherent_memory_only_recovery": rec("coherent_memory", "latent_memory_only"),
        "destroyed_memory_recovery": rec("destroyed_memory", "memory_bounded_arbiter"),
        "wrong_memory_recovery": rec("wrong_memory", "memory_bounded_arbiter"),
        "adversarial_memory_recovery": rec("adversarial_memory", "memory_bounded_arbiter"),
        "shuffled_memory_recovery": rec("shuffled_memory", "memory_bounded_arbiter"),
        "memory_swap_recovery": rec("memory_swap", "memory_bounded_arbiter"),
        "shuffled_label_recovery": rec("coherent_memory", "shuffled_label_control"),
        "shuffled_label_false_positive": float(
            summary_condition_mode[
                (summary_condition_mode["condition"] == "coherent_memory")
                & (summary_condition_mode["mode"] == "shuffled_label_control")
            ]["false_positive"].iloc[0]
        ),
    }

    interpretation["anti_cheat_observable_pass"] = (
        interpretation["observable_only_recovery"] <= CHANCE + CHANCE_TOL
    )
    interpretation["anti_cheat_speed_pass"] = (
        interpretation["speed_only_no_memory_recovery"] <= CHANCE + CHANCE_TOL
    )
    interpretation["coherent_memory_success"] = (
        interpretation["coherent_memory_bounded_recovery"] >= SUCCESS_FLOOR
    )
    interpretation["wrong_memory_collapse"] = (
        interpretation["wrong_memory_recovery"] <= COLLAPSE_CEIL
    )
    interpretation["destroyed_memory_collapse"] = (
        interpretation["destroyed_memory_recovery"] <= COLLAPSE_CEIL
    )
    interpretation["shuffled_label_pass"] = (
        interpretation["shuffled_label_recovery"] <= CHANCE + CHANCE_TOL
    )

    interpretation["strict_memory_necessity_supported"] = bool(
        interpretation["anti_cheat_observable_pass"]
        and interpretation["anti_cheat_speed_pass"]
        and interpretation["coherent_memory_success"]
        and interpretation["wrong_memory_collapse"]
        and interpretation["destroyed_memory_collapse"]
        and interpretation["shuffled_label_pass"]
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUTDIR / "raw_results.csv", index=False)
    summary_condition_mode.to_csv(OUTDIR / "summary_by_condition_mode.csv", index=False)
    summary_mode.to_csv(OUTDIR / "summary_by_mode.csv", index=False)
    with open(OUTDIR / "interpretation.json", "w", encoding="utf-8") as f:
        json.dump(interpretation, f, indent=2, ensure_ascii=False)

    return summary_condition_mode, summary_mode, interpretation


def main() -> None:
    summary_condition_mode, summary_mode, interpretation = run()

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 240)

    print("\n=== SUMMARY BY CONDITION × MODE ===")
    print(summary_condition_mode.to_string(index=False))

    print("\n=== SUMMARY BY MODE ===")
    print(summary_mode.to_string(index=False))

    print("\n=== INTERPRETATION ===")
    for k, v in interpretation.items():
        if isinstance(v, float):
            print(f"{k:40s} {v:.4f}")
        else:
            print(f"{k:40s} {v}")

    if interpretation["strict_memory_necessity_supported"]:
        print("\n[RESULT] Strict memory necessity is supported under true observable aliasing and anti-cheat controls.")
    else:
        print("\n[RESULT] Strict memory necessity is NOT yet decisive; inspect failed anti-cheat flags above.")

    print(f"\n[OK] wrote results to {OUTDIR}")
    print("[DONE] Strict double-blind memory-necessity benchmark complete")


if __name__ == "__main__":
    main()
