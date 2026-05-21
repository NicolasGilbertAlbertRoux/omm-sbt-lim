#!/usr/bin/env python3
# ============================================================
# Preliminary memory-necessity anti-cheat experiment — Memory Necessity / Anti-Cheat Protocol
# ------------------------------------------------------------
# Goal:
#   Test whether recovery truly depends on coherent,
#   episode-specific latent memory, or whether residual
#   correlations still allow cheating.
#
# Run:
#   python preliminary_memory_necessity_anticheat.py
#
# Outputs:
#   results/generated/preliminary_memory_necessity_anticheat/
# ============================================================

import os
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd


OUTDIR = "results/generated/preliminary_memory_necessity_anticheat"
SEED = 40340


@dataclass(frozen=True)
class Config:
    n_episodes: int = 96
    n_states: int = 8
    dim: int = 64
    latent_dim: int = 64
    noise: float = 0.08
    memory_strength: float = 0.85
    observable_strength: float = 0.35
    arbiter_strength: float = 0.50
    adversarial_strength: float = 0.90


def unit(x, axis=-1, eps=1e-12):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def make_orthogonal_matrix(rng, dim):
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q


def make_episode(rng, cfg: Config):
    """
    Full cross-instance isolation:
    - independent latent basis
    - independent observable projection
    - independent coherent memory transition
    - independent arbiter field
    """
    latent_basis = unit(rng.normal(size=(cfg.n_states, cfg.latent_dim)))
    obs_proj = make_orthogonal_matrix(rng, cfg.dim)[: cfg.latent_dim, :]

    observable = unit(
        latent_basis @ obs_proj
        + cfg.noise * rng.normal(size=(cfg.n_states, cfg.dim))
    )

    transition = np.roll(np.eye(cfg.n_states), shift=1, axis=1)
    transition += 0.35 * np.eye(cfg.n_states)
    transition = transition / transition.sum(axis=1, keepdims=True)

    memory_latent = unit(transition @ latent_basis)
    memory = unit(
        memory_latent @ obs_proj
        + cfg.noise * rng.normal(size=(cfg.n_states, cfg.dim))
    )

    arbiter = unit(
        cfg.arbiter_strength * memory
        + (1.0 - cfg.arbiter_strength) * observable
        + cfg.noise * rng.normal(size=(cfg.n_states, cfg.dim))
    )

    return {
        "latent_basis": latent_basis,
        "observable": observable,
        "memory": memory,
        "arbiter": arbiter,
        "transition": transition,
    }


def corrupt_destroy_memory(rng, memory, severity=1.0):
    """
    Destroys topology, adjacency, continuity and coordinates.
    This is stronger than a simple label shuffle.
    """
    n, d = memory.shape
    perm_rows = rng.permutation(n)
    perm_cols = rng.permutation(d)

    destroyed = memory[perm_rows][:, perm_cols]
    destroyed = unit((1.0 - severity) * memory + severity * destroyed)
    destroyed = unit(destroyed + 0.25 * severity * rng.normal(size=memory.shape))
    return destroyed


def adversarial_memory(rng, episode, target_idx, cfg: Config):
    """
    Builds a coherent-looking memory that points away from target
    and toward a nearest competitor.
    """
    obs = episode["observable"]
    target = obs[target_idx]

    sims = obs @ target
    order = np.argsort(sims)[::-1]
    competitors = [i for i in order if i != target_idx][:2]

    if not competitors:
        competitors = [int((target_idx + 1) % cfg.n_states)]

    competitor_vec = unit(np.mean(obs[competitors], axis=0))

    memory = episode["memory"].copy()
    memory[target_idx] = unit(
        cfg.adversarial_strength * competitor_vec
        - 0.35 * target
        + 0.25 * rng.normal(size=cfg.dim)
    )
    return unit(memory)


def reconstruct(observable_vec, memory_vec, arbiter_vec, cfg: Config, mode):
    """
    Each mode receives only the channels it is supposed to receive.
    """
    if mode == "observable_only":
        return observable_vec

    if mode == "null_memory":
        return unit(cfg.observable_strength * observable_vec)

    if mode == "speed_only_no_memory":
        speed_gate = 1.0 / (1.0 + np.exp(-4.0 * (np.linalg.norm(arbiter_vec) - 0.85)))
        return unit(cfg.observable_strength * observable_vec + 0.15 * speed_gate * arbiter_vec)

    if mode == "latent_memory_only":
        return memory_vec

    if mode == "arbiter_only":
        return arbiter_vec

    if mode == "memory_bounded_arbiter":
        gate = max(0.0, cosine(memory_vec, arbiter_vec))
        return unit(
            cfg.observable_strength * observable_vec
            + cfg.memory_strength * memory_vec
            + gate * cfg.arbiter_strength * arbiter_vec
        )

    if mode == "overdriven_arbiter":
        return unit(
            0.15 * observable_vec
            + 0.25 * memory_vec
            + 1.25 * arbiter_vec
        )

    if mode == "exact_template_upper_bound":
        return None

    raise ValueError(f"Unknown mode: {mode}")


def evaluate_candidate(candidate, templates, target_idx):
    if candidate is None:
        sims = np.full(len(templates), -1.0)
        sims[target_idx] = 1.0
        target_similarity = 1.0
    else:
        sims = templates @ candidate
        target_similarity = float(sims[target_idx])

    winner = int(np.argmax(sims))
    nearest_non_target = float(np.max(np.delete(sims, target_idx)))
    dominance = float(target_similarity - nearest_non_target)
    recovered = float(winner == target_idx)
    competitor_won = float(winner != target_idx)

    return target_similarity, nearest_non_target, dominance, recovered, competitor_won


def run():
    cfg = Config()
    rng = np.random.default_rng(SEED)
    os.makedirs(OUTDIR, exist_ok=True)

    memory_caps = [0.15, 0.30, 0.50, 0.75, 1.00]
    memory_leaks = [0.00, 0.10, 0.25]
    swap_distances = [0, 1, 7, 31]
    destroy_severities = [0.25, 0.50, 0.75, 1.00]

    modes = [
        "exact_template_upper_bound",
        "memory_bounded_arbiter",
        "latent_memory_only",
        "arbiter_only",
        "overdriven_arbiter",
        "speed_only_no_memory",
        "observable_only",
        "null_memory",
    ]

    memory_conditions = [
        "coherent_memory",
        "destroyed_memory",
        "wrong_coherent_memory",
        "shuffled_memory",
        "adversarial_memory",
        "memory_swap",
    ]

    episodes = [make_episode(rng, cfg) for _ in range(cfg.n_episodes)]
    rows = []

    for memory_cap, memory_leak, swap_distance, destroy_severity in product(
        memory_caps, memory_leaks, swap_distances, destroy_severities
    ):
        for ep_idx, episode in enumerate(episodes):
            templates = episode["observable"]

            for target_idx in range(cfg.n_states):
                observable_vec = unit(
                    (1.0 - memory_leak) * templates[target_idx]
                    + memory_leak * episode["memory"][target_idx]
                    + cfg.noise * rng.normal(size=cfg.dim)
                )

                for memory_condition in memory_conditions:
                    if memory_condition == "coherent_memory":
                        memory = episode["memory"]
                        arbiter = episode["arbiter"]

                    elif memory_condition == "destroyed_memory":
                        memory = corrupt_destroy_memory(rng, episode["memory"], destroy_severity)
                        arbiter = corrupt_destroy_memory(rng, episode["arbiter"], destroy_severity)

                    elif memory_condition == "wrong_coherent_memory":
                        wrong_idx = (ep_idx + max(1, swap_distance)) % cfg.n_episodes
                        memory = episodes[wrong_idx]["memory"]
                        arbiter = episodes[wrong_idx]["arbiter"]

                    elif memory_condition == "shuffled_memory":
                        perm = rng.permutation(cfg.n_states)
                        memory = episode["memory"][perm]
                        arbiter = episode["arbiter"][perm]

                    elif memory_condition == "adversarial_memory":
                        memory = adversarial_memory(rng, episode, target_idx, cfg)
                        arbiter = corrupt_destroy_memory(rng, episode["arbiter"], destroy_severity)

                    elif memory_condition == "memory_swap":
                        swap_idx = (ep_idx + swap_distance) % cfg.n_episodes
                        memory = episodes[swap_idx]["memory"]
                        arbiter = episodes[swap_idx]["arbiter"]

                    else:
                        raise ValueError(memory_condition)

                    memory_vec_raw = memory[target_idx]
                    arbiter_vec_raw = arbiter[target_idx]

                    memory_vec = unit(memory_cap * memory_vec_raw + (1.0 - memory_cap) * observable_vec)
                    arbiter_vec = unit(memory_cap * arbiter_vec_raw + (1.0 - memory_cap) * observable_vec)

                    for mode in modes:
                        candidate = reconstruct(observable_vec, memory_vec, arbiter_vec, cfg, mode)

                        target_similarity, nearest_non_target, dominance, recovered, competitor_won = evaluate_candidate(
                            candidate, templates, target_idx
                        )

                        shuffled_target = int((target_idx + rng.integers(1, cfg.n_states)) % cfg.n_states)
                        shuffled_similarity, _, _, shuffled_recovered, _ = evaluate_candidate(
                            candidate, templates, shuffled_target
                        )

                        common = {
                            "memory_condition": memory_condition,
                            "memory_cap": memory_cap,
                            "memory_leak": memory_leak,
                            "swap_distance": swap_distance,
                            "destroy_severity": destroy_severity,
                            "target_similarity": target_similarity,
                            "observable_similarity": cosine(observable_vec, templates[target_idx]),
                            "memory_similarity": cosine(memory_vec, templates[target_idx]),
                            "arbiter_similarity": cosine(arbiter_vec, templates[target_idx]),
                            "nearest_non_target": nearest_non_target,
                            "dominance": dominance,
                            "closure_gain": target_similarity - cosine(observable_vec, templates[target_idx]),
                            "recovered": recovered,
                            "competitor_won": competitor_won,
                        }

                        rows.append({
                            **common,
                            "mode": mode,
                            "scored_similarity": target_similarity,
                            "scored_recovered": recovered,
                            "false_positive": 0.0,
                        })

                        rows.append({
                            **common,
                            "mode": "shuffled_label_control",
                            "scored_similarity": shuffled_similarity,
                            "scored_recovered": shuffled_recovered,
                            "false_positive": shuffled_recovered,
                        })

    df = pd.DataFrame(rows)

    metric_cols = [
        "target_similarity",
        "scored_similarity",
        "observable_similarity",
        "memory_similarity",
        "arbiter_similarity",
        "nearest_non_target",
        "dominance",
        "closure_gain",
        "recovered",
        "scored_recovered",
        "competitor_won",
        "false_positive",
    ]

    def summarize(by):
        return (
            df.groupby(by, dropna=False)[metric_cols]
            .mean()
            .join(df.groupby(by, dropna=False).size().rename("count"))
            .reset_index()
            .sort_values(["scored_recovered", "target_similarity"], ascending=False)
        )

    summary_by_mode = summarize(["mode"])
    summary_by_condition = summarize(["memory_condition", "mode"])
    summary_by_cap = summarize(["memory_cap", "memory_condition", "mode"])
    summary_by_swap = summarize(["swap_distance", "memory_condition", "mode"])
    summary_by_destroy = summarize(["destroy_severity", "memory_condition", "mode"])

    df.to_csv(os.path.join(OUTDIR, "raw_results.csv"), index=False)
    summary_by_mode.to_csv(os.path.join(OUTDIR, "summary_by_mode.csv"), index=False)
    summary_by_condition.to_csv(os.path.join(OUTDIR, "summary_by_condition.csv"), index=False)
    summary_by_cap.to_csv(os.path.join(OUTDIR, "summary_by_memory_cap.csv"), index=False)
    summary_by_swap.to_csv(os.path.join(OUTDIR, "summary_by_swap_distance.csv"), index=False)
    summary_by_destroy.to_csv(os.path.join(OUTDIR, "summary_by_destroy_severity.csv"), index=False)

    print("\n=== SUMMARY BY MODE ===")
    print(summary_by_mode.to_string(index=False))

    print("\n=== SUMMARY BY MEMORY CONDITION × MODE ===")
    print(summary_by_condition.to_string(index=False))

    def val(mode, condition):
        q = df[(df["mode"] == mode) & (df["memory_condition"] == condition)]
        return float(q["scored_recovered"].mean())

    full = val("memory_bounded_arbiter", "coherent_memory")
    destroyed = val("memory_bounded_arbiter", "destroyed_memory")
    wrong = val("memory_bounded_arbiter", "wrong_coherent_memory")
    shuffled = val("memory_bounded_arbiter", "shuffled_memory")
    adversarial = val("memory_bounded_arbiter", "adversarial_memory")
    swap = val("memory_bounded_arbiter", "memory_swap")
    speed = val("speed_only_no_memory", "coherent_memory")
    label_fp = val("shuffled_label_control", "coherent_memory")

    print("\n=== INTERPRETATION ===")
    print(f"coherent_memory_recovery           {full:.4f}")
    print(f"destroyed_memory_recovery          {destroyed:.4f}")
    print(f"wrong_coherent_memory_recovery      {wrong:.4f}")
    print(f"shuffled_memory_recovery           {shuffled:.4f}")
    print(f"adversarial_memory_recovery         {adversarial:.4f}")
    print(f"memory_swap_recovery               {swap:.4f}")
    print(f"speed_only_no_memory_recovery       {speed:.4f}")
    print(f"shuffled_label_false_positive       {label_fp:.4f}")
    print(f"gain_over_destroyed_memory          {full - destroyed:.4f}")
    print(f"gain_over_wrong_memory              {full - wrong:.4f}")
    print(f"gain_over_shuffled_memory           {full - shuffled:.4f}")
    print(f"gain_over_adversarial_memory        {full - adversarial:.4f}")
    print(f"gain_over_memory_swap               {full - swap:.4f}")
    print(f"gain_over_speed_only                {full - speed:.4f}")

    if (
        full > 0.90
        and destroyed < 0.50
        and wrong < 0.50
        and shuffled < 0.50
        and adversarial < 0.35
        and speed < 0.60
        and label_fp < 0.20
    ):
        result = "Strong memory-necessity signal with anti-cheat controls."
    elif (
        full > destroyed + 0.20
        and full > wrong + 0.20
        and full > adversarial + 0.25
        and label_fp < 0.20
    ):
        result = "Memory-necessity is supported, but not yet decisive."
    else:
        result = "Memory-necessity remains weak, ambiguous, or confounded."

    print(f"\n[RESULT] {result}")
    print(f"\n[OK] wrote results to {OUTDIR}")
    print("[DONE] Preliminary memory-necessity anti-cheat experiment complete")


if __name__ == "__main__":
    run()
