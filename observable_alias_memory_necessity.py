#!/usr/bin/env python3
# ============================================================
# Observable-alias memory-necessity experiment — Observable-Alias / Memory-Necessity Anti-Cheat
# ------------------------------------------------------------
# Goal:
#   Repair Preliminary memory-necessity anti-cheat experiment by making the observable deliberately non-
#   identifying. Several targets share the same observable alias.
#   Therefore observable-only and speed-only channels should fail
#   unless identity leaks through memory/arbitration.
#
# Core falsification criterion:
#   If coherent memory is necessary, then:
#     coherent memory-bounded arbiter  >> observable-only
#     coherent memory-bounded arbiter  >> speed-only-no-memory
#     coherent memory-bounded arbiter  >> destroyed / wrong / shuffled memory
#     shuffled-label control remains near chance.
#
# Run:
#   python observable_alias_memory_necessity.py
#
# Outputs:
#   results/generated/observable_alias_memory_necessity/
# ============================================================

import os
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd


OUTDIR = "results/generated/observable_alias_memory_necessity"
SEED = 41341


@dataclass(frozen=True)
class Config:
    n_episodes: int = 32
    n_states: int = 8
    n_alias_classes: int = 2
    dim: int = 64
    noise: float = 0.035
    alias_strength: float = 0.94
    identity_leak: float = 0.00
    memory_strength: float = 0.92
    arbiter_strength: float = 0.55
    observable_strength: float = 0.18
    adversarial_strength: float = 0.95


def unit(x, axis=-1, eps=1e-12):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def make_orthogonal_family(rng, n, dim):
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return unit(q[:n])


def make_episode(rng, cfg: Config, alias_leak: float):
    """
    Creates one isolated episode.

    identity_templates are the hidden target identities.
    observable states are intentionally aliased: states in the same alias class
    have almost the same observable vector, so observable-only cannot uniquely
    identify the target.
    """
    identities = make_orthogonal_family(rng, cfg.n_states, cfg.dim)
    alias_carriers = make_orthogonal_family(rng, cfg.n_alias_classes, cfg.dim)

    alias_ids = np.arange(cfg.n_states) % cfg.n_alias_classes

    observable = []
    for i in range(cfg.n_states):
        carrier = alias_carriers[alias_ids[i]]
        obs = unit(
            cfg.alias_strength * carrier
            + alias_leak * identities[i]
            + cfg.noise * rng.normal(size=cfg.dim)
        )
        observable.append(obs)
    observable = unit(np.array(observable))

    # Memory contains identity-specific information but is not simply a copy:
    # it is a bound mixture of target identity and local relational context.
    transition = np.roll(np.eye(cfg.n_states), shift=1, axis=1)
    transition += 0.20 * np.roll(np.eye(cfg.n_states), shift=-1, axis=1)
    transition += 0.35 * np.eye(cfg.n_states)
    transition = transition / transition.sum(axis=1, keepdims=True)

    relational_memory = unit(transition @ identities)
    memory = unit(
        0.78 * identities
        + 0.22 * relational_memory
        + cfg.noise * rng.normal(size=(cfg.n_states, cfg.dim))
    )

    arbiter = unit(
        cfg.arbiter_strength * memory
        + 0.22 * identities
        + 0.12 * observable
        + cfg.noise * rng.normal(size=(cfg.n_states, cfg.dim))
    )

    return {
        "identities": identities,
        "observable": observable,
        "memory": memory,
        "arbiter": arbiter,
        "alias_ids": alias_ids,
    }


def destroy_channel(rng, channel, severity):
    n, d = channel.shape
    row_perm = rng.permutation(n)
    col_perm = rng.permutation(d)
    destroyed = channel[row_perm][:, col_perm]
    destroyed = unit((1.0 - severity) * channel + severity * destroyed)
    destroyed = unit(destroyed + 0.22 * severity * rng.normal(size=channel.shape))
    return destroyed


def adversarial_memory(rng, episode, target_idx, cfg: Config):
    """
    Replace target memory with a coherent-looking vector pointing to a same-alias
    competitor. This is deliberately difficult because same-alias competitors are
    observationally indistinguishable.
    """
    identities = episode["identities"]
    alias_ids = episode["alias_ids"]
    same_alias = np.where(alias_ids == alias_ids[target_idx])[0]
    competitors = [i for i in same_alias if i != target_idx]
    if not competitors:
        competitors = [int((target_idx + 1) % cfg.n_states)]
    comp = int(rng.choice(competitors))

    mem = episode["memory"].copy()
    mem[target_idx] = unit(
        cfg.adversarial_strength * identities[comp]
        - 0.20 * identities[target_idx]
        + cfg.noise * rng.normal(size=cfg.dim)
    )
    return unit(mem)


def reconstruct(observable_vec, memory_vec, arbiter_vec, mode, cfg: Config):
    if mode == "exact_template_upper_bound":
        return None

    if mode == "observable_only":
        return observable_vec

    if mode == "speed_only_no_memory":
        # Only non-directional speed/norm-like gate. No identity-bearing memory.
        speed_gate = 1.0 / (1.0 + np.exp(-6.0 * (np.linalg.norm(arbiter_vec) - 0.95)))
        return unit(cfg.observable_strength * observable_vec + 0.12 * speed_gate * arbiter_vec)

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
        return unit(0.05 * observable_vec + 0.20 * memory_vec + 1.35 * arbiter_vec)

    if mode == "null_memory":
        return unit(cfg.observable_strength * observable_vec)

    raise ValueError(f"Unknown mode: {mode}")


def evaluate_candidate(candidate, identity_templates, target_idx):
    if candidate is None:
        sims = np.full(len(identity_templates), -1.0)
        sims[target_idx] = 1.0
        target_similarity = 1.0
    else:
        sims = identity_templates @ candidate
        target_similarity = float(sims[target_idx])

    winner = int(np.argmax(sims))
    nearest_non_target = float(np.max(np.delete(sims, target_idx)))
    recovered = float(winner == target_idx)
    competitor_won = float(winner != target_idx)
    dominance = float(target_similarity - nearest_non_target)
    return target_similarity, nearest_non_target, dominance, recovered, competitor_won


def run():
    cfg = Config()
    rng = np.random.default_rng(SEED)
    os.makedirs(OUTDIR, exist_ok=True)

    alias_leaks = [0.00, 0.02]
    memory_caps = [0.50, 1.00]
    destroy_severities = [0.70, 1.00]
    swap_distances = [1, 7]

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

    rows = []

    for alias_leak in alias_leaks:
        episodes = [make_episode(rng, cfg, alias_leak) for _ in range(cfg.n_episodes)]

        for memory_cap, destroy_severity, swap_distance in product(
            memory_caps, destroy_severities, swap_distances
        ):
            for ep_idx, episode in enumerate(episodes):
                identities = episode["identities"]
                observable = episode["observable"]

                for target_idx in range(cfg.n_states):
                    observable_vec = unit(
                        observable[target_idx]
                        + cfg.noise * rng.normal(size=cfg.dim)
                    )

                    for memory_condition in memory_conditions:
                        if memory_condition == "coherent_memory":
                            memory = episode["memory"]
                            arbiter = episode["arbiter"]

                        elif memory_condition == "destroyed_memory":
                            memory = destroy_channel(rng, episode["memory"], destroy_severity)
                            arbiter = destroy_channel(rng, episode["arbiter"], destroy_severity)

                        elif memory_condition == "wrong_coherent_memory":
                            wrong_idx = (ep_idx + swap_distance) % cfg.n_episodes
                            memory = episodes[wrong_idx]["memory"]
                            arbiter = episodes[wrong_idx]["arbiter"]

                        elif memory_condition == "shuffled_memory":
                            perm = rng.permutation(cfg.n_states)
                            memory = episode["memory"][perm]
                            arbiter = episode["arbiter"][perm]

                        elif memory_condition == "adversarial_memory":
                            memory = adversarial_memory(rng, episode, target_idx, cfg)
                            arbiter = destroy_channel(rng, episode["arbiter"], destroy_severity)

                        elif memory_condition == "memory_swap":
                            swap_idx = (ep_idx + swap_distance) % cfg.n_episodes
                            memory = episodes[swap_idx]["memory"]
                            arbiter = episodes[swap_idx]["arbiter"]

                        else:
                            raise ValueError(memory_condition)

                        memory_vec = unit(
                            memory_cap * memory[target_idx]
                            + (1.0 - memory_cap) * observable_vec
                        )
                        arbiter_vec = unit(
                            memory_cap * arbiter[target_idx]
                            + (1.0 - memory_cap) * observable_vec
                        )

                        for mode in modes:
                            candidate = reconstruct(observable_vec, memory_vec, arbiter_vec, mode, cfg)
                            target_similarity, nearest_non_target, dominance, recovered, competitor_won = evaluate_candidate(
                                candidate, identities, target_idx
                            )

                            # Shuffled label control: same candidate scored against wrong label.
                            wrong_label = int((target_idx + rng.integers(1, cfg.n_states)) % cfg.n_states)
                            shuffled_similarity, _, _, shuffled_recovered, _ = evaluate_candidate(
                                candidate, identities, wrong_label
                            )

                            alias_id = int(episode["alias_ids"][target_idx])
                            same_alias = [i for i in range(cfg.n_states) if episode["alias_ids"][i] == alias_id and i != target_idx]
                            observable_alias_rank = 1.0 / (len(same_alias) + 1)

                            common = {
                                "alias_leak": alias_leak,
                                "memory_cap": memory_cap,
                                "destroy_severity": destroy_severity,
                                "swap_distance": swap_distance,
                                "memory_condition": memory_condition,
                                "target_similarity": target_similarity,
                                "observable_similarity": cosine(observable_vec, identities[target_idx]),
                                "memory_similarity": cosine(memory_vec, identities[target_idx]),
                                "arbiter_similarity": cosine(arbiter_vec, identities[target_idx]),
                                "nearest_non_target": nearest_non_target,
                                "dominance": dominance,
                                "closure_gain": target_similarity - cosine(observable_vec, identities[target_idx]),
                                "recovered": recovered,
                                "competitor_won": competitor_won,
                                "observable_alias_chance": observable_alias_rank,
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
        "observable_alias_chance",
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
    summary_by_alias_leak = summarize(["alias_leak", "memory_condition", "mode"])
    summary_by_memory_cap = summarize(["memory_cap", "memory_condition", "mode"])
    summary_by_destroy = summarize(["destroy_severity", "memory_condition", "mode"])

    df.to_csv(os.path.join(OUTDIR, "raw_results.csv"), index=False)
    summary_by_mode.to_csv(os.path.join(OUTDIR, "summary_by_mode.csv"), index=False)
    summary_by_condition.to_csv(os.path.join(OUTDIR, "summary_by_condition.csv"), index=False)
    summary_by_alias_leak.to_csv(os.path.join(OUTDIR, "summary_by_alias_leak.csv"), index=False)
    summary_by_memory_cap.to_csv(os.path.join(OUTDIR, "summary_by_memory_cap.csv"), index=False)
    summary_by_destroy.to_csv(os.path.join(OUTDIR, "summary_by_destroy_severity.csv"), index=False)

    print("\n=== SUMMARY BY MODE ===")
    print(summary_by_mode.to_string(index=False))

    print("\n=== SUMMARY BY MEMORY CONDITION × MODE ===")
    print(summary_by_condition.to_string(index=False))

    def val(mode, condition="coherent_memory"):
        q = df[(df["mode"] == mode) & (df["memory_condition"] == condition)]
        return float(q["scored_recovered"].mean())

    full = val("memory_bounded_arbiter", "coherent_memory")
    obs = val("observable_only", "coherent_memory")
    speed = val("speed_only_no_memory", "coherent_memory")
    memory_only = val("latent_memory_only", "coherent_memory")
    arbiter_only = val("arbiter_only", "coherent_memory")
    destroyed = val("memory_bounded_arbiter", "destroyed_memory")
    wrong = val("memory_bounded_arbiter", "wrong_coherent_memory")
    shuffled = val("memory_bounded_arbiter", "shuffled_memory")
    adversarial = val("memory_bounded_arbiter", "adversarial_memory")
    swap = val("memory_bounded_arbiter", "memory_swap")
    label_fp = val("shuffled_label_control", "coherent_memory")

    print("\n=== INTERPRETATION ===")
    print(f"coherent_memory_bounded_recovery    {full:.4f}")
    print(f"latent_memory_only_recovery         {memory_only:.4f}")
    print(f"arbiter_only_recovery               {arbiter_only:.4f}")
    print(f"observable_only_recovery            {obs:.4f}")
    print(f"speed_only_no_memory_recovery       {speed:.4f}")
    print(f"destroyed_memory_recovery           {destroyed:.4f}")
    print(f"wrong_memory_recovery               {wrong:.4f}")
    print(f"shuffled_memory_recovery            {shuffled:.4f}")
    print(f"adversarial_memory_recovery         {adversarial:.4f}")
    print(f"memory_swap_recovery                {swap:.4f}")
    print(f"shuffled_label_false_positive       {label_fp:.4f}")
    print(f"gain_over_observable                {full - obs:.4f}")
    print(f"gain_over_speed_only                {full - speed:.4f}")
    print(f"gain_over_destroyed                 {full - destroyed:.4f}")
    print(f"gain_over_wrong                     {full - wrong:.4f}")
    print(f"gain_over_shuffled                  {full - shuffled:.4f}")
    print(f"gain_over_adversarial               {full - adversarial:.4f}")
    print(f"gain_over_swap                      {full - swap:.4f}")

    if (
        full > 0.90
        and obs < 0.40
        and speed < 0.45
        and destroyed < 0.55
        and wrong < 0.55
        and shuffled < 0.55
        and adversarial < 0.55
        and label_fp < 0.20
    ):
        result = "Strong memory-necessity signal under observable-alias anti-cheat."
    elif (
        full > obs + 0.35
        and full > speed + 0.30
        and full > destroyed + 0.20
        and full > wrong + 0.20
        and label_fp < 0.20
    ):
        result = "Memory-necessity is supported under observable aliasing, but not yet decisive."
    else:
        result = "Memory-necessity remains weak, ambiguous, or confounded."

    print(f"\n[RESULT] {result}")
    print(f"\n[OK] wrote results to {OUTDIR}")
    print("[DONE] Observable-alias memory-necessity experiment complete")


if __name__ == "__main__":
    run()
