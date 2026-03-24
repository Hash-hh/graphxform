import math
from dataclasses import dataclass
from typing import List, Optional
import torch
from contextlib import nullcontext

from molecule_design import MoleculeDesign
from model.molecule_transformer import MoleculeTransformer


@dataclass
class TrajectoryRecord:
    design: MoleculeDesign
    history: List[int]
    reward: float
    log_probs_history: Optional[List[float]] = None
    log_prob_sum: Optional[torch.Tensor] = None
    length: Optional[int] = None
    advantage: Optional[float] = None
    replay_failed: bool = False  # (not used in streaming path unless you extend soft-fail)
    # --- PPO Actor-Critic fields (per-step) ---
    values_history: Optional[List[float]] = None        # V_old(s_t) at each step
    advantages_per_step: Optional[List[float]] = None   # A_t = R_final - V_old(s_t)

def apply_novelty_bonus(records: List[TrajectoryRecord],
                        memory: dict,
                        beta: float) -> float:
    """Applies novelty bonus to records' rewards in-place and updates the memory."""
    if not beta > 0:
        return 0.0

    total_bonus_added = 0.0
    new_smiles_in_batch = {}

    for r in records:
        smiles = r.design.smiles_string
        if not smiles:
            print("[DR-GRPO] Warning: empty SMILES string; skipping novelty bonus.")
            continue

        # Get count from global memory + count from molecules seen so far in this batch
        global_count = memory.get(smiles, 0)
        batch_count = new_smiles_in_batch.get(smiles, 0)
        total_count = global_count + batch_count
        # if total_count != 0:
        #     print(f"[DR-GRPO] Novelty bonus: SMILES {smiles} seen {total_count} times before.")

        # Calculate bonus and add it to the record's reward
        novelty_bonus = 1.0 / math.sqrt(total_count + 1)
        bonus_to_add = beta * novelty_bonus
        r.reward += bonus_to_add
        total_bonus_added += bonus_to_add

        # Update the batch count
        new_smiles_in_batch[smiles] = batch_count + 1

    # After processing the whole batch, update the global memory
    for smiles, count in new_smiles_in_batch.items():
        memory[smiles] = memory.get(smiles, 0) + count

    return total_bonus_added / len(records) if records else 0.0

def filter_and_build_records(designs: List[MoleculeDesign]) -> (List[TrajectoryRecord], int, int):
    records: List[TrajectoryRecord] = []
    none_dropped = 0
    nonfinite_dropped = 0
    for d in designs:
        if not d.history:
            continue
        # if not (d.history[-1] == 0 and d.synthesis_done):
        #     continue
        obj = d.objective

        # if obj is None:
        #     none_dropped += 1
        #     continue
        # if not math.isfinite(obj):
        #     nonfinite_dropped += 1
        #     continue

        # Instead of dropping non-finite (invalid) molecules, give them 0 reward
        if obj is None or not math.isfinite(obj):
            if obj is None:
                none_dropped += 1
                continue
            elif not math.isfinite(obj):
                print("Smiles with non-finite objective:", d.smiles_string, "objective:", obj)
                nonfinite_dropped += 1
                # Treat invalid molecules as 0.0 reward so the agent learns to avoid them
                # We must verify we have history/log_probs to construct the record
                if d.history and d.log_probs_history:
                    records.append(TrajectoryRecord(
                        design=d,
                        history=list(d.history),
                        reward=0.0,  # PENALTY
                        log_probs_history=list(d.log_probs_history)
                    ))
                continue
        # records.append(TrajectoryRecord(design=d, history=list(d.history), reward=float(obj)))
        records.append(TrajectoryRecord(design=d, history=list(d.history), reward=float(obj),
                                        log_probs_history=list(d.log_probs_history)))
    # if none_dropped or nonfinite_dropped:
    print(f"[DR-GRPO] filter: dropped none={none_dropped} nonfinite={nonfinite_dropped} kept={len(records)}")
    return records, none_dropped, nonfinite_dropped


def compute_baseline_and_advantages(records: List[TrajectoryRecord],
                                    normalize: bool = False) -> float:
    if not records:
        return 0.0
    rewards = torch.tensor([r.reward for r in records], dtype=torch.float32)
    baseline = rewards.mean().item()
    advantages = rewards - rewards.mean()
    if normalize:
        std = advantages.std()
        if std > 1e-8:
            advantages = advantages / std
    for i, r in enumerate(records):
        r.advantage = advantages[i].item()
    return baseline


def _fresh_initial_clone(final_design: MoleculeDesign) -> MoleculeDesign:
    """
        Creates the correct starting environment for replay.
        - If it was from a fragment, re-create the fragment.
        - If it was from a single atom, re-create the single atom.
    """
    if final_design.prompt_smiles:
        # Re-create the fragment prompt state
        return MoleculeDesign.from_smiles(
            config=final_design.config,
            smiles=final_design.prompt_smiles,
            do_finish=False
        )

    first_atom_token = int(final_design.atoms[1])
    return MoleculeDesign(config=final_design.config, initial_atom=first_atom_token)


def _prepare_masked_log_probs(level_logits: torch.Tensor,
                              mask_list,
                              device: torch.device,
                              assert_masks: bool = True) -> torch.Tensor:
    if isinstance(mask_list, torch.Tensor):
        infeasible_mask = mask_list.to(device=device, dtype=torch.bool)
    else:
        infeasible_mask = torch.tensor(mask_list, dtype=torch.bool, device=device)
    if assert_masks:
        assert level_logits.shape[0] >= infeasible_mask.shape[0], \
            f"Logits shorter ({level_logits.shape[0]}) than mask ({infeasible_mask.shape[0]})"
    level_logits = level_logits[:infeasible_mask.shape[0]]
    feasible_any = (~infeasible_mask).any().item()
    if assert_masks:
        assert feasible_any, "All actions masked; invalid environment state."
    if not feasible_any:
        level_logits = torch.zeros_like(level_logits)
    level_logits = level_logits.masked_fill(infeasible_mask, float("-inf"))
    return torch.log_softmax(level_logits.float(), dim=-1)
    # return torch.log_softmax(level_logits, dim=-1)

def _make_autocast_ctx(config):
    if not config.use_amp:
        return nullcontext(), None
    amp_dtype = getattr(config, "amp_dtype", "bf16").lower()
    if amp_dtype == "fp16":
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:  # bf16
        scaler = None
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return ctx, scaler

def streaming_replay_and_backward(model: MoleculeTransformer,
                                  optimizer: torch.optim.Optimizer,
                                  records: List[TrajectoryRecord],
                                  config,
                                  device: torch.device,
                                  autocast_ctx,
                                  scaler,
                                  current_epoch: int):
    """
    Streaming (per-step) backward with microbatching.
    For each replay step (single forward over current active subset) we:
      - accumulate a scalar loss_batch across active trajectories
      - call backward exactly once on that scalar
    This avoids reusing a freed graph while keeping activation memory low.
    """
    epsilon = config.rl_ppo_clip_epsilon
    clip_val = config.optimizer.get("gradient_clipping")
    # Get the entropy coefficient from the config
    entropy_beta = config.rl_entropy_beta

    model.eval()  # deterministic
    assert all(r.advantage is not None for r in records), "Compute advantages first."
    N = len(records)
    if N == 0:
        return 0.0

    # Init metrics fields
    for rec in records:
        rec.log_prob_sum = torch.tensor(0.0, device=device)
        rec.length = 0

    optimizer.zero_grad(set_to_none=True)

    micro = getattr(config, "rl_replay_microbatch_size", 0) or N
    assert micro > 0, "Microbatch size must be > 0"

    # For logging: accumulate Σ_i A_i Σ_t logp
    sum_adv_logp = 0.0
    total_ppo_loss = 0.0

    total_entropy = 0.0
    step_count = 0

    # Slice into microbatches of trajectories
    for start in range(0, N, micro):
        batch_records = records[start:start + micro]
        clones = [_fresh_initial_clone(r.design) for r in batch_records]  # replay
        cursors = [0] * len(batch_records)
        finished = [False] * len(batch_records)

        while True:
            active_local_indices = [
                i for i, (r, cur, fin, cl) in enumerate(zip(batch_records, cursors, finished, clones))
                if (not fin) and (cur < len(r.history)) and (not cl.synthesis_done)
            ]
            if not active_local_indices:
                break

            active_clones = [clones[i] for i in active_local_indices]
            with autocast_ctx:
                head0, head1, head2 = model(MoleculeDesign.list_to_batch(active_clones, device=device))

            # Accumulate per-step loss across active trajectories, then backward once.
            if scaler is not None:
                # Use FP32 accumulator outside scaled region? Simpler to accumulate inside.
                loss_batch = 0.0
            else:
                loss_batch = torch.tensor(0.0, device=device, dtype=head0.dtype)

            for pos_in_active, local_idx in enumerate(active_local_indices):
                rec = batch_records[local_idx]
                clone = clones[local_idx]
                cursor = cursors[local_idx]

                lvl = clone.current_action_level
                if lvl == 0:
                    level_logits = head0[pos_in_active]
                elif lvl == 1:
                    level_logits = head1[pos_in_active]
                else:
                    level_logits = head2[pos_in_active]

                log_probs = _prepare_masked_log_probs(
                    level_logits, clone.current_action_mask, device,
                    assert_masks=getattr(config, "rl_assert_masks", False)
                )
                action = rec.history[cursor]
                if action >= log_probs.shape[0]:
                    raise RuntimeError(f"[StreamingReplay] Action {action} out of bounds {log_probs.shape[0]}")

                # DR. GRPO CLIPPED OBJECTIVE
                chosen_logp = log_probs[action]  # from new policy
                # Get advantage
                advantage = rec.advantage

                # Retrieve the old log probability from when the action was originally sampled
                old_logp_float = rec.log_probs_history[cursor]

                if not (math.isfinite(old_logp_float) and torch.isfinite(chosen_logp)):
                    # We can't advance the clone if it's in a bad state,
                    # so we must terminate this trajectory's replay.
                    finished[local_idx] = True
                    continue  # Skip to the next active trajectory in the microbatch

                old_logp = torch.tensor(old_logp_float, device=device, dtype=chosen_logp.dtype)
                ratio = torch.exp(chosen_logp - old_logp)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
                ppo_loss_step = -torch.min(surr1, surr2) / N

                # The PPO loss is the negative of the minimum of the two surrogate objectives.
                # We normalize by the total number of trajectories (N) as in the original implementation.
                contrib = ppo_loss_step

                total_ppo_loss += contrib.detach().cpu().item()

                # Calculate the entropy of the action distribution
                # H(p) = -sum(p * log(p)). Here, p = exp(log_probs).
                finite_mask = torch.isfinite(log_probs)
                finite_log_probs = log_probs[finite_mask]

                # # We only need to compute exp for the finite values.
                finite_probs = torch.exp(finite_log_probs)

                entropy = -torch.sum(finite_probs * finite_log_probs)

                step_count += 1

                # Metrics accumulation
                rec.log_prob_sum = rec.log_prob_sum + chosen_logp.detach().float()
                rec.length += 1
                sum_adv_logp += rec.advantage * float(chosen_logp.detach().cpu())

                entropy_term = (entropy_beta / N) * entropy  # Scaled by 1/N like the main loss
                contrib -= entropy_term

                total_entropy += entropy.detach().cpu().item()

                if scaler is not None:
                    # Accumulate as Python float; we will wrap in a tensor at backward time
                    loss_batch += float(contrib.detach().cpu())
                else:
                    loss_batch = loss_batch + contrib

                # Advance environment
                clone.take_action(action)
                cursors[local_idx] += 1
                if clone.synthesis_done or cursors[local_idx] >= len(rec.history):
                    finished[local_idx] = True

            # Single backward for this step
            if scaler is not None:
                loss_batch_tensor = torch.tensor(loss_batch, device=device, dtype=head0.dtype, requires_grad=True)
                scaler.scale(loss_batch_tensor).backward()
            else:
                loss_batch.backward()

    # Finish optimizer step

    if scaler is not None:
        if clip_val and clip_val > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        scaler.step(optimizer)
        scaler.update()
    else:
        if clip_val and clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()

    mean_entropy = total_entropy / step_count if step_count > 0 else 0.0

    return total_ppo_loss, mean_entropy


def compute_policy_loss_for_logging(records: List[TrajectoryRecord]) -> float:
    if not records:
        return 0.0
    with torch.no_grad():
        log_probs = torch.stack([r.log_prob_sum for r in records])
        adv = torch.tensor([r.advantage for r in records], device=log_probs.device, dtype=log_probs.dtype)
        return float(-(adv * log_probs).mean().item())


def ppo_update(model: MoleculeTransformer,
               optimizer: torch.optim.Optimizer,
               designs_groups: List[List[MoleculeDesign]],
               config,
               device: torch.device,
               logger=None,
               novelty_memory: Optional[dict] = None):
    """
    Full PPO Actor-Critic update.

    Architecture:
        Actor  = existing 3-head policy (level 0/1/2 logits)
        Critic = value_head MLP on the shared transformer backbone → V(s_t)

    Steps:
        1. Flatten all scaffold groups into one batch (no per-group baselines).
        2. No-grad replay to collect V_old(s_t) at every step.
        3. Compute per-step advantages  A_t = R_final − V_old(s_t), normalise.
        4. For K PPO epochs, replay with grad and optimise the combined loss:
               L = −L_clip  +  c1 · L_vf  −  c2 · entropy
    """
    # ---- 1. Flatten & filter ------------------------------------------------
    all_designs_flat: List[MoleculeDesign] = []
    for group in designs_groups:
        all_designs_flat.extend(group)

    records, none_dropped, nonfinite_dropped = filter_and_build_records(all_designs_flat)

    if not records:
        metrics = {
            "skipped": True,
            "num_trajectories": 0,
            "mean_reward": float("-inf"),
            "best_reward": float("-inf"),
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "invalid_dropped": nonfinite_dropped,
            "none_dropped": none_dropped
        }
        if logger:
            logger.info(f"[PPO] {metrics}")
        return metrics

    print(f"[PPO] {len(records)} trajectories (flattened from {len(designs_groups)} groups).")

    # Collect aux metrics
    aux_metrics_sum = {}
    aux_metrics_count = 0
    for r in records:
        if hasattr(r.design, 'aux_metrics') and r.design.aux_metrics:
            for key, val in r.design.aux_metrics.items():
                aux_metrics_sum[key] = aux_metrics_sum.get(key, 0.0) + float(val)
            aux_metrics_count += 1

    # Optional novelty bonus (applied globally)
    avg_novelty_bonus = 0.0
    if novelty_memory is not None and config.rl_use_novelty_bonus:
        novelty_beta = config.rl_novelty_beta
        if novelty_beta > 0:
            avg_novelty_bonus = apply_novelty_bonus(records, novelty_memory, novelty_beta)

    best_objective_score = float(max(r.reward for r in records))

    # ---- 2. Compute V_old(s_t) at every step (no grad) ----------------------
    _compute_old_values(model, records, config, device)

    # ---- 3. Per-step advantages: A_t = R_final − V_old(s_t), normalised -----
    mean_value = _compute_ppo_per_step_advantages(records, normalize=True)

    # Set trajectory-level advantage for logging compatibility
    for rec in records:
        if rec.advantages_per_step:
            rec.advantage = sum(rec.advantages_per_step) / len(rec.advantages_per_step)
        else:
            rec.advantage = 0.0

    # ---- 4. K PPO update epochs ---------------------------------------------
    ppo_epochs = config.ppo_epochs
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_mean_entropy = 0.0

    for i in range(ppo_epochs):
        autocast_ctx, scaler = _make_autocast_ctx(config)
        ploss, vloss, ent = ppo_replay_and_backward(
            model, optimizer, records, config, device, autocast_ctx, scaler, i
        )
        total_policy_loss += ploss
        total_value_loss += vloss
        total_mean_entropy += ent

    total_policy_loss /= ppo_epochs
    total_value_loss /= ppo_epochs
    total_mean_entropy /= ppo_epochs

    # ---- 5. Metrics ---------------------------------------------------------
    rewards = [r.reward for r in records]
    advantages = [r.advantage for r in records]
    mean_reward = sum(rewards) / len(rewards)
    mean_adv = sum(advantages) / len(advantages)
    std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5 if advantages else 0.0

    metrics = {
        "baseline": float(mean_value),
        "mean_reward": float(mean_reward),
        "best_reward": float(max(rewards)),
        "best_objective": best_objective_score,
        "mean_advantage": float(mean_adv),
        "std_advantage": float(std_adv),
        "fraction_pos_adv": float(sum(a > 0 for a in advantages) / len(advantages)),
        "policy_loss": float(total_policy_loss),
        "value_loss": float(total_value_loss),
        "num_trajectories": len(records),
        "mean_traj_length": float(sum(r.length for r in records) / len(records)) if records else 0.0,
        "invalid_dropped": nonfinite_dropped,
        "none_dropped": none_dropped,
        "amp_enabled": bool(getattr(config, "use_amp", False)),
        "amp_dtype": getattr(config, "amp_dtype", "none"),
        "mean_novelty_bonus": float(avg_novelty_bonus),
        "mean_entropy": float(total_mean_entropy),
        "adv_norm": True
    }

    if aux_metrics_count > 0:
        for key, total_val in aux_metrics_sum.items():
            metrics[f"prodrug/{key}"] = total_val / aux_metrics_count

    if logger:
        logger.info(f"[PPO] {metrics}")
    return metrics


# ---------------------------------------------------------------------------
#  PPO helper: compute V_old(s_t) via no-grad replay
# ---------------------------------------------------------------------------
def _compute_old_values(model: MoleculeTransformer,
                        records: List[TrajectoryRecord],
                        config,
                        device: torch.device):
    """
    No-grad forward pass that replays every trajectory step-by-step and stores
    V_old(s_t) (from the value head) in ``rec.values_history``.
    """
    model.eval()
    autocast_ctx, _ = _make_autocast_ctx(config)
    micro = getattr(config, "rl_replay_microbatch_size", 0) or len(records)

    for rec in records:
        rec.values_history = []

    for start in range(0, len(records), micro):
        batch_records = records[start:start + micro]
        clones = [_fresh_initial_clone(r.design) for r in batch_records]
        cursors = [0] * len(batch_records)
        finished = [False] * len(batch_records)

        while True:
            active_local_indices = [
                i for i, (r, cur, fin, cl) in enumerate(
                    zip(batch_records, cursors, finished, clones))
                if (not fin) and (cur < len(r.history)) and (not cl.synthesis_done)
            ]
            if not active_local_indices:
                break

            active_clones = [clones[i] for i in active_local_indices]
            with torch.no_grad(), autocast_ctx:
                _, _, _, values = model(
                    MoleculeDesign.list_to_batch(active_clones, device=device),
                    return_value=True
                )

            for pos_in_active, local_idx in enumerate(active_local_indices):
                rec = batch_records[local_idx]
                clone = clones[local_idx]
                cursor = cursors[local_idx]

                rec.values_history.append(values[pos_in_active].float().item())

                action = rec.history[cursor]
                clone.take_action(action)
                cursors[local_idx] += 1
                if clone.synthesis_done or cursors[local_idx] >= len(rec.history):
                    finished[local_idx] = True


# ---------------------------------------------------------------------------
#  PPO helper: per-step advantages   A_t = R_final − V_old(s_t)
# ---------------------------------------------------------------------------
def _compute_ppo_per_step_advantages(records: List[TrajectoryRecord],
                                     normalize: bool = True) -> float:
    """
    Fills ``rec.advantages_per_step`` for every record.
    Returns the global mean V_old (useful as the "baseline" metric).
    """
    all_advs = []
    all_vals = []
    for rec in records:
        R = rec.reward
        rec.advantages_per_step = [R - v for v in rec.values_history]
        all_advs.extend(rec.advantages_per_step)
        all_vals.extend(rec.values_history)

    mean_val = sum(all_vals) / len(all_vals) if all_vals else 0.0

    if normalize and all_advs:
        mean_a = sum(all_advs) / len(all_advs)
        var_a = sum((a - mean_a) ** 2 for a in all_advs) / len(all_advs)
        std_a = var_a ** 0.5
        if std_a > 1e-8:
            for rec in records:
                rec.advantages_per_step = [
                    (a - mean_a) / std_a for a in rec.advantages_per_step
                ]
        else:
            for rec in records:
                rec.advantages_per_step = [
                    a - mean_a for a in rec.advantages_per_step
                ]

    return mean_val


# ---------------------------------------------------------------------------
#  PPO replay: clipped actor loss + value loss + entropy
# ---------------------------------------------------------------------------
def ppo_replay_and_backward(model: MoleculeTransformer,
                            optimizer: torch.optim.Optimizer,
                            records: List[TrajectoryRecord],
                            config,
                            device: torch.device,
                            autocast_ctx,
                            scaler,
                            current_epoch: int):
    """
    One PPO optimisation pass over the rollout buffer.

    Combined loss per step (averaged over N trajectories):
        L  =  −L_clip  +  c1 · L_vf  −  c2 · H[π]

    where
        L_clip  = min(r·A, clip(r, 1−ε, 1+ε)·A)
        L_vf    = (V_new(s_t) − R_final)²
        H[π]    = entropy of the action distribution

    Returns (total_policy_loss, total_value_loss, mean_entropy).
    """
    epsilon = config.rl_ppo_clip_epsilon
    clip_val = config.optimizer.get("gradient_clipping")
    vf_coeff = getattr(config, "ppo_vf_coeff", 0.5)
    entropy_coeff = getattr(config, "ppo_entropy_coeff", 0.01)

    model.train()
    N = len(records)
    if N == 0:
        return 0.0, 0.0, 0.0

    # Init per-record metric accumulators
    for rec in records:
        rec.log_prob_sum = torch.tensor(0.0, device=device)
        rec.length = 0

    optimizer.zero_grad(set_to_none=True)

    micro = getattr(config, "rl_replay_microbatch_size", 0) or N
    assert micro > 0, "Microbatch size must be > 0"

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    step_count = 0

    for start in range(0, N, micro):
        batch_records = records[start:start + micro]
        clones = [_fresh_initial_clone(r.design) for r in batch_records]
        cursors = [0] * len(batch_records)
        finished = [False] * len(batch_records)

        while True:
            active_local_indices = [
                i for i, (r, cur, fin, cl) in enumerate(
                    zip(batch_records, cursors, finished, clones))
                if (not fin) and (cur < len(r.history)) and (not cl.synthesis_done)
            ]
            if not active_local_indices:
                break

            active_clones = [clones[i] for i in active_local_indices]
            with autocast_ctx:
                head0, head1, head2, values = model(
                    MoleculeDesign.list_to_batch(active_clones, device=device),
                    return_value=True
                )

            if scaler is not None:
                loss_batch = 0.0
            else:
                loss_batch = torch.tensor(0.0, device=device, dtype=head0.dtype)

            for pos_in_active, local_idx in enumerate(active_local_indices):
                rec = batch_records[local_idx]
                clone = clones[local_idx]
                cursor = cursors[local_idx]

                # --- select logits for the correct action level ---
                lvl = clone.current_action_level
                if lvl == 0:
                    level_logits = head0[pos_in_active]
                elif lvl == 1:
                    level_logits = head1[pos_in_active]
                else:
                    level_logits = head2[pos_in_active]

                log_probs = _prepare_masked_log_probs(
                    level_logits, clone.current_action_mask, device,
                    assert_masks=getattr(config, "rl_assert_masks", False)
                )
                action = rec.history[cursor]
                if action >= log_probs.shape[0]:
                    raise RuntimeError(
                        f"[PPO Replay] Action {action} OOB {log_probs.shape[0]}")

                chosen_logp = log_probs[action]
                old_logp_float = rec.log_probs_history[cursor]

                if not (math.isfinite(old_logp_float) and torch.isfinite(chosen_logp)):
                    finished[local_idx] = True
                    continue

                # --- per-step advantage from value baseline ---
                advantage = rec.advantages_per_step[cursor]

                # --- clipped policy (actor) loss ---
                old_logp = torch.tensor(old_logp_float, device=device,
                                        dtype=chosen_logp.dtype)
                ratio = torch.exp(chosen_logp - old_logp)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - epsilon,
                                    1.0 + epsilon) * advantage
                policy_loss = -torch.min(surr1, surr2) / N

                # --- value (critic) loss: (V_new − R_final)² ---
                v_pred = values[pos_in_active]
                v_target = torch.tensor(rec.reward, device=device,
                                        dtype=v_pred.dtype)
                value_loss = ((v_pred - v_target) ** 2) / N

                # --- entropy bonus ---
                finite_mask = torch.isfinite(log_probs)
                finite_lp = log_probs[finite_mask]
                finite_p = torch.exp(finite_lp)
                entropy = -torch.sum(finite_p * finite_lp)

                # --- combined loss: −L_clip + c1·L_vf − c2·H ---
                contrib = policy_loss + vf_coeff * value_loss \
                          - (entropy_coeff / N) * entropy

                total_policy_loss += policy_loss.detach().cpu().item()
                total_value_loss += value_loss.detach().cpu().item()
                total_entropy += entropy.detach().cpu().item()
                step_count += 1

                rec.log_prob_sum = rec.log_prob_sum + chosen_logp.detach().float()
                rec.length += 1

                if scaler is not None:
                    loss_batch += float(contrib.detach().cpu())
                else:
                    loss_batch = loss_batch + contrib

                # Advance environment
                clone.take_action(action)
                cursors[local_idx] += 1
                if clone.synthesis_done or cursors[local_idx] >= len(rec.history):
                    finished[local_idx] = True

            # Single backward for this replay step
            if scaler is not None:
                loss_batch_tensor = torch.tensor(
                    loss_batch, device=device, dtype=head0.dtype,
                    requires_grad=True)
                scaler.scale(loss_batch_tensor).backward()
            else:
                loss_batch.backward()

    # ---- optimizer step ----
    if scaler is not None:
        if clip_val and clip_val > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        scaler.step(optimizer)
        scaler.update()
    else:
        if clip_val and clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()

    mean_entropy = total_entropy / step_count if step_count > 0 else 0.0
    return total_policy_loss, total_value_loss, mean_entropy


def dr_grpo_update(model: MoleculeTransformer,
                   optimizer: torch.optim.Optimizer,
                   designs_groups: List[MoleculeDesign],
                   config,
                   device: torch.device,
                   logger=None,
                   novelty_memory: Optional[dict] = None):
    # records, none_dropped, nonfinite_dropped = filter_and_build_records(designs)
    # if not records:
    #     metrics = {
    #         "skipped": True,
    #         "num_trajectories": 0,
    #         "mean_reward": float("-inf"),
    #         "best_reward": float("-inf"),
    #         "policy_loss": 0.0,
    #         "invalid_dropped": nonfinite_dropped,
    #         "none_dropped": none_dropped
    #     }
    #     if logger:
    #         logger.info(f"[DR-GRPO] {metrics}")
    #     return metrics

    all_records_flat: List[TrajectoryRecord] = []
    total_none_dropped = 0
    total_nonfinite_dropped = 0
    avg_novelty_bonus = 0.0
    total_bonus_count = 0
    all_baselines = []  # For logging
    aux_metrics_sum = {}
    aux_metrics_count = 0

    # # print length of each group:
    # for i, group in enumerate(designs_groups):
    #     print(f"Group {i} length: {len(group)}")

    print(f"[GRPO] Received {len(designs_groups)} groups for update.")

    for i, group in enumerate(designs_groups):
        # Filter this group
        records_group, none_dropped, nonfinite_dropped = filter_and_build_records(group)
        total_none_dropped += none_dropped
        total_nonfinite_dropped += nonfinite_dropped

        if not records_group:
            print(f"[GRPO] Group {i} was empty after filtering.")
            continue

        for r in records_group:
            # Check if our evaluator attached metrics
            if hasattr(r.design, 'aux_metrics') and r.design.aux_metrics:
                for key, val in r.design.aux_metrics.items():
                    # Convert booleans/numpy types to float for summation
                    val_float = float(val)
                    aux_metrics_sum[key] = aux_metrics_sum.get(key, 0.0) + val_float
                aux_metrics_count += 1

        # Apply novelty bonus (optional, now applied per-group)
        if novelty_memory is not None and config.rl_use_novelty_bonus:
            novelty_beta = config.rl_novelty_beta
            if novelty_beta > 0:
                avg_novelty_bonus += apply_novelty_bonus(records_group, novelty_memory, novelty_beta) * len(
                    records_group)
                total_bonus_count += len(records_group)

        # Compute baseline and advantages for this group
        normalize_adv = config.rl_advantage_normalize
        baseline = compute_baseline_and_advantages(records_group, normalize=normalize_adv)
        all_baselines.append(baseline)
        # print(all_baselines)

        # Add the processed records to the flat list
        all_records_flat.extend(records_group)

    if not all_records_flat:
        metrics = {
            "skipped": True,
            "num_trajectories": 0,
            "mean_reward": float("-inf"),
            "best_reward": float("-inf"),
            "policy_loss": 0.0,
            "invalid_dropped": nonfinite_dropped,
            "none_dropped": none_dropped
        }
        if logger:
            logger.info(f"[DR-GRPO] {metrics}")
        return metrics

    best_objective_score = float(max(r.reward for r in all_records_flat)) if all_records_flat else float("-inf")

    if total_bonus_count > 0:
        avg_novelty_bonus /= total_bonus_count

    ppo_epochs = config.ppo_epochs
    total_policy_loss = 0
    total_mean_entropy = 0

    for i in range(ppo_epochs):
        autocast_ctx, scaler = _make_autocast_ctx(config)

        # The model weights are updated in-place on each iteration of this loop
        policy_loss_val, mean_entropy = streaming_replay_and_backward(
            model, optimizer, all_records_flat, config, device, autocast_ctx, scaler, i
        )

        # For logging, we can average the loss and entropy over the update epochs
        total_policy_loss += policy_loss_val
        total_mean_entropy += mean_entropy

    total_policy_loss = total_policy_loss / ppo_epochs
    total_mean_entropy = total_mean_entropy / ppo_epochs

    rewards = [r.reward for r in all_records_flat]
    advantages = [r.advantage for r in all_records_flat]
    mean_reward = sum(rewards) / len(rewards)
    mean_adv = sum(advantages) / len(advantages)
    std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5 if advantages else 0.0

    # Calculate the mean of the *group baselines* for logging
    mean_baseline = sum(all_baselines) / len(all_baselines) if all_baselines else 0.0

    # print("num_trajectories:", len(all_records_flat))
    metrics = {
        "baseline": mean_baseline,
        "mean_reward": float(mean_reward),
        "best_reward": float(max(rewards)),
        "best_objective": best_objective_score,  # This is max(Objective)
        "mean_advantage": float(mean_adv),
        "std_advantage": float(std_adv),
        "fraction_pos_adv": float(sum(a > 0 for a in advantages) / len(advantages)),
        "policy_loss": float(total_policy_loss),
        "num_trajectories": len(all_records_flat),
        "mean_traj_length": float(sum(r.length for r in all_records_flat) / len(all_records_flat)) if all_records_flat else 0.0,
        "invalid_dropped": nonfinite_dropped,
        "none_dropped": none_dropped,
        "amp_enabled": bool(getattr(config, "use_amp", False)),
        "amp_dtype": getattr(config, "amp_dtype", "none"),
        "mean_novelty_bonus": float(avg_novelty_bonus),
        "mean_entropy": float(total_mean_entropy),
        "adv_norm": bool(normalize_adv)
    }

    if aux_metrics_count > 0:
        for key, total_val in aux_metrics_sum.items():
            # Add prefix "prodrug/" to keep logs organized
            metrics[f"prodrug/{key}"] = total_val / aux_metrics_count

    if logger:
        logger.info(f"[DR-GRPO] {metrics}")
    return metrics
