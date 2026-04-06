from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llm_ts_audit.attacks.objectives import (
    build_target_sequence,
    evaluate_attack_objective,
    objective_direction,
)
from llm_ts_audit.config import AttackConfig
from llm_ts_audit.models.base import ForecastModel


@dataclass
class AttackResult:
    adversarial_input: np.ndarray
    clean_prediction: np.ndarray
    adversarial_prediction: np.ndarray
    target_sequence: np.ndarray | None
    clean_objective_value: float
    adversarial_objective_value: float
    query_count: int
    loss_history: list[float]
    epsilon_budget: float
    perturbation_l1: float
    perturbation_l2: float
    perturbation_linf: float


class DGAStyleAttack:
    def __init__(self, config: AttackConfig):
        self.config = config

    def run(
        self,
        model: ForecastModel,
        clean_input: np.ndarray,
        clean_prediction: np.ndarray | None = None,
        ground_truth: np.ndarray | None = None,
        sample_seed: int | None = None,
    ) -> AttackResult:
        rng = np.random.default_rng(self.config.seed if sample_seed is None else sample_seed)
        x_clean = clean_input.astype(np.float32)
        x_adv = x_clean.copy()
        delta = np.zeros_like(x_clean, dtype=np.float32)
        epsilon = self._resolve_epsilon_budget(x_clean)
        objective_goal = objective_direction(self.config.objective)

        if clean_prediction is None:
            clean_prediction = model.predict(x_clean[None, ...])[0]
            query_count = 1
        else:
            query_count = 0
        target = None
        if self.config.objective == "target_match":
            target = build_target_sequence(
                mode=self.config.target_mode,
                reference_input=x_clean,
                clean_prediction=clean_prediction,
                rng=rng,
                source=self.config.target_source,
            )

        clean_objective_value = evaluate_attack_objective(
            objective=self.config.objective,
            prediction=clean_prediction,
            loss_kind=self.config.loss,
            clean_prediction=clean_prediction,
            target=target,
            ground_truth=ground_truth,
        )
        history = [clean_objective_value]

        for _step in range(self.config.steps):
            base_prediction = model.predict(x_adv[None, ...])[0]
            base_objective = evaluate_attack_objective(
                objective=self.config.objective,
                prediction=base_prediction,
                loss_kind=self.config.loss,
                clean_prediction=clean_prediction,
                target=target,
                ground_truth=ground_truth,
            )
            query_count += 1

            gradient_estimate = np.zeros_like(x_adv, dtype=np.float32)
            for _direction_idx in range(self.config.directions_per_step):
                probe_direction = rng.normal(size=x_adv.shape).astype(np.float32)
                direction_norm = float(np.linalg.norm(probe_direction.reshape(-1), ord=2))
                if direction_norm < 1e-12:
                    continue
                probe_direction = probe_direction / direction_norm
                probe = x_adv + self.config.probe_scale * probe_direction
                probe_prediction = model.predict(probe[None, ...])[0]
                probe_objective = evaluate_attack_objective(
                    objective=self.config.objective,
                    prediction=probe_prediction,
                    loss_kind=self.config.loss,
                    clean_prediction=clean_prediction,
                    target=target,
                    ground_truth=ground_truth,
                )
                directional_derivative = (
                    probe_objective - base_objective
                ) / max(self.config.probe_scale, 1e-8)
                gradient_estimate += directional_derivative * probe_direction
                query_count += 1

            gradient_estimate /= max(self.config.directions_per_step, 1)
            update_sign = -1.0 if objective_goal == "minimize" else 1.0
            update = update_sign * self.config.step_size * np.sign(gradient_estimate)
            delta = _project_delta(delta + update, epsilon, self.config.norm).astype(np.float32)
            x_adv = (x_clean + delta).astype(np.float32)

            updated_prediction = model.predict(x_adv[None, ...])[0]
            updated_objective = evaluate_attack_objective(
                objective=self.config.objective,
                prediction=updated_prediction,
                loss_kind=self.config.loss,
                clean_prediction=clean_prediction,
                target=target,
                ground_truth=ground_truth,
            )
            history.append(updated_objective)
            query_count += 1

        adversarial_prediction = model.predict(x_adv[None, ...])[0]
        query_count += 1
        adversarial_objective_value = evaluate_attack_objective(
            objective=self.config.objective,
            prediction=adversarial_prediction,
            loss_kind=self.config.loss,
            clean_prediction=clean_prediction,
            target=target,
            ground_truth=ground_truth,
        )

        perturbation = x_adv - x_clean
        return AttackResult(
            adversarial_input=x_adv,
            clean_prediction=clean_prediction.astype(np.float32),
            adversarial_prediction=adversarial_prediction.astype(np.float32),
            target_sequence=target.astype(np.float32) if target is not None else None,
            clean_objective_value=clean_objective_value,
            adversarial_objective_value=adversarial_objective_value,
            query_count=query_count,
            loss_history=history,
            epsilon_budget=epsilon,
            perturbation_l1=float(np.linalg.norm(perturbation.reshape(-1), ord=1)),
            perturbation_l2=float(np.linalg.norm(perturbation.reshape(-1), ord=2)),
            perturbation_linf=float(np.linalg.norm(perturbation.reshape(-1), ord=np.inf)),
        )

    def _resolve_epsilon_budget(self, clean_input: np.ndarray) -> float:
        if self.config.epsilon_relative_to == "mean_abs_input":
            base = float(np.mean(np.abs(clean_input)))
            return max(self.config.epsilon * max(base, 1e-8), 1e-8)
        return self.config.epsilon


def _project_delta(delta: np.ndarray, epsilon: float, norm: str) -> np.ndarray:
    flat = delta.reshape(-1)
    if norm == "linf":
        clipped = np.clip(flat, -epsilon, epsilon)
        return clipped.reshape(delta.shape)
    if norm == "l2":
        l2 = np.linalg.norm(flat, ord=2)
        if l2 <= epsilon:
            return delta
        return (flat * (epsilon / max(l2, 1e-8))).reshape(delta.shape)
    if norm == "l1":
        return _project_onto_l1_ball(flat, epsilon).reshape(delta.shape)
    raise ValueError(f"Unsupported norm: {norm}")


def _project_onto_l1_ball(vector: np.ndarray, epsilon: float) -> np.ndarray:
    if epsilon <= 0:
        return np.zeros_like(vector)
    abs_vector = np.abs(vector)
    if abs_vector.sum() <= epsilon:
        return vector
    sorted_abs = np.sort(abs_vector)[::-1]
    cssv = np.cumsum(sorted_abs)
    rho = np.nonzero(sorted_abs * np.arange(1, len(sorted_abs) + 1) > (cssv - epsilon))[0][-1]
    theta = (cssv[rho] - epsilon) / (rho + 1.0)
    return np.sign(vector) * np.maximum(abs_vector - theta, 0.0)
