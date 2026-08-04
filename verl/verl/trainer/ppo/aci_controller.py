from __future__ import annotations

from collections import deque
from typing import Iterable

import torch


class ACIController:
    """Adaptive Conformal Inference controller for DCR-GRPO."""

    def __init__(
        self,
        initial_scores_path: str,
        buffer_size: int,
        alpha_t: float,
        alpha_target: float,
        gamma: float,
        ema_ratio: float,
        ema_err: float,
        penalty_weight: float,
    ) -> None:
        raw_scores = torch.load(initial_scores_path, map_location="cpu")
        if isinstance(raw_scores, dict):
            if "scores" in raw_scores:
                raw_scores = raw_scores["scores"]
            elif "initial_scores" in raw_scores:
                raw_scores = raw_scores["initial_scores"]
            else:
                raise ValueError(
                    f"Unsupported initial score dict keys: {list(raw_scores.keys())}. "
                    "Expected one of: ['scores', 'initial_scores']."
                )

        scores = torch.as_tensor(raw_scores, dtype=torch.float32).reshape(-1)
        if scores.numel() == 0:
            raise ValueError("`initial_scores.pt` is empty; cannot initialize ACIController.")

        self.score_buffer = deque(scores.tolist(), maxlen=buffer_size)
        self.alpha_t = float(alpha_t)
        self.alpha_target = float(alpha_target)
        self.gamma = float(gamma)
        self.ema_ratio = float(ema_ratio)
        self.ema_err = float(ema_err)
        self.penalty_weight = float(penalty_weight)

    def _quantile(self, q: float) -> float:
        buffer_tensor = torch.tensor(list(self.score_buffer), dtype=torch.float32)
        return float(torch.quantile(buffer_tensor, q=q).item())

    def step(self, current_batch_scores: torch.Tensor | Iterable[float]) -> tuple[float, torch.Tensor]:
        scores = torch.as_tensor(current_batch_scores, dtype=torch.float32).reshape(-1).detach().cpu()
        if scores.numel() == 0:
            raise ValueError("current_batch_scores is empty")

        q_t = self._quantile(1.0 - self.alpha_t)
        err_t = float((scores > q_t).float().mean().item())
        self.ema_err = (1.0 - self.ema_ratio) * self.ema_err + self.ema_ratio * err_t
        self.alpha_t = float(min(0.99, max(0.01, self.alpha_t + self.gamma * (self.alpha_target - self.ema_err))))

        for val in scores.tolist():
            self.score_buffer.append(val)

        low_conf_weights = torch.full_like(scores, self.penalty_weight)
        u = torch.where(scores <= q_t, torch.ones_like(scores), low_conf_weights)
        return q_t, u
