"""
PPO-based RL agent for the Inventory Reasoning Environment.

Usage:
    # Train
    python agent/rl_agent.py --train --env-type 0 --timesteps 365000

    # Evaluate
    python agent/rl_agent.py --eval --model-path ppo_inventory
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import os
from typing import Any

import httpx
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from client.inventory_client import InventoryAction, InventoryEnvClient, InventoryObservation

# ---------------------------------------------------------------------------
# Observation layout (22 floats):
#   [0]      day
#   [1]      current_inventory
#   [2..6]   demand_last_5  (zero-padded to 5 values)
#   [7]      demand_mean_30d
#   [8]      demand_std_30d
#   [9]      fill_rate_so_far
#   [10]     recent_stockouts
#   [11]     recent_lost_sales
#   [12..21] pending_orders (5 slots × [arrival_day, quantity], zero-padded)
# ---------------------------------------------------------------------------
OBS_DIM = 22
MAX_PENDING_SLOTS = 5


class InventoryGymEnv(gym.Env):
    """Gymnasium wrapper around the HTTP Inventory Reasoning Environment."""

    metadata = {"render_modes": []}

    def __init__(self, base_url: str = "http://localhost:7860", env_type: int = 0) -> None:
        super().__init__()
        self._base_url = base_url
        self._env_type = env_type

        self._loop = asyncio.new_event_loop()
        self._http_client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
        self._inv_client = InventoryEnvClient(base_url)
        self._inv_client._client = self._http_client

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([5000.0], dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        obs = self._loop.run_until_complete(self._inv_client.reset(env_type=self._env_type))
        return self._obs_to_array(obs), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        result = self._loop.run_until_complete(
            self._inv_client.step(InventoryAction(reorder_point=float(action[0])))
        )
        return (
            self._obs_to_array(result.observation),
            float(result.reward),
            result.done,
            False,
            result.info,
        )

    def close(self) -> None:
        self._loop.run_until_complete(self._http_client.aclose())
        self._loop.close()

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _obs_to_array(self, obs: InventoryObservation) -> np.ndarray:
        demand_last_5 = list(obs.demand_last_5)
        demand_last_5 = (demand_last_5 + [0.0] * 5)[:5]

        pending_flat: list[float] = []
        for slot in range(MAX_PENDING_SLOTS):
            if slot < len(obs.pending_orders):
                o = obs.pending_orders[slot]
                pending_flat.extend([float(o.arrival_day), float(o.quantity)])
            else:
                pending_flat.extend([0.0, 0.0])

        vector = (
            [float(obs.day), float(obs.current_inventory)]
            + demand_last_5
            + [obs.demand_mean_30d, obs.demand_std_30d, obs.fill_rate_so_far]
            + [float(obs.recent_stockouts), float(obs.recent_lost_sales)]
            + pending_flat
        )
        assert len(vector) == OBS_DIM, f"Expected {OBS_DIM} elements, got {len(vector)}"
        return np.array(vector, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    base_url: str,
    env_type: int,
    timesteps: int,
    save_path: str,
) -> None:
    env = InventoryGymEnv(base_url=base_url, env_type=env_type)
    check_env(env)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=365,
        batch_size=73,
        n_epochs=10,
        learning_rate=3e-4,
        verbose=1,
    )
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    env.close()
    print(f"Training complete. Model saved to '{save_path}'.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    base_url: str,
    env_type: int,
    model_path: str,
    n_episodes: int = 5,
) -> None:
    model = PPO.load(model_path)
    env = InventoryGymEnv(base_url=base_url, env_type=env_type)

    fill_rates: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        info: dict = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
        fill_rate = float(info.get("fill_rate", 0.0))
        fill_rates.append(fill_rate)
        print(f"Episode {ep + 1}: fill_rate={fill_rate:.4f}")

    env.close()
    print(f"Average fill rate over {n_episodes} episodes: {np.mean(fill_rates):.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO Inventory Agent")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--env-type", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Environment type (0=GammaPoisson, 1=HighVariance, 2=Spiking, 3=LowVariance)")
    parser.add_argument("--base-url", type=str, default="http://localhost:7860",
                        help="Base URL of the inventory environment server")
    parser.add_argument("--timesteps", type=int, default=365_000,
                        help="Total training timesteps")
    parser.add_argument("--model-path", type=str, default="ppo_inventory",
                        help="Path to save/load the model (without .zip extension)")
    args = parser.parse_args()

    if not args.train and not args.eval:
        parser.error("Specify at least one of --train or --eval")

    if args.train:
        train(
            base_url=args.base_url,
            env_type=args.env_type,
            timesteps=args.timesteps,
            save_path=args.model_path,
        )

    if args.eval:
        evaluate(
            base_url=args.base_url,
            env_type=args.env_type,
            model_path=args.model_path,
        )


if __name__ == "__main__":
    main()
