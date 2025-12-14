from typing import Any, Optional
from collections import namedtuple
import uuid
import gymnasium as gym
import asyncio
from loguru import logger

GymEnvResetResult = namedtuple(typename='GymEnvResetResult', field_names=['observation',
                                                                          'info'])
GymEnvStepResult = namedtuple(typename="GymEnvStepResult", field_names=["observation",
                                                                        "reward",
                                                                        "terminated",
                                                                        "truncated",
                                                                        "info"])


class GymEnvManager:
    """
        Thread-safe async environment manager for Gymnasium environments.
        Supports multiple independent environments (for distributed RL setups).
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.envs: dict[str, gym.Env] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    def __len__(self) -> int:
        return len(self.envs)

    def __contains__(self, idx: str) -> bool:
        """Allow `if idx in manager:` syntax."""
        return self.is_alive(idx)

    def get_lock(self, idx: str) -> asyncio.Lock:
        if idx not in self.locks:
            raise ValueError(f"idx not in locks")
        return self.locks[idx]

    async def make(self, env_name: str, **kwargs) -> str:

        idx = uuid.uuid4().hex
        self.envs[idx] = gym.make(env_name, **kwargs)
        self.locks[idx] = asyncio.Lock()
        return idx

    async def close(self, idx: str) -> bool:
        async with self.get_lock(idx):
            if idx in self.envs:
                self.envs[idx].close()
                del self.envs[idx]
                return True
            return False

    async def step(self, idx: str, action: Any) -> GymEnvStepResult:
        async with self.get_lock(idx):
            env = self.envs.get(idx)
            if env is None:
                raise ValueError("Env not found.")
            observation, reward, terminated, truncated, info = env.step(action)
            return GymEnvStepResult(observation=observation, reward=reward,
                                    terminated=terminated, truncated=truncated, info=info)

    async def reset(self, idx: str, seed: Optional[int] = None, **kwargs) -> GymEnvResetResult:
        """Reset the environment and return (observation, info)."""
        async with self.get_lock(idx):
            env = self.envs.get(idx)
            if env is None:
                raise ValueError(f"Environment {idx} not found. Have you called make()?")

            obs, info = env.reset(seed=seed, **kwargs)
            logger.info(f"Reset environment {idx}")
            return GymEnvResetResult(observation=obs, info=info)

    def is_alive(self, idx: str) -> bool:
        """Check if an environment exists and is active."""
        return idx in self.envs and self.envs[str] is not None
