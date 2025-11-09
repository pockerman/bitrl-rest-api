import gymnasium as gym
import sys
from typing import Any
from fastapi import APIRouter, Body, status, HTTPException
from fastapi.responses import JSONResponse

from loguru import logger
from api.utils.time_step_response import TimeStep, TimeStepType
from api.utils.gym_env_manager import GymEnvManager

taxi_router = APIRouter(prefix="/gymnasium/taxi-env", tags=["taxi-env"])

ENV_NAME = "Taxi"

# the manager for the environments to create
manager = GymEnvManager(verbose=True)


@taxi_router.get("/{idx}/is-alive")
async def get_is_alive(idx: int) -> JSONResponse:
    is_alive = manager.is_alive(idx=idx)

    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"result": is_alive})


@taxi_router.post("/{idx}/close")
async def close(idx: int) -> JSONResponse:
    closed = await manager.close(idx=idx)

    if closed:
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                            content={"message": "OK"})

    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                        content={"message": "FAILED"})


@taxi_router.post("/{idx}/make")
async def make(idx: int,
               version: str = Body(default="v3"),
               options: dict[str, Any] = Body(default={"max_episode_steps": 500})
               ) -> JSONResponse:
    env_type = f"{ENV_NAME}-{version}"

    max_episode_steps = options.get("max_episode_steps", 500)

    await manager.make(idx=idx, env_name=env_type,
                       max_episode_steps=max_episode_steps)

    logger.info(f'Created environment  {ENV_NAME} and index {idx}')
    return JSONResponse(status_code=status.HTTP_201_CREATED,
                        content={"result": True})


@taxi_router.post("/{idx}/reset")
async def reset(idx: int, seed: int = Body(default=42), options: dict[str, Any] = Body(default={})) -> JSONResponse:
    """Reset the environment

    :return:
    """

    if idx not in manager:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"message": "NOT_ALIVE/NOT_CREATED"})

    try:
        reset_step = await manager.reset(idx=idx, seed=seed)

        action_mask = reset_step.info['action_mask']
        observation = reset_step.observation
        step_ = TimeStep(observation=observation,
                         reward=0.0,
                         step_type=TimeStepType.FIRST,
                         info={'action_mask': [int(i) for i in action_mask], 'prob': float(reset_step.info['prob'])},
                         discount=1.0)
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                            content={"time_step": step_.model_dump()})
    except Exception as e:
        exception = sys.exc_info()
        logger.opt(exception=exception).info("Logging exception traceback")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"message": f"Environment {ENV_NAME} is not initialized."
                                               " Have you called make()?"})


@taxi_router.post("/{idx]/step")
async def step(idx: int, action: int = Body(...)) -> JSONResponse:
    if idx not in manager:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"message": "NOT_ALIVE/NOT_CREATED. Call make/reset"})

    step_result = await manager.step(idx=idx, action=action)

    step_type = TimeStepType.MID
    if step_result.terminated:
        step_type = TimeStepType.LAST

    info = step_result.info
    if info is not None:
        info['truncated'] = step_result.truncated

    action_mask = info['action_mask']
    step = TimeStep(observation=step_result.observation,
                    reward=step_result.reward,
                    step_type=step_type,
                    info={'action_mask': [int(i) for i in action_mask], 'prob': float(info['prob'])},
                    discount=1.0)

    logger.info(f'Step in environment {ENV_NAME} and index {idx}')
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                        content={"time_step": step.model_dump()})


@taxi_router.get("/{idx}/dynamics")
async def get_dynamics(idx: int, stateId: int, actionId: int = None) -> JSONResponse:
    if idx not in manager:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"message": "NOT_ALIVE/NOT_CREATED. Call make/reset"})

    env = manager.envs[idx]
    if actionId is None or actionId < 0:
        state_dyns = env.P[stateId]
        return JSONResponse(status_code=status.HTTP_201_CREATED,
                            content={"dynamics": state_dyns})

    else:
        dynamics = env.P[stateId][actionId]
        return JSONResponse(status_code=status.HTTP_200_OK,
                            content={"dynamics": dynamics})


# @taxi_router.post("/sync")
# async def sync(cidx: int = Body(...), options: dict[str, Any] = Body(default={})) -> JSONResponse:
#     return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
#                         content={"message": "OK"})
