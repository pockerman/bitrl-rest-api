import sys
from typing import Any
from fastapi import APIRouter, Body, status
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from loguru import logger
from api.utils.time_step_response import TimeStep, TimeStepType
from api.utils.gym_env_manager import GymEnvManager

cliff_walking_router = APIRouter(prefix="/gymnasium/cliff-walking-env", tags=["cliff-walking-env"])

ENV_NAME = "CliffWalking"

# the manager for the environments to create
manager = GymEnvManager(verbose=True)


@cliff_walking_router.get("/{idx}/is-alive")
async def get_is_alive(idx: str) -> JSONResponse:
    is_alive_ = manager.is_alive(idx=idx)

    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"result": is_alive_})


@cliff_walking_router.post("/{idx}/close")
async def close(idx: str) -> JSONResponse:
    closed = await manager.close(idx=idx)

    if closed:
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                            content={"message": "OK"})

    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                        content={"message": "FAILED"})


@cliff_walking_router.post("/make")
async def make(version: str = Body(default="v1"),
               max_episode_steps: int = Body(default=500)) -> JSONResponse:
    env_type = f"{ENV_NAME}-{version}"

    idx = await manager.make(env_name=env_type,
                             max_episode_steps=max_episode_steps)

    logger.info(f'Created environment  {ENV_NAME} and index {idx}')
    return JSONResponse(status_code=status.HTTP_201_CREATED,
                        content={"message": "OK", "idx": idx})


@cliff_walking_router.post("/{idx}/reset")
async def reset(idx: str,
                seed: int = Body(default=42),
                options: dict[str, Any] = Body(default={})) -> JSONResponse:
    """Reset the environment

    :return:
    """
    if idx not in manager:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"message": "NOT_ALIVE/NOT_CREATED"})

    try:
        reset_step = await manager.reset(idx=idx, seed=seed)

        observation = reset_step.observation
        step_ = TimeStep(observation=observation,
                         reward=0.0,
                         step_type=TimeStepType.FIRST,
                         info=reset_step.info,
                         discount=1.0)
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                            content={"time_step": step_.model_dump()})
    except Exception as e:
        exception = sys.exc_info()
        logger.opt(exception=exception).info("Logging exception traceback")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"message": f"Environment {ENV_NAME} is not initialized."
                                               " Have you called make()?"})


@cliff_walking_router.post("/{idx}/step")
async def step(idx: str, action: int = Body(...)) -> JSONResponse:
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

    step = TimeStep(observation=step_result.observation,
                    reward=step_result.reward,
                    step_type=step_type,
                    info={'prob': float(info['prob'])},
                    discount=1.0)

    logger.info(f'Step in environment {ENV_NAME} and index {idx}')
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                        content={"time_step": step.model_dump()})


@cliff_walking_router.get("/{idx}/dynamics")
async def get_dynamics(idx: str, stateId: int, actionId: int = None) -> JSONResponse:
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