"""API for the Pendulum environment.
For general information see: https://gymnasium.farama.org/environments/classic_control/pendulum/

The action is a ndarray with shape (1,) representing the torque applied to free end of the pendulum.

"""
from loguru import logger
from typing import Any, Annotated
import sys
from fastapi import APIRouter, Body, status, Depends
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from api.utils.make_env_request_model import MakeEnvRequestModel
from api.utils.make_env_response_model import MakeEnvResponseModel
from api.utils.reset_request_model import RestEnvRequestModel
from api.utils.spaces.actions import ContinuousAction
from api.utils.time_step_response import TimeStep, TimeStepType, TimeStepResponse
from api.gymnasium_envs.gym_env_manager import GymEnvManager
from api.api_config import get_api_config, Config

pendulum_router = APIRouter(prefix="/gymnasium/pendulum-env",
                            tags=["pendulum-env"])

ENV_NAME = "Pendulum"

# the manager for the environments to create
manager = GymEnvManager(verbose=True)

# actions that the environment accepts
ACTIONS_SPACE = [1, ]

DEFAULT_OPTIONS = {"g": 10.0, "max_episode_steps": 200}
DEFAULT_VERSION = "v1"


@pendulum_router.get("/copies")
async def get_n_copies() -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"copies": len(manager)})


@pendulum_router.get("/action-space")
async def get_action_space() -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"action_space": ACTIONS_SPACE})


@pendulum_router.get("/{idx}/is-alive")
async def get_is_alive(idx: str) -> JSONResponse:
    is_alive_ = manager.is_alive(idx=idx)
    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"result": is_alive_})


@pendulum_router.post("/{idx}/close")
async def close(idx: str) -> JSONResponse:
    closed = await manager.close(idx=idx)

    if closed:
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                            content={"message": "OK"})

    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                        content={"message": "FAILED"})


@pendulum_router.post("/make", status_code=status.HTTP_201_CREATED,
                      response_model=MakeEnvResponseModel)
async def make(request: MakeEnvRequestModel,
               api_config: Annotated[Config, Depends(get_api_config)]) -> JSONResponse:
    version = request.version or DEFAULT_VERSION

    # merge defaults with user overrides
    options = DEFAULT_OPTIONS | (request.options or {})

    env_type = f"{ENV_NAME}-{version}"

    g = options.get("g", 10.0)
    max_episode_steps = options.get("max_episode_steps", 200)

    if api_config.LOG_INFO:
        logger.info(f'Creating environment  {env_type}')

    idx = await manager.make(env_name=env_type,
                             max_episode_steps=max_episode_steps,
                             g=g)

    if api_config.LOG_INFO:
        logger.info(f'Created environment  {ENV_NAME} and index {idx}')
    return JSONResponse(status_code=status.HTTP_201_CREATED,
                        content={"message": "OK", "idx": idx})


@pendulum_router.post("/{idx}/reset", status_code=status.HTTP_202_ACCEPTED,
                      response_model=TimeStepResponse)
async def reset(idx: str,
                reset_ops: RestEnvRequestModel) -> JSONResponse:
    """Reset the environment

    :return:
    """

    if idx not in manager:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"message": "NOT_ALIVE/NOT_CREATED"})

    try:
        reset_step = await manager.reset(idx=idx, seed=reset_ops.seed)

        observation = reset_step.observation
        observation = [float(val) for val in observation]
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


@pendulum_router.post("/{idx}/step", status_code=status.HTTP_202_ACCEPTED,
                      response_model=TimeStepResponse)
async def step(idx: str, action: ContinuousAction,
                api_config: Annotated[Config, Depends(get_api_config)]
               ) -> JSONResponse:
    if idx not in manager:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"message": "NOT_ALIVE/NOT_CREATED. Call make/reset"})

    if not (-2.0 <= action.action <= 2.0):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Action {action} not in [-2.0, 2.0] range")

    step_result = await manager.step(idx=idx, action=[action.action])

    step_type = TimeStepType.MID
    if step_result.terminated:
        step_type = TimeStepType.LAST

    info = step_result.info
    if info is not None:
        info['truncated'] = step_result.truncated

    observation = step_result.observation
    observation = [float(val) for val in observation]
    step_ = TimeStep(observation=observation,
                     reward=step_result.reward,
                     step_type=step_type,
                     info=step_result.info,
                     discount=1.0)

    if api_config.LOG_INFO:
        logger.info(f'Step in environment {ENV_NAME} and index {idx}')
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                        content={"time_step": step_.model_dump()})
