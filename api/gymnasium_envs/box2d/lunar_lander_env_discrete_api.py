"""The Acrobot environment from gymnasium:
for more information check: https://gymnasium.farama.org/environments/classic_control/acrobot/

"""
import sys
from typing import Any, Annotated
from loguru import logger
from fastapi import APIRouter, Body, status, Depends
from fastapi.responses import JSONResponse
from fastapi import HTTPException

from api.utils.make_env_request_model import MakeEnvRequestModel
from api.utils.make_env_response_model import MakeEnvResponseModel
from api.utils.reset_request_model import RestEnvRequestModel
from api.utils.spaces.actions import DiscreteAction
from api.utils.time_step_response import TimeStep, TimeStepType, TimeStepResponse
from api.utils.gym_env_manager import GymEnvManager
from api.api_config import get_api_config, Config

lunar_lander_discrete_router = APIRouter(prefix="/gymnasium/lunar-lander-discrete-env",
                                         tags=["Lunar Lander Discrete API"])

ENV_NAME = "LunarLander"

# the manager for the environments to create
manager = GymEnvManager(verbose=True)

# actions that the environment accepts
ACTIONS_SPACE = {0: "do nothing",
                 1: "fire left orientation engine",
                 2: "fire main engine",
                 3: "fire right orientation engine"
                 }

DEFAULT_OPTIONS = {'gravity': -10.0, 'enable_wind': False, 'wind_power': 15.0, 'turbulence_power': 1.5}
DEFAULT_VERSION = "v3"


@lunar_lander_discrete_router.get("/copies")
async def get_n_copies() -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"copies": len(manager)})


@lunar_lander_discrete_router.get("/action-space")
async def get_action_space() -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"action_space": ACTIONS_SPACE})


@lunar_lander_discrete_router.get("/{idx}/is-alive")
async def get_is_alive(idx: str) -> JSONResponse:
    is_alive_ = manager.is_alive(idx=idx)
    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"result": is_alive_})


@lunar_lander_discrete_router.post("/{idx}/close")
async def close(idx: str) -> JSONResponse:
    closed = await manager.close(idx=idx)

    if closed:
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                            content={"message": "OK"})

    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                        content={"message": "FAILED"})


@lunar_lander_discrete_router.post("/make", status_code=status.HTTP_201_CREATED,
                                   response_model=MakeEnvResponseModel)
async def make(request: MakeEnvRequestModel,
               api_config: Annotated[Config, Depends(get_api_config)]) -> JSONResponse:
    version = request.version or DEFAULT_VERSION

    # merge defaults with user overrides
    options = DEFAULT_OPTIONS | (request.options or {})

    if version == 'v1' or version == 'v2':
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail='Environment version v1/v2 for `LunarLander` '
                                   'is deprecated. Please use `LunarLander-v3` instead.')

    env_type = f"{ENV_NAME}-{version}"

    options['continuous'] = False
    idx = await manager.make(env_name=env_type, **options)

    if api_config.LOG_INFO:
        logger.info(f'Created environment  {env_type} with index {idx}')
    return JSONResponse(status_code=status.HTTP_201_CREATED,
                        content={"message": "OK", "idx": idx})


@lunar_lander_discrete_router.post("/{idx}/reset", status_code=status.HTTP_202_ACCEPTED,
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


@lunar_lander_discrete_router.post("/{idx}/step", status_code=status.HTTP_202_ACCEPTED,
                                   response_model=TimeStepResponse)
async def step(idx: str, action: DiscreteAction,
               api_config: Annotated[Config, Depends(get_api_config)]) -> JSONResponse:
    if idx not in manager:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"message": "NOT_ALIVE/NOT_CREATED. Call make/reset"})

    if action.action not in ACTIONS_SPACE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Action {action} not in {list(ACTIONS_SPACE.keys())}")

    step_result = await manager.step(idx=idx, action=action.action)

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
