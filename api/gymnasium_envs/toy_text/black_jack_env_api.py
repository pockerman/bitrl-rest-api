import sys
from typing import Annotated
from fastapi import APIRouter, status, Depends
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from loguru import logger

from api.utils.make_env_request_model import MakeEnvRequestModel
from api.utils.make_env_response_model import MakeEnvResponseModel
from api.utils.time_step_response import TimeStep, TimeStepType, TimeStepResponse
from api.utils.gym_env_manager import GymEnvManager
from api.utils.reset_request_model import RestEnvRequestModel
from api.utils.spaces.actions import DiscreteAction
from api.api_config import get_api_config, Config

black_jack_router = APIRouter(prefix="/gymnasium/black-jack-env", tags=["black-jack-env"])

ENV_NAME = "Blackjack"

# the manager for the environments to create
manager = GymEnvManager(verbose=True)

# actions that the environment accepts
ACTIONS_SPACE = {0: "STICK", 1: "HIT"}

DEFAULT_OPTIONS = {"natural": False, "sab": False}
DEFAULT_VERSION = "v1"


@black_jack_router.get("/copies")
async def get_n_copies() -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"copies": len(manager)})


@black_jack_router.get("/{idx}/is-alive")
async def get_is_alive(idx: str) -> JSONResponse:
    is_alive_ = manager.is_alive(idx=idx)
    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"result": is_alive_})


@black_jack_router.post("/{idx}/close")
async def close(idx: str) -> JSONResponse:
    closed = await manager.close(idx=idx)

    if closed:
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                            content={"message": "OK"})

    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                        content={"message": "FAILED"})


@black_jack_router.post("/make",
                        status_code=status.HTTP_201_CREATED,
                        response_model=MakeEnvResponseModel)
async def make(request: MakeEnvRequestModel,
               api_config: Annotated[Config, Depends(get_api_config)]) -> JSONResponse:
    version = request.version or DEFAULT_VERSION

    # merge defaults with user overrides
    options = DEFAULT_OPTIONS | (request.options or {})
    env_type = f"{ENV_NAME}-{version}"
    if api_config.LOG_INFO:
        logger.info(f'Creating environment  {env_type}')

    natural = options.get("natural", False)
    sab = options.get("sab", False)
    idx = await manager.make(env_name=env_type,
                             natural=natural, sab=sab)

    if api_config.LOG_INFO:
        logger.info(f'Created environment  {ENV_NAME} and index {idx}')
    return JSONResponse(status_code=status.HTTP_201_CREATED,
                        content={"message": "OK", "idx": idx})


@black_jack_router.post("/{idx}/reset",
                        status_code=status.HTTP_202_ACCEPTED,
                        response_model=TimeStepResponse)
async def reset(idx: str, reset_ops: RestEnvRequestModel) -> JSONResponse:
    """Reset the environment

    :return:
    """

    if idx not in manager:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"message": "NOT_ALIVE/NOT_CREATED"})

    try:
        reset_step = await manager.reset(idx=idx, seed=reset_ops.seed)

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


@black_jack_router.post("/{idx}/step",
                        status_code=status.HTTP_202_ACCEPTED,
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

    step_ = TimeStep(observation=step_result.observation,
                     reward=step_result.reward,
                     step_type=step_type,
                     info=info,
                     discount=1.0)

    if api_config.LOG_INFO:
        logger.info(f'Step in environment {ENV_NAME} and index {idx}')
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                        content={"time_step": step_.model_dump()})
