import sys

from typing import Annotated
from fastapi import APIRouter, status, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from loguru import logger

from api.utils.get_env_dynamics_request_model import GetEnvDynmicsRequestModel
from api.utils.get_env_dynamics_response_model import GetEnvDynmicsResponseModel
from api.utils.make_env_request_model import MakeEnvRequestModel
from api.utils.time_step_response import TimeStep, TimeStepType, TimeStepResponse
from api.utils.gym_env_manager import GymEnvManager
from api.utils.spaces.discrete_action import DiscreteAction
from api.utils.reset_request_model import RestEnvRequestModel
from api.utils.make_env_response_model import MakeEnvResponseModel
from api.api_config import get_api_config, Config

taxi_router = APIRouter(prefix="/gymnasium/taxi-env", tags=["taxi-env"])

ENV_NAME = "Taxi"

# the manager for the environments to create
manager = GymEnvManager(verbose=True)

DEFAULT_OPTIONS = {"max_episode_steps": 500, }
DEFAULT_VERSION = "v3"


@taxi_router.get("/copies")
async def get_n_copies() -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"copies": len(manager)})


@taxi_router.get("/{idx}/is-alive")
async def get_is_alive(idx: str) -> JSONResponse:
    is_alive_ = manager.is_alive(idx=idx)

    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"result": is_alive_})


@taxi_router.post("/{idx}/close", status_code=status.HTTP_202_ACCEPTED)
async def close(idx: str) -> JSONResponse:
    closed = await manager.close(idx=idx)

    if closed:
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                            content={"message": "OK"})

    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                        content={"message": "FAILED"})


@taxi_router.post("/make",
                  status_code=status.HTTP_201_CREATED,
                  response_model=MakeEnvResponseModel)
async def make(request: MakeEnvRequestModel,
               api_config: Annotated[Config, Depends(get_api_config)]
               ) -> JSONResponse:
    version = request.version or DEFAULT_VERSION

    # merge defaults with user overrides
    options = DEFAULT_OPTIONS | (request.options or {})

    env_type = f"{ENV_NAME}-{version}"

    if api_config.LOG_INFO:
        logger.info(f'Creating environment  {env_type}')

    max_episode_steps = options.get("max_episode_steps", 500)

    idx = await manager.make(env_name=env_type,
                             max_episode_steps=max_episode_steps)

    if api_config.LOG_INFO:
        logger.info(f'Created environment  {ENV_NAME} and index {idx}')

    return JSONResponse(status_code=status.HTTP_201_CREATED,
                        content={"message": "OK", "idx": idx})


@taxi_router.post("/{idx}/reset",
                  status_code=status.HTTP_202_ACCEPTED,
                  response_model=TimeStepResponse)
async def reset(idx: str, reset_ops: RestEnvRequestModel,
                api_config: Annotated[Config, Depends(get_api_config)], ) -> JSONResponse:
    """Reset the environment

    :return:
    """

    if idx not in manager:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"message": "NOT_ALIVE/NOT_CREATED"})

    try:
        reset_step = await manager.reset(idx=idx, seed=reset_ops.seed)

        action_mask = reset_step.info['action_mask']
        observation = reset_step.observation
        step_ = TimeStep(observation=observation,
                         reward=0.0,
                         step_type=TimeStepType.FIRST,
                         info={'action_mask': [int(i) for i in action_mask],
                               'prob': float(reset_step.info['prob'])},
                         discount=1.0)

        if api_config.LOG_INFO:
            logger.info(f"Reset environment {ENV_NAME} and index {idx}")

        return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                            content={"time_step": step_.model_dump()})
    except Exception as e:
        exception = sys.exc_info()
        logger.opt(exception=exception).info("Logging exception traceback")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"message": f"Environment {ENV_NAME} is not initialized."
                                               " Have you called make()?"})


@taxi_router.post("/{idx}/step",
                  status_code=status.HTTP_202_ACCEPTED,
                  response_model=TimeStepResponse)
async def step(idx: str, action: DiscreteAction,
               api_config: Annotated[Config, Depends(get_api_config)]) -> JSONResponse:
    if idx not in manager:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"message": "NOT_ALIVE/NOT_CREATED. Call make/reset"})

    step_result = await manager.step(idx=idx, action=action.action)

    step_type = TimeStepType.MID
    if step_result.terminated:
        step_type = TimeStepType.LAST

    info = step_result.info
    if info is not None:
        info['truncated'] = step_result.truncated

    action_mask = info['action_mask']
    step_ = TimeStep(observation=step_result.observation,
                     reward=step_result.reward,
                     step_type=step_type,
                     info={'action_mask': [int(i) for i in action_mask], 'prob': float(info['prob'])},
                     discount=1.0)

    if api_config.LOG_INFO:
        logger.info(f'Step in environment {ENV_NAME} and index {idx}')

    return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                        content={"time_step": step_.model_dump()})


@taxi_router.get("/{idx}/dynamics", response_model=GetEnvDynmicsResponseModel)
async def get_dynamics(idx: str, dyn_req: Annotated[GetEnvDynmicsRequestModel, Query()],
                       api_config: Annotated[Config, Depends(get_api_config)]) -> JSONResponse:
    if idx not in manager:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"message": "NOT_ALIVE/NOT_CREATED. Call make/reset"})

    env = manager.envs[idx]
    if dyn_req.action_id is None or dyn_req.action_id < 0:
        state_dyns = env.unwrapped.P[dyn_req.state_id]

        if api_config.LOG_INFO:
            logger.info(f'Get dynamics for state={dyn_req.state_id}')

        return JSONResponse(status_code=status.HTTP_200_OK,
                            content={"dynamics": state_dyns})

    else:
        dynamics = env.unwrapped.P[dyn_req.state_id][dyn_req.action_id]

        if api_config.LOG_INFO:
            logger.info(f'Get dynamics for state={dyn_req.state_id}/action={dyn_req.action_id}')
        return JSONResponse(status_code=status.HTTP_200_OK,
                            content={"dynamics": dynamics})
