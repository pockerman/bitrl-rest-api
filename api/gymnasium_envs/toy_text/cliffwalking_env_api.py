import sys

from typing import Annotated
from fastapi import APIRouter, status, Depends, Query
from fastapi.responses import JSONResponse
from fastapi import HTTPException
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

cliff_walking_router = APIRouter(prefix="/gymnasium/cliff-walking-env",
                                 tags=["cliff-walking-env"])

ENV_NAME = "CliffWalking"

# the manager for the environments to create
manager = GymEnvManager(verbose=True)

DEFAULT_OPTIONS = {"max_episode_steps": 500, }
DEFAULT_VERSION = "v1"


@cliff_walking_router.get("/copies")
async def get_n_copies() -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"copies": len(manager)})


@cliff_walking_router.get("/{idx}/is-alive")
async def get_is_alive(idx: str) -> JSONResponse:
    is_alive_ = manager.is_alive(idx=idx)

    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"result": is_alive_})


@cliff_walking_router.post("/{idx}/close",
                           status_code=status.HTTP_202_ACCEPTED)
async def close(idx: str) -> JSONResponse:
    closed = await manager.close(idx=idx)

    if closed:
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                            content={"message": "OK"})

    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                        content={"message": "FAILED"})


@cliff_walking_router.post("/make",
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


@cliff_walking_router.post("/{idx}/reset",
                           response_model=TimeStepResponse,
                           status_code=status.HTTP_202_ACCEPTED)
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


@cliff_walking_router.post("/{idx}/step",
                           response_model=TimeStepResponse,
                           status_code=status.HTTP_202_ACCEPTED)
async def step(idx: str,
               action: DiscreteAction,
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

    step = TimeStep(observation=step_result.observation,
                    reward=step_result.reward,
                    step_type=step_type,
                    info={'prob': float(info['prob'])},
                    discount=1.0)

    if api_config.LOG_INFO:
        logger.info(f'Step in environment {ENV_NAME} and index {idx}')
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                        content={"time_step": step.model_dump()})


@cliff_walking_router.get("/{idx}/dynamics", response_model=GetEnvDynmicsResponseModel)
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

        state_dyns = [int(item) for item in state_dyns]
        return JSONResponse(status_code=status.HTTP_200_OK,
                            content={"dynamics": state_dyns})

    else:
        dynamics = env.unwrapped.P[dyn_req.state_id][dyn_req.action_id]
        dynamics = [float(item[0]) for item in dynamics]

        if api_config.LOG_INFO:
            logger.info(f'Get dynamics for state={dyn_req.state_id}/action={dyn_req.action_id}')
        return JSONResponse(status_code=status.HTTP_200_OK,
                            content={"dynamics": dynamics})

