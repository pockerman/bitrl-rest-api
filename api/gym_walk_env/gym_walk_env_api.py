import sys
from loguru import logger
from typing import Annotated
from fastapi import APIRouter, status, Depends, Query
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from api.utils.time_step_response import TimeStep, TimeStepType, TimeStepResponse

from api.utils.get_env_dynamics_request_model import GetEnvDynmicsRequestModel
from api.utils.get_env_dynamics_response_model import GetEnvDynmicsResponseModel
from api.utils.make_env_request_model import MakeEnvRequestModel
from api.utils.make_env_response_model import MakeEnvResponseModel
from api.utils.reset_request_model import RestEnvRequestModel
from api.utils.spaces.actions import DiscreteAction

from api.api_config import get_api_config, Config
from .gym_walk_manager import GymWalkEnvManager

gym_walk_env_router = APIRouter(prefix="/gdrl/gym-walk-env", tags=["gym-walk-env"])

ENV_NAME = "GymWalkEnv"

# actions that the environment accepts
ACTIONS_SPACE = {0: "WEST", 1: "EAST"}

DEFAULT_OPTIONS = {"n_states": 7, "p_stay": 0.0, "p_backward": 0.5}
DEFAULT_VERSION = "v1"

manager = GymWalkEnvManager(verbose=True)


@gym_walk_env_router.get("/copies")
async def get_n_copies() -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"copies": len(manager)})


@gym_walk_env_router.get("/{idx}/is-alive")
async def get_is_alive(idx: str) -> JSONResponse:
    is_alive_ = manager.is_alive(idx=idx)

    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"result": is_alive_})


@gym_walk_env_router.post("/{idx}/close")
async def close(idx: str) -> JSONResponse:
    closed = await manager.close(idx=idx)

    if closed:
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                            content={"message": "OK"})

    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                        content={"message": "FAILED"})


@gym_walk_env_router.post("/make", status_code=status.HTTP_201_CREATED,
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

    idx = await manager.make(env_name=env_type, **options)

    if api_config.LOG_INFO:
        logger.info(f'Created environment  {ENV_NAME} and index {idx}')
    return JSONResponse(status_code=status.HTTP_201_CREATED,
                        content={"message": "OK", "idx": idx})


@gym_walk_env_router.post("/{idx}/reset", status_code=status.HTTP_202_ACCEPTED,
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

    # global envs
    # if cidx in envs:
    #     env = envs[cidx]
    #
    #     if env is not None:
    #         observation, info = envs[cidx].reset(seed=seed)
    #
    #         step = TimeStep(observation=observation,
    #                         reward=0.0,
    #                         step_type=TimeStepType.FIRST,
    #                         info=info,
    #                         discount=1.0)
    #         logger.info(f'Reset environment {ENV_NAME}  and index {cidx}')
    #         return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
    #                             content={"time_step": step.model_dump()})
    #
    # raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
    #                     detail={"message": f"Environment {ENV_NAME} is not initialized."
    #                                        " Have you called make()?"})


@gym_walk_env_router.post("/{idx}/step", status_code=status.HTTP_202_ACCEPTED,
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
    step_ = TimeStep(observation=observation,
                    reward=step_result.reward,
                    step_type=step_type,
                    info=step_result.info,
                    discount=1.0)

    if api_config.LOG_INFO:
        logger.info(f'Step in environment {ENV_NAME} and index {idx}')
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
                        content={"time_step": step_.model_dump()})

    # global envs
    # if cidx in envs:
    #     env = envs[cidx]
    #
    #     if env is not None:
    #         observation, reward, terminated, truncated, info = envs[cidx].step(action)
    #
    #         step_type = TimeStepType.MID
    #         if terminated or truncated:
    #             step_type = TimeStepType.LAST
    #
    #         step = TimeStep(observation=observation,
    #                         reward=reward,
    #                         step_type=step_type,
    #                         info=info,
    #                         discount=1.0)
    #         logger.info(f'Step in environment {ENV_NAME} and index {cidx}')
    #         return JSONResponse(status_code=status.HTTP_202_ACCEPTED,
    #                             content={"time_step": step.model_dump()})
    #
    # raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
    #                     detail=f"Environment {ENV_NAME} is not initialized. Have you called make()?")


@gym_walk_env_router.get("/{idx}/dynamics", response_model=GetEnvDynmicsResponseModel)
async def get_dynamics(idx: str, dyn_req: Annotated[GetEnvDynmicsRequestModel, Query()],
                       api_config: Annotated[Config, Depends(get_api_config)], ) -> JSONResponse:
    if idx not in manager:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"message": "NOT_ALIVE/NOT_CREATED. Call make/reset"})

    env = manager.envs[idx]
    if dyn_req.action_id is None or dyn_req.action_id < 0:
        state_dyns = env.P[dyn_req.state_id]

        if api_config.LOG_INFO:
            logger.info(f'Get dynamics for state={dyn_req.state_id}')

        return JSONResponse(status_code=status.HTTP_200_OK,
                            content={"dynamics": state_dyns})

    else:
        dynamics = env.P[dyn_req.state_id][dyn_req.action_id]

        if api_config.LOG_INFO:
            logger.info(f'Get dynamics for state={dyn_req.state_id}/action={dyn_req.action_id}')
        return JSONResponse(status_code=status.HTTP_200_OK,
                            content={"dynamics": dynamics})

    # global envs
    #
    # env = None
    # if cidx in envs:
    #     env = envs[cidx]
    #
    # if env is None:
    #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
    #                         detail=f"Environment {ENV_NAME} does not exposes dynamics.")
    #
    # if state >= env.nS:
    #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
    #                         detail=f"Action {state} should be in [0, {env.nS})")
    #
    # if action is not None:
    #
    #     if action not in ACTIONS_SPACE:
    #         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
    #                             detail=f"Action {action} not in {list(ACTIONS_SPACE.keys())}")
    #
    #     p = env.P[state][action]
    #     return JSONResponse(status_code=status.HTTP_200_OK,
    #                         content={"p": p})
    #
    # p = env.P[state]
    # return JSONResponse(status_code=status.HTTP_200_OK,
    #                     content={"p": p})
