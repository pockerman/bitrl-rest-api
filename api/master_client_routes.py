from fastapi import APIRouter, Body, status, HTTPException
from fastapi.responses import JSONResponse
import redis
import json

master_user_route = APIRouter(prefix="/master-client")

r = redis.Redis(host="redis", port=6379)


@master_user_route.get("/envs")
async def get_gymnasium_envs() -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_200_OK,
                        content={"envs": ["FrozenLake-4x4", "FrozenLake-8x8", "Taxi-v3",
                                          "CliffWalking-v0", "Pendulum-v0", "Pendulum-v1"]})


@master_user_route.post("/create")
async def make(env_id: str, n_copies: int, options: dict) -> JSONResponse:
    if n_copies > 5:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="n_copies cannot be more than 5")

    for idx in range(n_copies):
        task = {
            "type": "create",
            "env_type": env_id,
            "options": options,
            "copy_idx": idx
        }
        r.rpush("env_tasks", json.dumps(task))
