from pydantic import BaseModel


class GetEnvDynmicsResponseModel(BaseModel):
    dynamics: list[float] | float
