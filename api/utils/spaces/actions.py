from pydantic import BaseModel


class DiscreteAction(BaseModel):
    action: int


class ContinuousAction(BaseModel):
    action: float
