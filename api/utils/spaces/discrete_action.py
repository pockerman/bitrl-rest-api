from pydantic import BaseModel


class DiscreteAction(BaseModel):
    action: int