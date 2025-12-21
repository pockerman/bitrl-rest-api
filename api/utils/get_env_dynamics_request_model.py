from pydantic import BaseModel
from typing import Optional, Any


class GetEnvDynmicsRequestModel(BaseModel):
    state_id: int = None
    action_id: Optional[int] = None