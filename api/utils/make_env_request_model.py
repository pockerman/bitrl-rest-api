from pydantic import BaseModel
from typing import Optional, Any


class MakeEnvRequestModel(BaseModel):
    version: Optional[str] = None
    options: Optional[dict[str, Any]] = None
