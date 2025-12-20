from pydantic import BaseModel, Field
from typing import Any, Optional


class RestEnvRequestModel(BaseModel):
    seed: int = Field(default=42, description="The seed to use to resent the environment")
    options: Optional[dict[str, Any]] = Field(default=None,
                                              description="Various options the environment may be using for resetting")
