from pydantic import BaseModel, Field
from typing import Any


class InfoResponseModel(BaseModel):
    message: Any