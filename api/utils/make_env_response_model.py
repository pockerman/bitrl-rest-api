from pydantic import BaseModel, Field


class MakeEnvResponseModel(BaseModel):
    idx: str = Field(description="The idx of the newly created environment")
    message: str = Field(default="OK", description="Info message")
