from pydantic import BaseModel


class Config(BaseModel):
    API_TITLE: str = "bitrl-rest-api"
    DEBUG: bool = True
    LOG_INFO: bool = True


def get_api_config() -> Config:
    return Config()
