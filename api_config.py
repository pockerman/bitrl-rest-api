class Config:
    API_TITLE: str = "bitrl-rest-api"
    DEBUG: bool = True


def get_api_config():
    return Config()