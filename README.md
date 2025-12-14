# bitrl-envs-api

API for reinforcement learning environments. Each environment exposes a Gymnasium like interface.
The API is based on <a href="https://fastapi.tiangolo.com/">FastAPI</a>. You can launch a server using the ```entrypoint.sh``` script.
Alternatively, you can use Docker. You can find examples with C++ at: <a href="https://github.com/pockerman/bitrl">bitrl</a>

The available endpoints are described below.

## Environment API

### Query liveness

```commandline
GET /{idx}/is-alive
```

**Description:** Returns true if the environment with the given id is alive on the server. The response format is

```commandline
{"result": is_alive_}
```

### Close the environment

```commandline
POST /{idx}/close
```

**Description:** Closes the environment with the given id. The response format on successful op is:

```commandline
{"message": "OK"}
```

returns 

```commandline
{"message": "FAILED"}
```

if the the op was not successful.

### Create a new environment

```commandline
POST /make
```

**Description:** Creates a new environment. Response format:

```commandline
{"message": "OK", "idx": "idx"}
```
#### Payload:
```commandline
{
 "version": "version",
 "options": json
}
```

### Reset the environment

```commandline
POST /{idx}/reset
```

**Description:** Reset the environment with the given id. Response format:

```commandline
{"time_step": step}
```

where _step_ has the following structure:

```commandline
class TimeStep(BaseModel, Generic[_Reward, _Discount, _Observation]):
    step_type: TimeStepType = Field(title="step_type")
    reward: Optional[_Reward] = Field(title="reward")
    discount: Optional[_Discount] = Field(title="discount")
    observation: _Observation = Field(title="observation")
    info: dict = Field(title="info")

    def first(self) -> bool:
        return self.step_type == TimeStepType.FIRST

    def mid(self) -> bool:
        return self.step_type == TimeStepType.MID

    def last(self) -> bool:
        return self.step_type == TimeStepType.LAST

    @property
    def done(self) -> bool:
        return self.last()
```
#### Payload:
```commandline
{
seed: int = 42,
options: dict[str, Any] = {}
}
```

### Step in the environment

```commandline
POST /{idx}/step
```

**Description:** Step in the environment with the given id. Response format:

```commandline
{"time_step": step}
```

#### Payload:

```commandline
 action: Any
```

where _action_ is the admissible action to be executed on the environment.

### Query number of copies

```commandline
GET /copies
```

**Description:** Query the number of copies available for a specific environment. Response format:

```commandline
{"copies": len(manager)}
```

## Tensorboard API

There is also a limited API for Tensorboard:

```commandline
POST /init
POST /close
POST /add-text
POST /add-scalar
POST /add-scalars
```


## Installation

You can use Docker to run the API:

```commandline
docker build -t bitrl-rest-api:v1 .
docker run -p 8001:8001 bitrl-rest-api:v1
```

You can access the API documentation at http://0.0.0.0:8001/docs

There is also a pre-built Docker image at: https://hub.docker.com/repository/docker/alexgiavaras/bitrl-rest-api/general






