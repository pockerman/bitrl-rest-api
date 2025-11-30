# bitrl-envs-api

API for reinforcement learning environments. Each environment exposes a Gymnasium like interface.
Specifically:

```commandline
GET  /{idx}/is-alive
POST /{idx}/close
POST /make
POST /{idx}/reset
POST /{idx}/step
```

There is also a limited API for Tensorboard:

```commandline
POST /init
POST /close
POST /add-text
POST /add-scalar
POST /add-scalars
```


The API is based on <a href="https://fastapi.tiangolo.com/">FastAPI</a>. You can launch a server using the ```entrypoint.sh``` script.
Alternatively, you can use Docker. You can find examples with C++ at: <a href="https://github.com/pockerman/bitrl">bitrl</a>
