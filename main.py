import uvicorn
from bitrlapp import app

app = app

if __name__ == "__main__":
    print("Starting uvicorn...")
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8001,
        log_level="Debug",
    )



