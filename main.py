import torch

from fastapi import FastAPI
from app.api.predict import predict_router

app = FastAPI()

app.include_router(predict_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)
