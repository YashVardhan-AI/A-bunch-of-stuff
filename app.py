import uvicorn
from fastapi import FastAPI, Header, HTTPException
import logging
from tflitex import predict
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()

class InputDoc(BaseModel):
    text  : str


@app.on_event("startup")
def startup():
    logger.info("Starting...")


@app.on_event("shutdown")
def shutdown():
    logger.info("Shutting down...")


@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/classify")
def detect(item: InputDoc):
    try:
        result = predict(item.text)
        return {"result": result}
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)