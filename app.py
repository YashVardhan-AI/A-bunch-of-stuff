import uvicorn
from fastapi import FastAPI, Header, HTTPException
import logging
from tflitex import predict
from pydantic import BaseModel
from cvutils import *
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()

class InputDoc(BaseModel):
    text  : str

class Item(BaseModel):
    url  : str
    effect : str

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


@app.post("/cv")
def effect(input : Item):
    url = input.url
    effect = input.effect
    img = GnP(url)
    imgout = effectsdoer(img, effect)
    if imgout is None:
        raise HTTPException(status_code=400, detail="Invalid effect")
    bytes = to_bytes(imgout)
    return StreamingResponse(bytes, media_type="image/png")

@app.post("/style")
def style(input : Item):
    url = input.url
    effect = input.effect
    img = GnP(url)
    model = getmodel(effect)
    if model is None: raise HTTPException(status_code=400, detail="Invalid style")
    imgout = style_transfer(img, model)
    bytes = to_bytes(imgout)
    return StreamingResponse(bytes, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)