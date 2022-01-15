import utils
import uvicorn
from fastapi import FastAPI, Header, HTTPException
import logging
from tflitex import predict
from pydantic import BaseModel
import cvutils
from starlette.responses import StreamingResponse
import cv2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI()

class InputDoc(BaseModel):
    text  : str

class Item(BaseModel):
    url  : str
    effect : str

class Item_style(BaseModel):
    url  : str
    url2 : str
    ratio : float

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
    img = cvutils.GnP(url)
    imgout = cvutils.effectsdoer(img, effect)
    imgout = cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB)
    if imgout is None:
        raise HTTPException(status_code=400, detail="Invalid effect")
    bytes = cvutils.to_bytes(imgout)
    return StreamingResponse(bytes, media_type="image/png")

@app.post("/style")
def style(input : Item):
    url = input.url
    effect = input.effect
    img = cvutils.GnP(url)
    model = cvutils.getmodel(effect)
    if model is None: raise HTTPException(status_code=400, detail="Invalid style")
    imgout = cvutils.style_transfer(img, model)
    imgout = cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB)
    bytes = cvutils.to_bytes(imgout)
    return StreamingResponse(bytes, media_type="image/png")

@app.post("/style_predict")
def style_predict(input : Item_style):
    url = input.url
    url2 = input.url2
    ratio = input.ratio
    if url is None or url2 is None or ratio is None:
        raise HTTPException(status_code=400, detail="Incorrect input")

    img = cvutils.GnP(url)
    style = cvutils.GnP(url2)
    
    imgout = utils.blending(img, style,'models/prediction.tflite', 'models/transfer.tflite', ratio)
    bytes = cvutils.to_bytes(imgout)
    return StreamingResponse(bytes, media_type="image/png")
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)