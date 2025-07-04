from typing import Annotated
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from neural_network.predictor import Predictor

server = FastAPI()

origins = ["http://localhost:5173"]


server.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Image(BaseModel):
    pixels: list[list[float]]

# The elements in the `image` must be in the range [0, 1]
@server.post("/predict")
async def predict(image: Image, predictor: Annotated[Predictor, Depends(Predictor)]):
    trasformed_pixels = np.array(image.pixels).flatten()/255
    result = predictor.predict(trasformed_pixels) 
    return {"message": result.tolist()} 