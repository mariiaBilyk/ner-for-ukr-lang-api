from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.ner_router import router
from infrastructure.inference.factory import InferenceFactory
from infrastructure.config import get_settings
from infrastructure.logging_config import configure_logging

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    InferenceFactory.initialize(get_settings())
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)
