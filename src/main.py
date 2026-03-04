from fastapi import FastAPI
from api.ner_router import router

app = FastAPI()
app.include_router(router)