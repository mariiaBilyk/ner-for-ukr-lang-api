from fastapi import APIRouter, Depends
from models.ner_request import NerRequest
from application.ner_service import NerService
from application.dependencies import get_ner_service

router = APIRouter()

@router.post("/ner")
async def ner(ner_request: NerRequest,
              service: NerService = Depends(get_ner_service)):
    text = ner_request.text
    entities = await service.generate(text)
    return {"entities": entities}

@router.get("/health")
async def health_check():
    return "OK"

# @router.get("/metrics")
# async def metrics(service: NerService = Depends(get_ner_service)):
#     return {"model_name": service.model_name}    