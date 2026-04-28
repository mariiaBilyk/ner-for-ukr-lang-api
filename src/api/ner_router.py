import time
import uuid

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from models.ner_request import NerRequest
from application.ner_service import NerService
from application.dependencies import get_ner_service

router = APIRouter()
logger = structlog.get_logger()


@router.post("/ner")
async def ner(request: Request, ner_request: NerRequest,
              service: NerService = Depends(get_ner_service)):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    entities = await service.generate(ner_request.text)

    label_counts: dict[str, int] = {}
    for e in entities:
        label_counts[e["label"]] = label_counts.get(e["label"], 0) + 1

    m = service.metrics()
    logger.info(
        "ner_request",
        request_id=request_id,
        text_length=len(ner_request.text),
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
        entity_count=len(entities),
        label_counts=label_counts,
        inference_method=m["method"],
        inference_provider=m["provider"],
        inference_model=m["model"],
    )

    return {"entities": entities}


@router.get("/health")
async def health_check(service: NerService = Depends(get_ner_service)):
    result = await service.health()
    code = 200 if result.reachable else 503
    return JSONResponse(content=result.model_dump(), status_code=code)


@router.get("/metrics")
async def metrics(service: NerService = Depends(get_ner_service)):
    return service.metrics()
