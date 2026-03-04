from pydantic import BaseModel

class NerRequest(BaseModel):
    text: str