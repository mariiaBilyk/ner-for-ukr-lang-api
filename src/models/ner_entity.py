from pydantic import BaseModel

class NerEntity(BaseModel):
    text: str
    label: str
