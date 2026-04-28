from pydantic import BaseModel
from models.ner_label import NerLabel


class NerEntity(BaseModel):
    text: str
    label: NerLabel
