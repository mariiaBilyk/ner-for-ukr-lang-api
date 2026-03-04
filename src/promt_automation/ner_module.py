import dspy
from src.promt_automation.extract_entities import ExtractEntities

class NERModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(ExtractEntities)

    def forward(self, text):
        return self.predictor(text=text)