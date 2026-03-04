import dspy

class ExtractEntities(dspy.Signature):
    """Extract NER entities from Ukrainian text and return JSON array."""
    text = dspy.InputField()
    entities_json = dspy.OutputField()