import json


def normalize_entities(raw):
    """
    Convert any of the following formats into a normalized set of tuples:
    (label, text)

    Accepted formats:
    - [{"label": "...", "text": "..."}]
    - JSON string containing such list
    """
    
    # If input is a JSON string
    if isinstance(raw, str):
        raw = raw.strip()
        try:
            raw = json.loads(raw)
        except Exception:
            # If broken, return empty
            return set()

    if not isinstance(raw, list):
        return set()

    norm = set()

    for ent in raw:
        if not isinstance(ent, dict):
            continue
        
        # Text may be unicode-escaped → decode
        text = ent.get("text", "")
        if isinstance(text, str):
            try:
                text = text.encode("utf-8").decode("unicode_escape")
            except Exception:
                pass
        
        # Map 'type' → 'label'
        label = ent.get("label")
        
        if label is None or not text:
            continue
        
        # Normalize format of label ("misc" → "MISC")
        label = label.strip().upper()

        norm.add((label, text))

    return norm

