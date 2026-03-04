import json

def decode_unicode_escape(s: str):
    try:
        return json.loads(s) if s.startswith('"') else s.encode().decode('unicode_escape')
    except Exception:
        return s

def decode_entities(entities_str: str):
    decoded = decode_unicode_escape(entities_str)
    if isinstance(decoded, str):
        return json.loads(decoded)
    return decoded