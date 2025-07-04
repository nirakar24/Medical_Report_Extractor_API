# Shared parsing and fuzzy-matching utilities for all report types 
import re
from rapidfuzz import fuzz, process

def fuzzy_match(label, aliases, min_score=70):
    best = process.extractOne(label.lower(), aliases, scorer=fuzz.token_sort_ratio)
    if best and best[1] >= min_score:
        return best[0]
    return None

NUMBER_RE = re.compile(r'^\d{1,3}(?:,\d{3})*(?:\.\d+)?$')
RANGE_RE  = re.compile(r'(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)') 