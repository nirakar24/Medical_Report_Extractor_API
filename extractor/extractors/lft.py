# LFT-specific extraction logic

from ..ocr_utils import load_document, run_ocr
from ..parsing_utils import NUMBER_RE, RANGE_RE
from collections import defaultdict
from statistics import mean
import re
from rapidfuzz import fuzz

# Expanded and robust alias map
PARAM_ALIASES = {
    "SGOT (AST)": ["sgot", "ast", "sgot (ast)", "serum glutamate oxaloacetate transaminase", "aspartate transaminase"],
    "SGPT (ALT)": ["sgpt", "alt", "sgpt (alt)", "serum glutamate pyruvate transaminase", "alanine transaminase"],
    "ALP": ["alp", "alkaline phosphatase", "alk phosphatase", "alkaline", "phosphatase"],
    "Total Bilirubin": ["total bilirubin", "bilirubin total", "bilirubin t", "t bilirubin", "bilirubin", "total", "bilirubin total", "total bilirubin"],
    "Direct Bilirubin": [
        "direct bilirubin", "bilirubin direct", "d bilirubin", "bilirubin", "direct", "direct bilirubin", "bilirubin direct",
        "serumi bilirubin", "serum direct bilirubin", "serum bilirubin (direct)", "bilirubin (direct)", "(direct)", "serum bilirubin direct"
    ],
    "Indirect Bilirubin": [
        "indirect bilirubin", "bilirubin indirect", "i bilirubin", "bilirubin", "indirect", "indirect bilirubin", "bilirubin indirect",
        "serum bilirubin (indirect)", "bilirubin (indirect)", "(indirect)", "serum bilirubin indirect"
    ],
    "Albumin": ["albumin", "serum albumin"],
    "Globulin": ["globulin", "serum globulin"],
    "A/G Ratio": ["a/g ratio", "ag ratio", "albumin globulin ratio", "a:g ratio", "a:g"],
    "Total Protein": [
        "total protein", "protein total", "serum protein", "total proteins", "proteins", "total", "protein",
        "serum total protein", "serum proteins", "serum total proteins"
    ],
    "GGT": [
        "ggt", "gamma glutamyl transferase", "gamma gt", "gamma glutamyl", "gamma-glutamyl transferase", "gamma-glutamyl", "gamma glutamyl (ggt)", "(ggt)", "ggt (gamma glutamyl)", "gamma glutamyltransferase"
    ],
    "SGOT/SGPT Ratio": ["sgot/sgpt ratio", "sgot/sgpt", "sgot sgot/sgpt ratio", "sgot/sgpt", "ratio"],
}

# Canonical metadata for LFT parameters
PARAMETER_META = {
    "SGOT (AST)": {"unit": "U/L", "range": "5 - 40"},
    "SGPT (ALT)": {"unit": "U/L", "range": "7 - 56"},
    "ALP": {"unit": "U/L", "range": "44 - 147"},
    "Total Bilirubin": {"unit": "mg/dL", "range": "0.1 - 1.2"},
    "Direct Bilirubin": {"unit": "mg/dL", "range": "<0.3"},
    "Indirect Bilirubin": {"unit": "mg/dL", "range": "<1.0"},
    "Albumin": {"unit": "g/dL", "range": "3.5 - 5.0"},
    "Globulin": {"unit": "g/dL", "range": "2.0 - 3.5"},
    "A/G Ratio": {"unit": "-", "range": "1.0 - 2.2"},
    "Total Protein": {"unit": "g/dL", "range": "6.0 - 8.3"},
    "GGT": {"unit": "U/L", "range": "9 - 48"},
    "SGOT/SGPT Ratio": {"unit": "RATIO", "range": "0 - 46"},
}

HEADER_KEYWORDS = {
    "liver", "function", "test", "description", "reference",
    "range", "unit", "result", "protein",
    "enzyme", "remarks", "interpretation", "parameters"
}

VALID_UNITS = {
    "u/l", "mg/dl", "g/dl", "g/l", "%", "", "-", "gm/dl", "iu/l", "ratio"
}

# Robust unit normalization
UNIT_NORMALIZATION = {
    'mgidl': 'mg/dL', 'mgidl': 'mg/dL', 'mg/di': 'mg/dL', 'mgldi': 'mg/dL', 'mg/dl': 'mg/dL',
    'gidi': 'g/dL', 'gidl': 'g/dL', 'g/di': 'g/dL', 'gldi': 'g/dL', 'g/dl': 'g/dL',
    'uai': 'U/L', 'u/i': 'U/L', 'u/l': 'U/L', 'iu/l': 'U/L',
}
def normalize_unit(unit):
    u = unit.lower().replace(' ', '').replace('.', '')
    for k, v in UNIT_NORMALIZATION.items():
        if k in u:
            return v
    if 'mgid' in u:
        return 'mg/dL'
    if 'gid' in u:
        return 'g/dL'
    return unit

def all_aliases():
    return [a for aliases in PARAM_ALIASES.values() for a in aliases]

def extract_lft_from_image(image_path_or_bytes):
    doc = load_document(image_path_or_bytes)
    json_output = run_ocr(doc)
    page_width = 1.0
    if json_output['pages'] and json_output['pages'][0]['dimensions']:
        page_width = json_output['pages'][0]['dimensions'][0]
    y_tolerance = 0.015
    x_indent_threshold_px = 0.08 * page_width
    grouped = defaultdict(list)
    for page in json_output['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    center_y = (word['geometry'][0][1] + word['geometry'][1][1]) / 2
                    center_x = (word['geometry'][0][0] + word['geometry'][1][0]) / 2
                    key = round(center_y / y_tolerance)
                    grouped[key].append({
                        'text': word['value'],
                        'x': center_x * page_width,
                        'y': center_y
                    })
    rows = []
    for _, words in sorted(grouped.items()):
        tokens = [w['text'] for w in words]
        xs = [w['x'] for w in words]
        ys = [w['y'] for w in words]
        rows.append({
            'tokens': tokens,
            'avg_x': mean(xs) if xs else 0,
            'avg_y': mean(ys) if ys else 0,
            'words': words
        })
    for row in rows:
        row['is_header'] = any(any(kw in t.lower() for kw in HEADER_KEYWORDS) for t in row['tokens'])
        row['is_data'] = not row['is_header']
    current_section = 'main'
    for row in rows:
        if row['is_header']:
            current_section = ' '.join(row['tokens'])
        row['section'] = current_section
    def is_orphan_label(row):
        return (
            all(not NUMBER_RE.match(t) for t in row['tokens'])
            and any(t.lower() in {a.lower() for a in all_aliases()} for t in row['tokens'])
        )
    def is_data_only(row):
        return any(NUMBER_RE.match(t) for t in row['tokens']) and all(
            not t.isalpha() for t in row['tokens'] if not NUMBER_RE.match(t)
        )
    final_rows = []
    i = 0
    while i < len(rows):
        cur = rows[i]
        nxt = rows[i+1] if i+1 < len(rows) else None
        if nxt and is_data_only(cur) and is_orphan_label(nxt):
            merged = {
                'section': cur['section'],
                'tokens': cur['tokens'] + nxt['tokens'],
                'avg_x': (cur['avg_x']+nxt['avg_x'])/2,
                'avg_y': (cur['avg_y']+nxt['avg_y'])/2
            }
            final_rows.append(merged)
            i += 2
        else:
            final_rows.append(cur)
            i += 1
    rows = final_rows
    def split_multi_param_row(row, next_row=None):
        tokens = row['tokens']
        results = []
        unit = ''
        def norm(s):
            return ''.join(c for c in s.lower() if c.isalnum())
        # --- Bilirubin parenthetical priority logic ---
        if any('bilirubin' in t.lower() for t in tokens):
            label = ' '.join(tokens).lower()
            value = None
            unit = ''
            rng = ''
            # Try to extract value from this row
            for t in tokens:
                if NUMBER_RE.match(t):
                    value = t
                    break
            # Try to extract range from this row
            rng_match = RANGE_RE.search(' '.join(tokens))
            if rng_match:
                rng = f"{rng_match.group(1)} to {rng_match.group(2)}"
            # Try to extract unit from this row
            for t in tokens:
                if any(u in t.lower() for u in ['mg', 'g', 'iu', 'u/', 'g/']):
                    unit = t
                    break
            # Parenthetical priority
            if '(total)' in label:
                param = 'Total Bilirubin'
            elif '(direct)' in label or 'serumi bilirubin' in label or (next_row and '(direct)' in (next_row['tokens'][0].lower() if next_row['tokens'] else '')):
                param = 'Direct Bilirubin'
            elif '(indirect)' in label:
                param = 'Indirect Bilirubin'
            else:
                param = None
            if param and value:
                meta = PARAMETER_META.get(param, {})
                if not unit:
                    unit = meta.get('unit', '')
                if not rng:
                    rng = meta.get('range', '')
                unit = normalize_unit(unit)
                results.append({'parameter': param, 'value': value, 'unit': unit, 'range': rng})
                return results
        # --- Total Protein special case ---
        if any('protein' in t.lower() for t in tokens):
            label = ' '.join(tokens).lower()
            if 'serum protein' in label or 'total protein' in label:
                value = None
                for t in tokens:
                    if NUMBER_RE.match(t):
                        value = t
                        break
                if not value and next_row:
                    for t in next_row['tokens']:
                        if NUMBER_RE.match(t):
                            value = t
                            break
                unit = ''
                for t in tokens:
                    if any(u in t.lower() for u in ['mg', 'g', 'iu', 'u/', 'g/']):
                        unit = t
                        break
                rng = ''
                rng_match = RANGE_RE.search(' '.join(tokens))
                if rng_match:
                    rng = f"{rng_match.group(1)} to {rng_match.group(2)}"
                param = 'Total Protein'
                meta = PARAMETER_META.get(param, {})
                if not unit:
                    unit = meta.get('unit', '')
                if not rng:
                    rng = meta.get('range', '')
                unit = normalize_unit(unit)
                if value:
                    results.append({'parameter': param, 'value': value, 'unit': unit, 'range': rng})
                    return results
        # --- Fuzzy/alias logic (fallback) ---
        alias_tuples = []
        for name, aliases in PARAM_ALIASES.items():
            for alias in aliases:
                alias_toks = alias.lower().split()
                alias_tuples.append((name, alias, alias_toks))
        alias_tuples.sort(key=lambda x: -len(x[2]))
        i = 0
        while i < len(tokens):
            best_score = 0
            best_name = None
            best_alias = None
            best_start = None
            best_end = None
            for name, alias, alias_toks in alias_tuples:
                n = len(alias_toks)
                if i+n <= len(tokens):
                    token_slice = [t.lower() for t in tokens[i:i+n]]
                    alias_str = ' '.join(alias_toks)
                    token_str = ' '.join(token_slice)
                    score = fuzz.ratio(token_str, alias_str)
                    joined_score = fuzz.ratio(' '.join(tokens[i:i+n]).lower().replace(':',''), alias_str)
                    reversed_token_str = ' '.join(reversed(token_slice))
                    reversed_score = fuzz.ratio(reversed_token_str, alias_str)
                    if joined_score > score:
                        score = joined_score
                    if reversed_score > score:
                        score = reversed_score
                    if score > best_score:
                        best_score = score
                        best_name = name
                        best_alias = alias
                        best_start = i
                        best_end = i+n
            if best_score >= 80 and best_name is not None and best_start is not None and best_end is not None:
                name = best_name
                start = best_start
                end = best_end
                value = None
                unit = ''
                rng = ''
                for j in range(end, min(end+4, len(tokens))):
                    if NUMBER_RE.match(tokens[j]):
                        value = tokens[j]
                        value_idx = j
                        break
                else:
                    for j in range(end+4, len(tokens)):
                        if NUMBER_RE.match(tokens[j]):
                            value = tokens[j]
                            value_idx = j
                            break
                rng_match = RANGE_RE.search(' '.join(tokens[end:end+4]))
                if rng_match:
                    rng = f"{rng_match.group(1)} to {rng_match.group(2)}"
                unit_found = False
                for k in range((value_idx+1) if value else end, min((value_idx+4) if value else end+4, len(tokens))):
                    candidate = ''.join(tokens[k:k+2]).lower().replace('fl', 'fL').replace('µl', 'µL')
                    if candidate in {u.lower().replace(' ', '') for u in VALID_UNITS}:
                        unit = ' '.join(tokens[k:k+2])
                        unit_found = True
                        break
                    candidate = tokens[k].lower().replace('fl', 'fL').replace('µl', 'µL')
                    if candidate in {u.lower().replace(' ', '') for u in VALID_UNITS}:
                        unit = tokens[k]
                        unit_found = True
                        break
                meta = PARAMETER_META.get(name, {})
                if not unit:
                    unit = meta.get('unit', '')
                if not rng:
                    rng = meta.get('range', '')
                unit = normalize_unit(unit)
                if name and value:
                    results.append({'parameter': name, 'value': value, 'unit': unit, 'range': rng})
                i = (value_idx+1) if value else end+1
            else:
                i += 1
        return results
    extracted = []
    for idx, row in enumerate(rows):
        is_protein_row = any('protein' in t.lower() for t in row['tokens'])
        if row['is_data'] or is_protein_row:
            next_row = rows[idx+1] if idx+1 < len(rows) else None
            extracted.extend(split_multi_param_row(row, next_row=next_row))
    for item in extracted:
        meta = PARAMETER_META.get(item['parameter'], {})
        if not item.get('unit'):
            item['unit'] = meta.get('unit', '')
        if not item.get('range'):
            item['range'] = meta.get('range', '')
    # Deduplicate by parameter name, keeping the first occurrence
    seen = set()
    deduped = []
    for item in extracted:
        key = item['parameter'].lower()
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped 