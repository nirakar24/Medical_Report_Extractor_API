import sys
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from rapidfuzz import fuzz, process
import re
import json
from collections import defaultdict
from statistics import mean

# Comprehensive alias map
PARAM_ALIASES = {
    "Haemoglobin":          ["haemoglobin","hemoglobin","hb"],
    "Total Leucocyte Count":["total leucocyte count","wbc","total w.b.c. count","leucocyte count","tlc","total leukocyte count","total leucocytes"],
    "Neutrophils":          ["neutrophils","neutrophil","neut"],
    "Lymphocytes":          ["lymphocytes","lymphocyte","lymph"],
    "Eosinophils":          ["eosinophils","eosinophil","eos"],
    "Monocytes":            ["monocytes","monocyte","mono"],
    "Basophils":            ["basophils","basophil","baso"],
    "Absolute Neutrophils": ["absolute neutrophils","abs neutrophils","abs neut","absolute neut","absolute neutrophil count","abs neutrophil count"],
    "Absolute Lymphocytes": ["absolute lymphocytes","abs lymphocytes","abs lymph","absolute lymph","absolute lymphocyte count","abs lymphocyte count"],
    "Absolute Eosinophils": ["absolute eosinophils","abs eosinophils","abs eos","absolute eos","absolute eosinophil count","abs eosinophil count"],
    "Absolute Monocytes":   ["absolute monocytes","abs monocytes","abs mono","absolute mono","absolute monocyte count","abs monocyte count"],
    "RBC Count":            ["rbc count","total r.b.c. count","rbc","r b c count","r b c"],
    "MCV":                  ["mcv","mean corpuscular volume","m c v"],
    "MCH":                  ["mch","mean corpuscular hemoglobin","m c h"],
    "MCHC":                 ["mchc","mean corpuscular hemoglobin concentration","m c h c"],
    "Hct":                  ["hct","pcv","hematocrit","packed cell volume"],
    "RDW-CV":               ["rdw-cv","rdw cv","rdwcv","red cell distribution width cv"],
    "RDW-SD":               ["rdw-sd","rdw sd","rdwsd","red cell distribution width sd"],
    "Platelet Count":       ["platelet count","platelets","plt count","plt","platelet cnt"],
    "MPV":                  ["mpv","mean platelet volume"],
    "PCT":                  ["pct","plateletcrit"],
    "PDW":                  ["pdw","platelet distribution width"],
    # Section headers as aliases for context
    "Differential Leucocyte Count": ["Differential Leucocyte Count", "Differential", "Differential Count"],
    "Absolute Leucocyte Count": ["Absolute Leucocyte Count", "Absolute", "Absolute Count"],
    "RBC Indices": ["RBC Indices", "RBC Index", "Indices"],
    "Platelets Indices": ["Platelets Indices", "Platelet Indices", "Platelets Index"]
}

# Parameter metadata (add more as needed)
PARAMETER_META = {
    "Haemoglobin": {"unit": "g/dL", "range": "13 - 17"},
    "Total Leucocyte Count": {"unit": "/cumm", "range": "4000 - 10000"},
    "Neutrophils": {"unit": "%", "range": "40 - 80"},
    "Lymphocytes": {"unit": "%", "range": "20 - 40"},
    "Eosinophils": {"unit": "%", "range": "1 - 6"},
    "Monocytes": {"unit": "%", "range": "2 - 10"},
    "Basophils": {"unit": "%", "range": "0 - 1"},
    "Absolute Neutrophils": {"unit": "/cumm", "range": "2000 - 7000"},
    "Absolute Lymphocytes": {"unit": "/cumm", "range": "1000 - 3000"},
    "Absolute Eosinophils": {"unit": "/cumm", "range": "20 - 500"},
    "Absolute Monocytes": {"unit": "/cumm", "range": "200 - 1000"},
    "RBC Count": {"unit": "Million/cumm", "range": "4.5 - 5.5"},
    "MCV": {"unit": "fL", "range": "81 - 101"},
    "MCH": {"unit": "pg", "range": "27 - 32"},
    "MCHC": {"unit": "g/dL", "range": "31.5 - 34.5"},
    "Hct": {"unit": "%", "range": "40 - 50"},
    "RDW-CV": {"unit": "%", "range": "11.6 - 14.0"},
    "RDW-SD": {"unit": "fL", "range": "39 - 46"},
    "Platelet Count": {"unit": "/cumm", "range": "150000 - 410000"},
    "PCT": {"unit": "", "range": ""},
    "MPV": {"unit": "fL", "range": "7.5 - 11.5"},
    "PDW": {"unit": "", "range": ""},
    "Platelets on Smear": {"unit": "", "range": ""}
}

HEADER_KEYWORDS = {"complete", "test", "description", "ref.", "range",
                   "differential", "absolute", "indices", "interpretation"}
VALID_UNITS    = {"%","g/dL","pg","fL","/cumm","million/cumm","x10³/µL"}

NUMBER_RE = re.compile(r'^\d{1,3}(?:,\d{3})*(?:\.\d+)?$')
RANGE_RE  = re.compile(r'(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)')

def canonical_name(label):
    best = process.extractOne(
        label.lower(),
        PARAM_ALIASES,
        scorer=fuzz.token_sort_ratio
    )
    if best and best[1] > 70:
        return PARAM_ALIASES[best[0]]
    return None

def all_aliases():
    return [a for aliases in PARAM_ALIASES.values() for a in aliases]

def parse_row(row):
    tokens = row['tokens'][:]
    while tokens and (NUMBER_RE.match(tokens[0]) or RANGE_RE.match(tokens[0])):
        tokens.pop(0)
    for i, t in enumerate(tokens):
        if NUMBER_RE.match(t):
            label_tokens = tokens[:i]
            data_tokens = tokens[i:]
            break
    else:
        return None
    label = ' '.join(label_tokens)
    joined_label = ''.join(label_tokens).replace(' ', '').lower()
    print(f"[DEBUG] Extracted label: '{label}' from tokens: {row['tokens']}")
    # Use section context to prioritize aliases
    section = row.get('section', 'main').lower()
    # Build alias lookup for this section
    alias_lookup = {alias.lower(): name for name, aliases in PARAM_ALIASES.items() for alias in aliases}
    section_aliases = alias_lookup
    # If section matches a parameter, prioritize those aliases
    for param, aliases in PARAM_ALIASES.items():
        if param.lower() in section or any(a in section for a in aliases):
            section_aliases = {alias.lower(): param for alias in aliases}
            break
    # 1. Exact match priority (case-insensitive)
    if label.lower() in section_aliases:
        name = section_aliases[label.lower()]
        print(f"[DEBUG] Exact match: '{label}' -> '{name}' (section: {section})")
    # 2. Substring match (hyphen/space-insensitive)
    else:
        found = False
        for n, aliases in PARAM_ALIASES.items():
            for a in aliases:
                if a.replace(' ', '').replace('-', '').lower() in joined_label.replace('-', ''):
                    name = n
                    print(f"[DEBUG] Substring match: '{label}' -> '{name}' via alias '{a}' (section: {section})")
                    found = True
                    break
            if found:
                break
        # 3. Fuzzy match with length-based threshold
        if not found:
            min_score = 90 if len(label) <= 3 else 50
            best = process.extractOne(
                label.lower(),
                section_aliases,
                scorer=fuzz.token_sort_ratio
            )
            if best:
                print(f"[DEBUG] Fuzzy match: '{label}' -> '{best[0]}' (score={best[1]}) (section: {section})")
            name = None
            if best and best[1] >= min_score:
                name = section_aliases[best[0].lower()]
    value = next((t for t in data_tokens if NUMBER_RE.match(t)), None)
    rng_match = RANGE_RE.search(' '.join(data_tokens))
    rng = f"{rng_match.group(1)} to {rng_match.group(2)}" if rng_match else ""
    unit = ""
    for idx in range(len(data_tokens)):
        for length in (1,2):
            candidate = ''.join(data_tokens[idx:idx+length]).lower().replace('fl', 'fL').replace('µl', 'µL')
            if candidate in {u.lower().replace(' ', '') for u in VALID_UNITS}:
                unit = ' '.join(data_tokens[idx:idx+length])
                break
        if unit:
            break
    # Fallback to canonical unit/range if missing
    if name and (not unit or not rng):
        meta = PARAMETER_META.get(name, {})
        if not unit:
            unit = meta.get('unit', '')
        if not rng:
            rng = meta.get('range', '')
    if name and value:
        return {"parameter": name, "value": value, "unit": unit, "range": rng}
    else:
        print(f"[DEBUG] Skipped: name={name}, value={value}, unit={unit}, range={rng}")
    return None

if len(sys.argv) < 2:
    print("Usage: python doctr_cbc_extract.py <image_or_pdf_path>")
    sys.exit(1)

file_path = sys.argv[1]
doc = DocumentFile.from_images(file_path) if file_path.lower().endswith(('.png', '.jpg', '.jpeg')) else DocumentFile.from_pdf(file_path)
model = ocr_predictor(pretrained=True)
result = model(doc)
json_output = result.export()

# Get page width for relative indentation
page_width = 1.0
if json_output['pages'] and json_output['pages'][0]['dimensions']:
    page_width = json_output['pages'][0]['dimensions'][0]

# --- Build rows with X/Y position ---
y_tolerance = 0.015
x_indent_threshold_px = 0.08 * page_width  # 8% of page width

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
                    'x': center_x * page_width,  # convert to px
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

# --- Pass 1: Tag rows as header or data ---
for row in rows:
    row['is_header'] = any(any(kw in t.lower() for kw in HEADER_KEYWORDS) for t in row['tokens'])
    row['is_data'] = not row['is_header']

# --- Pass 2: Assign section context ---
current_section = 'main'
for row in rows:
    if row['is_header']:
        current_section = ' '.join(row['tokens'])
    row['section'] = current_section

# --- Targeted merge: only merge orphan label rows with immediate data-only row ---
def is_orphan_label(row):
    # no numeric tokens & matches an alias
    return (
        all(not NUMBER_RE.match(t) for t in row['tokens'])
        and any(t.lower() in {a.lower() for a in all_aliases()} for t in row['tokens'])
    )

def is_data_only(row):
    return any(NUMBER_RE.match(t) for t in row['tokens']) and all(
        not t.isalpha() for t in row['tokens'] if not NUMBER_RE.match(t)
    )

# --- Second pass: merge data-only + orphan-label rows ---
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

# --- Intra-row splitting for multi-parameter rows ---
def split_multi_param_row(row):
    tokens = row['tokens']
    results = []
    # Build alias list sorted by descending token count
    alias_tuples = []
    for name, aliases in PARAM_ALIASES.items():
        for alias in aliases:
            alias_toks = alias.lower().split()
            alias_tuples.append((name, alias, alias_toks))
    alias_tuples.sort(key=lambda x: -len(x[2]))
    i = 0
    while i < len(tokens):
        match = None
        for name, alias, alias_toks in alias_tuples:
            n = len(alias_toks)
            if i+n <= len(tokens) and [t.lower() for t in tokens[i:i+n]] == alias_toks:
                match = (name, i, i+n)
                break
        if match:
            name, start, end = match
            # Value: next token after label that matches NUMBER_RE
            value = None
            unit = ''
            rng = ''
            j = end
            while j < len(tokens):
                if NUMBER_RE.match(tokens[j]):
                    value = tokens[j]
                    break
                j += 1
            # Range: next tokens matching RANGE_RE
            rng_match = RANGE_RE.search(' '.join(tokens[end:j+3]))
            if rng_match:
                rng = f"{rng_match.group(1)} to {rng_match.group(2)}"
            # Unit: next token(s) in VALID_UNITS
            for k in range(j+1, min(j+4, len(tokens))):
                candidate = ''.join(tokens[k:k+2]).lower().replace('fl', 'fL').replace('µl', 'µL')
                if candidate in {u.lower().replace(' ', '') for u in VALID_UNITS}:
                    unit = ' '.join(tokens[k:k+2])
                    break
                candidate = tokens[k].lower().replace('fl', 'fL').replace('µl', 'µL')
                if candidate in {u.lower().replace(' ', '') for u in VALID_UNITS}:
                    unit = tokens[k]
                    break
            # Fallbacks
            meta = PARAMETER_META.get(name, {})
            if not unit:
                unit = meta.get('unit', '')
            if not rng:
                rng = meta.get('range', '')
            if name and value:
                results.append({'parameter': name, 'value': value, 'unit': unit, 'range': rng})
            i = j+1
        else:
            i += 1
    # Special: handle 'Total Leucocyte Count' if present in tokens
    for idx, t in enumerate(tokens):
        if t.lower() in {a.lower() for a in PARAM_ALIASES['Total Leucocyte Count']}:
            # Look for value and unit after
            for j in range(idx+1, len(tokens)):
                if NUMBER_RE.match(tokens[j]):
                    value = tokens[j]
                    unit = ''
                    rng = ''
                    # Range: next tokens matching RANGE_RE
                    rng_match = RANGE_RE.search(' '.join(tokens[j+1:j+4]))
                    if rng_match:
                        rng = f"{rng_match.group(1)} to {rng_match.group(2)}"
                    # Unit: next token(s) in VALID_UNITS
                    for k in range(j+1, min(j+4, len(tokens))):
                        candidate = ''.join(tokens[k:k+2]).lower().replace('fl', 'fL').replace('µl', 'µL')
                        if candidate in {u.lower().replace(' ', '') for u in VALID_UNITS}:
                            unit = ' '.join(tokens[k:k+2])
                            break
                        candidate = tokens[k].lower().replace('fl', 'fL').replace('µl', 'µL')
                        if candidate in {u.lower().replace(' ', '') for u in VALID_UNITS}:
                            unit = tokens[k]
                            break
                    meta = PARAMETER_META.get('Total Leucocyte Count', {})
                    if not unit:
                        unit = meta.get('unit', '')
                    if not rng:
                        rng = meta.get('range', '')
                    results.append({'parameter': 'Total Leucocyte Count', 'value': value, 'unit': unit, 'range': rng})
                    break
    return results

# --- Extraction ---
clean = []
for r in rows:
    # If row contains multiple parameters, split and parse each
    multi_results = split_multi_param_row(r)
    if multi_results:
        clean.extend(multi_results)
    else:
        parsed = parse_row(r)
        if parsed:
            clean.append(parsed)

# Save clean JSON
with open('cbc_parameters.json', 'w') as f:
    json.dump(clean, f, indent=2)

print(json.dumps(clean, indent=2)) 