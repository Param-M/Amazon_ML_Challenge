"""Name and address normalisation.

Everything here is deterministic string processing; the only learned input is
`token_map`, a native-script -> Latin token dictionary learned from the
training ground truth (see indic_dict.py).
"""
import re

from anyascii import anyascii

from .geo import IN_NATIVE_STATES, build_state_lookup

NONASCII = re.compile(r"[^\x00-\x7f]")
INDIC_RUN = re.compile(r"[ऀ-෿‌‍]+")
STATE_LOOKUP = build_state_lookup()

# --------------------------------------------------------------------------- names

LEGAL = {
    "inc": "inc", "incorporated": "inc", "incorp": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cos": "co",
    "llc": "llc", "ltd": "ltd", "limited": "ltd", "lmt": "ltd",
    "pvt": "pvt", "private": "pvt", "pte": "pvt",
    "llp": "llp", "lp": "lp", "plc": "plc", "pc": "pc", "pllc": "pllc", "pa": "pa",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "scop": "scop", "gie": "gie", "selarl": "selarl", "scm": "scm",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "pty": "pty", "srl": "srl", "spa": "spa",
}
LEGAL_CANON = set(LEGAL.values()) | {"public"}
NAME_STOP = {"the", "and", "of", "de", "du", "des", "la", "le", "les", "et", "d", "l", "a", "an",
             "au", "aux", "en"}

RE_ID = re.compile(r"\(?\bid\s*[:#]?\s*\d+\)?")
RE_HASHNUM = re.compile(r"#\s*\d+")
RE_PHONE = re.compile(r"(?:^|\s)[-–]?\s*\+?\d[\d\s\-]{6,}\d(?=\s|$)")
TLD = r"(?:c[o0]m|net|[o0]rg|in|c[o0]\.in|[o0]rg\.in|net\.in|c[o0]|biz|inf[o0]|fr|us|i[o0]|c[o0]\.uk)"
RE_PIPE_DOM = re.compile(r"\|\s*(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\." + TLD + r"\S*")
RE_DOMAIN = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\." + TLD + r"/?")
RE_DBA = re.compile(
    r"\s+(?:d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|doing business as|a\s*/\s*k\s*/\s*a|aka|"
    r"t\s*/\s*a|trading as)\s+")
RE_ORD = re.compile(r"\d+(?:st|nd|rd|th)")
OCR = str.maketrans("0134578", "oleastb")


def fold_ascii(s: str, token_map=None) -> str:
    """Latin-ise a string. Native-script words go through the learned map first."""
    if not NONASCII.search(s):
        return s
    if token_map:
        s = INDIC_RUN.sub(lambda m: token_map.get(m.group(0), m.group(0)), s)
    return anyascii(s)


def _ocr_fix(tok: str) -> str:
    """Undo OCR-style digit substitutions inside words: 'va1ley' -> 'valley'."""
    if tok.isalpha() or tok.isdigit() or RE_ORD.fullmatch(tok):
        return tok
    if sum(c.isalpha() for c in tok) >= 2:
        return tok.translate(OCR)
    return tok


def _name_tokens(s: str):
    s = s.replace("&", " and ").replace("+", " and ")
    s = re.sub(r"[.'`’]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    out = []
    for t in s.split():
        t = _ocr_fix(t)
        t = LEGAL.get(t, t)
        if out and out[-1] == t:  # "Inc Inc"
            continue
        out.append(t)
    # "public limited" is a legal form, "public library" is not
    for i in range(len(out) - 1):
        if out[i] == "public" and out[i + 1] == "ltd":
            out[i] = "public"
    return out


def _core(tokens):
    core = [t for t in tokens if t not in LEGAL_CANON and t not in NAME_STOP]
    return core if core else [t for t in tokens if t not in NAME_STOP]


def norm_name(raw: str, token_map=None):
    """Return (full, core, alias_core, legal, domain_label)."""
    s = fold_ascii(raw or "", token_map).lower()
    s = RE_ID.sub(" ", s)
    s = RE_HASHNUM.sub(" ", s)
    alt_dom = ""
    m = RE_PIPE_DOM.search(s)
    if m:
        alt_dom = m.group(1)
        s = s[: m.start()]
    s = RE_PHONE.sub(" ", s).strip()
    s = re.sub(r"^[^a-z0-9]+", "", s)
    m = RE_DOMAIN.fullmatch(s.strip(" -*#@.>[]()<|"))
    if m:
        # The whole name is a web domain; its label is segmented into words later.
        dom = "".join(_ocr_fix(p) for p in m.group(1).split("-"))
        return dom, dom, "", "", dom
    parts = RE_DBA.split(s, maxsplit=1)
    left = _name_tokens(parts[0])
    right = _name_tokens(parts[1]) if len(parts) > 1 else []
    full = left + right
    legal = sorted({t for t in full if t in LEGAL_CANON})
    core = _core(left)
    alias = _core(right) if right else []
    return (" ".join(full), " ".join(core), " ".join(alias), " ".join(legal), alt_dom)


# --------------------------------------------------------------------------- addresses

ORD_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
    "twelfth": "12", "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
    "sixteenth": "16", "seventeenth": "17", "eighteenth": "18", "nineteenth": "19",
    "twentieth": "20", "thirtieth": "30", "fortieth": "40", "fiftieth": "50",
}
ADDR_ABBR = {
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "aven": "ave", "avn": "ave", "drive": "dr", "drv": "dr", "lane": "ln", "court": "ct",
    "crt": "ct", "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "bvd": "blvd", "boul": "blvd",
    "highway": "hwy", "parkway": "pkwy", "pky": "pkwy", "place": "pl", "circle": "cir",
    "trail": "trl", "terrace": "ter", "square": "sq", "alley": "aly", "mount": "mt",
    "fort": "ft", "heights": "hts", "point": "pt", "center": "ctr", "centre": "ctr",
    "crossing": "xing", "expressway": "expy", "freeway": "fwy", "turnpike": "tpke",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne",
    "northwest": "nw", "southeast": "se", "southwest": "sw",
    "near": "nr", "opposite": "opp", "behind": "bhd",
    "village": "vill", "vil": "vill", "ngr": "nagar", "colony": "col", "sector": "sec",
    "block": "blk", "flr": "floor", "fl": "floor", "building": "bldg", "district": "dist",
    "crs": "cross", "apartment": "apt", "appartment": "apt", "appartement": "apt", "appt": "apt",
    "suite": "unit", "room": "rm", "allee": "allee", "chemin": "chemin", "impasse": "imp",
    "route": "rte", "sainte": "ste",
}
FR_ABBR = {"r": "rue", "all": "allee", "ch": "chemin", "che": "chemin", "pl": "pl",
           "st": "saint", "saint": "saint", "ste": "sainte", "sainte": "sainte"}
US_ABBR = {"ste": "unit"}
ADDR_DROP = {"no", "number", "num", "nos", "null", "none", "nil", "na", "n", "the", "of", "and",
             "de", "du", "des", "la", "le", "les", "l", "d", "et", "au", "aux"}
ADDR_DROP.discard("n")  # "n" is also the north abbreviation; keep it
STREET_WORDS = {"st", "rd", "ave", "dr", "ln", "ct", "blvd", "hwy", "pkwy", "pl", "cir", "trl",
                "ter", "sq", "aly", "way", "rue", "allee", "chemin", "imp", "rte", "quai", "cours",
                "marg", "nagar", "cross", "main", "sec", "blk", "plot", "door", "h", "hno",
                "flat", "floor", "bldg", "nr", "opp", "col", "vill", "path", "pike", "loop",
                "run", "row", "walk", "pass", "bypass", "gali", "lane", "wadi", "peth", "chowk"}
UNIT_WORDS = {"unit", "apt", "rm", "pmb", "po", "box", "suite"}
RE_NULL = re.compile(r"<null>|\bnull\b|\bn/a\b|\bnone\b|\bnil\b")
RE_ORD_SUFFIX = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
RE_HYPHEN_NUM = re.compile(r"(?<![\d-])\d+(?:-\d+)+(?![\d-])")


def _addr_tokens(c: str, country: str):
    c = c.replace("n°", "no ").replace("°", " ")
    c = RE_ORD_SUFFIX.sub(r"\1", c)
    c = re.sub(r"['`’]", "", c)
    c = re.sub(r"[^a-z0-9]+", " ", c)
    out = []
    for t in c.split():
        for p in re.findall(r"\d+|[a-z]+", t):
            if p.isdigit():
                p = p.lstrip("0") or "0"
            else:
                p = ORD_WORDS.get(p, p)
                if country == "france" and p in FR_ABBR:
                    p = FR_ABBR[p]
                elif country == "us" and p in US_ABBR:
                    p = US_ABBR[p]
                else:
                    p = ADDR_ABBR.get(p, p)
                if p in ADDR_DROP:
                    continue
            out.append(p)
    return out


def norm_addr(raw: str, country: str):
    """Return (full, street, locality, unit, numbers, house_number, state)."""
    country = (country or "").strip().lower()
    states = STATE_LOOKUP.get(country, {})
    full, street, loc, unit, nums = [], [], [], [], []
    state = ""
    for comp in (raw or "").split(","):
        comp = comp.strip().strip('"')
        if not comp:
            continue
        if comp in IN_NATIVE_STATES:
            state = IN_NATIVE_STATES[comp]
            continue
        comp = comp.replace("N\u00b0", "No ").replace("n\u00b0", "no ").replace("\u00b0", " ")
        c = fold_ascii(comp).lower()
        c = RE_NULL.sub(" ", c)
        key = " ".join(re.sub(r"[^a-z0-9]+", " ", c.replace("'", "")).split())
        if not key:
            continue
        if key in states:
            state = states[key]
            continue
        toks = _addr_tokens(c, country)
        if not toks:
            continue
        full.extend(toks)
        digits = [t for t in toks if t.isdigit()]
        nums.extend(digits)
        # "16-03" / "4-7/1A" are hyphen-corrupted versions of 1603 / 47: keep the joined form too
        for m in RE_HYPHEN_NUM.finditer(c):
            joined = m.group(0).replace("-", "").lstrip("0")
            if joined and joined not in nums:
                nums.append(joined)
        if toks[0] in UNIT_WORDS:
            unit.extend(toks)
        # a leading "st"/"saint" is a place name (St Louis), not a street type
        elif digits or any(t in STREET_WORDS for t in (toks[1:] if toks[0] in ("st", "saint") else toks)):
            street.extend(toks)
        else:
            loc.extend(toks)
    hnum = ""
    for t in street:
        if t.isdigit():
            hnum = t
            break
    return (" ".join(full), " ".join(street), " ".join(loc), " ".join(unit), " ".join(nums),
            hnum, state)
