"""Static reference lists used to canonicalise the state / region component.

These are plain naming conventions (US postal codes, Indian state names and
their common abbreviations, French regions/departments) used only to make
"CA" and "California" compare equal. No business identities are looked up.
"""

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
    "pr": "puerto rico", "gu": "guam", "vi": "virgin islands",
}

IN_STATES = {
    "andhra pradesh": ["ap", "andhra"],
    "arunachal pradesh": ["ar"],
    "assam": ["as"],
    "bihar": ["br"],
    "chhattisgarh": ["cg", "ct", "chattisgarh"],
    "goa": ["ga"],
    "gujarat": ["gj"],
    "haryana": ["hr"],
    "himachal pradesh": ["hp"],
    "jharkhand": ["jh"],
    "karnataka": ["ka"],
    "kerala": ["kl", "keralam"],
    "madhya pradesh": ["mp"],
    "maharashtra": ["mh"],
    "manipur": ["mn"],
    "meghalaya": ["ml"],
    "mizoram": ["mz"],
    "nagaland": ["nl"],
    "odisha": ["od", "or", "orissa"],
    "punjab": ["pb"],
    "rajasthan": ["rj"],
    "sikkim": ["sk"],
    "tamil nadu": ["tn", "tamilnadu"],
    "telangana": ["tg", "ts"],
    "tripura": ["tr"],
    "uttar pradesh": ["up"],
    "uttarakhand": ["uk", "ut", "uttaranchal"],
    "west bengal": ["wb"],
    "delhi": ["dl", "nct of delhi", "new delhi"],
    "jammu and kashmir": ["jk", "jammu & kashmir"],
    "ladakh": ["la"],
    "chandigarh": ["ch"],
    "puducherry": ["py", "pondicherry"],
    "dadra and nagar haveli and daman and diu": ["dn", "dd"],
    "andaman and nicobar islands": ["an"],
    "lakshadweep": ["ld"],
}

# State names written in native scripts, as they appear in Source 2/3.
IN_NATIVE_STATES = {
    "महाराष्ट्र": "maharashtra", "दिल्ली": "delhi", "उत्तर प्रदेश": "uttar pradesh",
    "ಕರ್ನಾಟಕ": "karnataka", "தமிழ்நாடு": "tamil nadu", "ગુજરાત": "gujarat",
    "পশ্চিমবঙ্গ": "west bengal", "తెలంగాణ": "telangana", "हरियाणा": "haryana",
    "राजस्थान": "rajasthan", "കേരളം": "kerala", "बिहार": "bihar",
    "मध्य प्रदेश": "madhya pradesh", "ఆంధ్రప్రదేశ్": "andhra pradesh", "ਪੰਜਾਬ": "punjab",
    "ଓଡ଼ିଶା": "odisha",
}

FR_REGIONS = {
    "auvergne rhone alpes": ["ain", "allier", "ardeche", "cantal", "drome", "isere", "loire",
                             "haute loire", "puy de dome", "rhone", "savoie", "haute savoie"],
    "bourgogne franche comte": ["cote d or", "doubs", "jura", "nievre", "haute saone",
                                "saone et loire", "yonne", "territoire de belfort"],
    "bretagne": ["cotes d armor", "finistere", "ille et vilaine", "morbihan"],
    "centre val de loire": ["cher", "eure et loir", "indre", "indre et loire", "loir et cher",
                            "loiret"],
    "corse": ["corse du sud", "haute corse"],
    "grand est": ["ardennes", "aube", "marne", "haute marne", "meurthe et moselle", "meuse",
                  "moselle", "bas rhin", "haut rhin", "vosges"],
    "hauts de france": ["aisne", "nord", "oise", "pas de calais", "somme"],
    "ile de france": ["paris", "seine et marne", "yvelines", "essonne", "hauts de seine",
                      "seine saint denis", "val de marne", "val d oise"],
    "normandie": ["calvados", "eure", "manche", "orne", "seine maritime"],
    "nouvelle aquitaine": ["charente", "charente maritime", "correze", "creuse", "dordogne",
                           "gironde", "landes", "lot et garonne", "pyrenees atlantiques",
                           "deux sevres", "vienne", "haute vienne"],
    "occitanie": ["ariege", "aude", "aveyron", "gard", "haute garonne", "gers", "herault", "lot",
                  "lozere", "hautes pyrenees", "pyrenees orientales", "tarn", "tarn et garonne"],
    "pays de la loire": ["loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "provence alpes cote d azur": ["alpes de haute provence", "hautes alpes", "alpes maritimes",
                                   "bouches du rhone", "var", "vaucluse"],
}


def _key(s: str) -> str:
    return " ".join(s.replace("-", " ").replace("'", " ").split())


def build_state_lookup():
    """Return {country_lower: {component_key: canonical_state}}."""
    us = {}
    for code, name in US_STATES.items():
        us[code] = name
        us[name] = name
    ind = {}
    for name, alts in IN_STATES.items():
        ind[name] = name
        for a in alts:
            ind[_key(a)] = name
    fr = {}
    for region, depts in FR_REGIONS.items():
        fr[region] = region
        for d in depts:
            # Departments are mapped to their region so "Gironde" == "Nouvelle-Aquitaine".
            fr[d] = region
    return {"us": us, "india": ind, "france": fr}
