import regex as re
import unicodedata
from text_to_num import text2num

PHRASE_MAP = {
    "erst mal": "erstmal",
    "letztes mal": "letztesmal",
    "letzten mal": "letztenmal",
    "nächstes mal": "naechstesmal",
    "nächsten mal": "naechstenmal",
    "dieses mal": "diesesmal"
}

# --- Mappings ---
WORD_NUM_MAP = {
    "null":0,"ein": 1, "eins": 1, "eine": 1, "einen": 1,
    "zwei": 2, "drei": 3, "vier": 4, "fünf": 5,
    "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10
}

DENOM_MAP = {
    "einhalb": 2,
    "halb": 2, "halbe": 2,
    "drittel": 3, "viertel": 4,
    "fünftel": 5, "sechstel": 6,
    "siebtel": 7, "achtel": 8,
    "neuntel": 9, "zehntel": 10
}

GERMAN_MATH_KEEP_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzäöüßABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789+-*/=^[]{}"
)

def keep_german_math_flatten_parentheses(s: str):
    """
    Keep German letters, numbers, math symbols.
    Flatten text inside parentheses by removing the parentheses themselves.
    Remove punctuation, emojis, and other irrelevant symbols.
    """
    # Replace parentheses with their content
    s = re.sub(r"\(([^)]+)\)", r"\1", s)

    cleaned = []
    for c in s:
        if c in GERMAN_MATH_KEEP_CHARS:
            cleaned.append(c)
        else:
            cat = unicodedata.category(c)
            # Keep numbers or math symbols
            if cat.startswith("N") or cat.startswith("S"):
                cleaned.append(c)
            else:
                # Replace everything else (punctuation, emoji) with space
                cleaned.append(" ")

    return "".join(cleaned)

# --- FRACTIONS (multi-word only) ---
def replace_fraction(match):
    num_str = match.group(1)
    frac_word = match.group(2)
    num = WORD_NUM_MAP.get(num_str.lower(), num_str)
    denom = DENOM_MAP.get(frac_word.lower(), None)
    return f"{num}/{denom}" if denom else match.group(0)


# --- SINGLE-WORD FRACTIONS (standalone only) ---
def german_fraction_to_numeric(text):
    text = text.strip().lower()

    # --- fraction patterns ---
    fraction_map = {
        "zwei fünftel": "2/5",
        "zwei drittel": "2/3",
        "dreiviertel": "3/4",
        "einhalb": "1/2",
        "halb": "1/2",
        "viertel": "1/4",
        "drittel": "1/3",
        "fünftel": "1/5",
        "sechstel": "1/6",
        "siebtel": "1/7",
        "achtel": "1/8",
        "neuntel": "1/9",
        "zehntel": "1/10",
        "zwölftel": "1/12",
        "hundertstel": "1/100",
        "tausendstel": "1/1000"
    }

    # sort longest first so "zwei drittel" is handled before "drittel"
    for word, frac in sorted(fraction_map.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r'\b' + re.escape(word) + r'\b'
        text = re.sub(pattern, frac, text)

    # --- Replace superscript numbers like ², ³ ---
    superscripts = {
        '²': '**2', '³': '**3', '⁴': '**4',
        '⁵': '**5', '⁶': '**6', '⁷': '**7',
        '⁸': '**8', '⁹': '**9', '⁰': '**0'
    }
    for sup, norm in superscripts.items():
        text = text.replace(sup, norm)

    # --- Clean spacing ---
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


# --- SUPERSCRIPTS ---
def handle_superscripts(text):
    superscripts = {
        '²': '**2', '³': '**3', '⁴': '**4',
        '⁵': '**5', '⁶': '**6', '⁷': '**7', '⁸': '**8', '⁹': '**9', '⁰': '**0'
    }
    for sup, norm in superscripts.items():
        text = text.replace(sup, norm)
    return text


# --- NUMBER WORDS ---
def convert_number_words(text):
    words = text.split()
    new_words = []
    for w in words:
        try:
            if w == "eins":
                new_words.append("1")
            elif w in ["ein", "eine", "einer", "einem", "einen"]:
                new_words.append(w)
            else:
                new_words.append(str(text2num(w, lang='de')))
        except ValueError:
            new_words.append(w)
    return " ".join(new_words)


def normalize_function_mapping(text):
    # Pattern: <function> von <start> (bis|nach) <end>
    pattern = re.compile(
        r"\b(\w+)\s+von\s+([-\w().]+)\s+(bis|nach)\s+([^\s]+)",
        flags=re.IGNORECASE
    )
    
    def repl(match):
        func = match.group(1)
        start = match.group(2)
        end = match.group(4)
        # Option 1: No space (concatenate function and start)
        return f"{func}{start} -> {end}"
        # Option 2: Function-style notation (preferred for clarity)
        # return f"{func}({start}) -> {end}"
    
    return pattern.sub(repl, text)



# --- YOUR ORIGINAL normalize_math_german FUNCTION (unchanged) ---
def normalize_math_german(text):
    # text = unicodedata.normalize("NFC", text)
    replacements = {
        r'\bplus\b': '+',
        r'\bminus\b': '-',
        r'\bmal\b': '*',
        r'\bgeteilt\s+durch\b': '/',
        r'\bdurch\s+geteilt\b': '/',
        r'\bdurch\s+teilt\b': '/',
        r'\bteilt\s+durch\b': '/',
        # don't replace all "durch" blindly
        r'(?<=\b(?:\d+|[a-zA-Z]))\s*durch\s+(?=\b(?:\d+|[a-zA-Z]))': '/',  # "zahl durch zahl" or "x durch y"
        r'(?<!\b(?:letztes|letzten|dieses|nächstes|nächsten)\s)\bmal\b': '*',
        r'\büber\b': '/',
        r'\bist\s+gleich\b': '=',
        r'\bgleich\b': '=',
        r'\bungefähr\b': '≈',
        r'\bkleiner\s+als\b': '<',
        r'\bgrößer\s+als\b': '>',
        r'\bhoch\b': '^',
        r'\bzum\s+quadrat\b': '**2',
        r'\bquadrat\b': '**2',
        r'\bquadrate\b': '**2',
        r'\bzum\s+kubik\b': '**3',
        r'\bkubik\b': '**3',
        r'\bprozent\b': '%'
    }


    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    text = re.sub(r'(\d),(\d)', r'\1.\2', text)
    text = re.sub(r'\bminus\s+(\d+)', r'-\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bplus\s+(\d+)', r'+\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bhoch\s+(\d+)', r'^\1', text, flags=re.IGNORECASE)

    symbol_map = {
        "×": "*",
        "÷": "/",
        "−": "-",
        "–": "-",
        "—": "-",
        "·": "*",
        "∙": "*"
    }
    for sym, repl in symbol_map.items():
        text = text.replace(sym, repl)

    text = re.sub(r'\s*([\+\-\*/=\^])\s*', r' \1 ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# --- SPACING CLEANUP ---
def normalize_math_spacing(text):
    text = re.sub(r'\s*([\+\-\*/=\^()])\s*', r' \1 ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# --- DECIMAL "KOMMA" HANDLER ---
def replace_numeric_komma(text):
    # Match patterns like "drei komma fünf" or "3 komma 5"
    pattern = re.compile(
        r'\b(?P<intpart>(\d+|null|ein|eins|eine|einen|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn))\s+komma\s+(?P<fracpart>(\d+|null|ein|eins|eine|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn))\b',
        flags=re.IGNORECASE,
    )

    def repl(m):
        intp = WORD_NUM_MAP.get(m.group('intpart').lower(), m.group('intpart'))
        fracp = WORD_NUM_MAP.get(m.group('fracpart').lower(), m.group('fracpart'))
        return f"{intp}.{fracp}"

    return pattern.sub(repl, text)

def normalize_phrases(text):
    for phrase, replacement in PHRASE_MAP.items():
        # Use \b for word boundaries to avoid partial replacements
        pattern = r"\b" + re.escape(phrase) + r"\b"
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

# --- MAIN PIPELINE ---
class NormalizeGermanMathExpression:
    def __call__(self, text: str):
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()

        text = normalize_phrases(text)

        text = normalize_function_mapping(text)

        text = keep_german_math_flatten_parentheses(text)

        # 1️⃣ Multi-word fractions
        text = re.sub(
            r"\b(\d+|null|ein|eins|eine|einen|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn)\s+(null|einhalb|halb|halbe|drittel|viertel|fünftel|sechstel|siebtel|achtel|neuntel|zehntel)\b",
            replace_fraction,
            text,
        )

        # 2️⃣ Single-word fractions
        text = german_fraction_to_numeric(text)

        # 3️⃣ Handle decimal Komma
        text = replace_numeric_komma(text)

        # 4️⃣ Number words → digits
        text = convert_number_words(text)

        # 5️⃣ Math words → symbols
        text = normalize_math_german(text)

        # 6️⃣ Superscripts
        text = handle_superscripts(text)

        # 7️⃣ Spacing cleanup
        text = normalize_math_spacing(text)

        return text
