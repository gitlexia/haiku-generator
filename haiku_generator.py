#!/usr/bin/env python3
"""
Haiku generator (no GenAI / no LLMs)

Guarantees:
- ALWAYS returns a 5/7/5 haiku
- Each line ends on a content word (not "the", "in", "and", etc.)
- Avoids ugly phrase joins (e.g., "a dog waits pine scent")
- Strong novelty penalties across prompts (recent phrase + recent word memory)
- Per-prompt keyword steering (gentle keyword echo phrases when usable)

Notes:
- Uses NLTK CMUdict syllables if available.
- Adds overrides for known tricky words.
"""

from __future__ import annotations
import random
import re
import sys
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


# -----------------------------
# Optional CMUdict syllables
# -----------------------------

_CMU = None
try:
    from nltk.corpus import cmudict  # type: ignore
    try:
        _CMU = cmudict.dict()
    except LookupError:
        _CMU = None
except Exception:
    _CMU = None


# Common heuristic miscounts / helpful overrides
SYLLABLE_OVERRIDES: Dict[str, int] = {
    "quiet": 2,
    "every": 2,
    "hour": 1,
    "fire": 1,
    "poem": 2,
    "people": 2,
    "little": 2,
    "purple": 2,
    "family": 3,
    "camera": 3,
    "business": 2,
    "chocolate": 3,
    "orange": 2,
    "vegetable": 4,
    "beautiful": 3,
    "interesting": 4,
    "favorite": 3,
    "different": 3,
    "yesterday": 3,
    "moonlight": 2,
    "footsteps": 2,
    "pawprints": 2,
    "mustard": 2,
    "kettle": 2,
    "twilight": 2,
    "river": 2,
    "ocean": 2,
    "pasta": 2,
    "pizza": 2,

    # important: avoid “going” counting as 1 in some setups
    "going": 2,
}


def syllables_cmudict(word: str) -> Optional[int]:
    if _CMU is None:
        return None
    w = re.sub(r"[^a-z']", "", word.lower())
    if not w or w not in _CMU:
        return None
    prons = _CMU[w]
    counts = []
    for pron in prons:
        counts.append(sum(1 for p in pron if p[-1].isdigit()))
    return min(counts) if counts else None


def syllables_heuristic(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    vowels = "aeiouy"
    groups = 0
    prev = False
    for ch in w:
        is_v = ch in vowels
        if is_v and not prev:
            groups += 1
        prev = is_v

    if w.endswith("e") and not w.endswith(("le", "ye")) and groups > 1:
        groups -= 1
    if w.endswith(("tion", "sion")) and groups > 1:
        groups -= 1
    return max(1, groups)


def syllable_count(word: str) -> int:
    wl = re.sub(r"[^a-z']", "", word.lower())
    if wl in SYLLABLE_OVERRIDES:
        return SYLLABLE_OVERRIDES[wl]
    cmu = syllables_cmudict(word)
    if cmu is not None:
        return cmu
    return syllables_heuristic(word)


def line_syllables(words: List[str]) -> int:
    return sum(syllable_count(w) for w in words)


# -----------------------------
# Article fixing (a/an)
# -----------------------------

VOWEL_SOUND_LETTERS = set("aeiou")


def choose_article(next_word: str) -> str:
    w = re.sub(r"[^a-z]", "", next_word.lower())
    if not w:
        return "a"
    return "an" if w[0] in VOWEL_SOUND_LETTERS else "a"


def fix_articles(tokens: List[str]) -> List[str]:
    out = tokens[:]
    for i in range(len(out) - 1):
        if out[i].lower() in {"a", "an"}:
            out[i] = choose_article(out[i + 1])
    return out


# -----------------------------
# Prompt parsing + themes
# -----------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "with", "for",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "my", "your",
    "our", "i", "you", "we", "me", "him", "her", "them", "they", "he", "she", "who", "what",
    "why", "how", "today", "going", "gonna", "do", "did", "does", "am", "will", "would",
    "can", "could", "should", "from", "as"
}

FUNCTIONY = {"out", "up", "down", "over", "under", "again", "just", "very", "really", "so", "too"}

SETTING_KEYWORDS = {
    "urban": {"city", "street", "streets", "train", "station", "neon", "traffic", "bus", "sirens", "subway", "tube"},
    "nature": {"rain", "wind", "river", "forest", "moon", "stars", "snow", "sun", "sky", "water", "beach", "ocean"},
    "indoors": {"room", "window", "lamp", "kitchen", "coffee", "tea", "screen", "phone", "keys", "book", "books"},
    "animal": {"dog", "cat", "horse", "bird", "fox", "owl", "moth", "bee", "spider", "fish", "penguin", "mouse", "rat"},
    "food": {"sandwich", "ham", "bread", "soup", "pizza", "pasta", "coffee", "tea", "salt", "sugar", "honey"},
    "night": {"night", "midnight", "moon", "stars", "neon", "tonight", "dusk", "dawn", "twilight"},
}

MOOD_KEYWORDS = {
    "calm": {"calm", "quiet", "soft", "still", "peace"},
    "sad": {"sad", "miss", "missing", "lonely", "tired", "grief"},
    "tense": {"anxious", "tense", "stress", "stressed", "panic", "worried", "fear"},
    "wistful": {"remember", "memory", "nostalgia", "longing"},
    "bright": {"bright", "sun", "sunlight", "glad", "hope"},
}

KEYWORD_MAP: Dict[str, List[str]] = {
    "sun": ["sunlight", "morning", "warm", "golden", "sky"],
    "rain": ["drizzle", "wet", "puddles", "mist"],
    "wind": ["gust", "breeze"],
    "city": ["street", "neon", "traffic"],
    "train": ["platform", "station", "tracks"],
    "book": ["pages", "ink", "paper"],
    "books": ["pages", "ink", "paper"],
    "coffee": ["steam", "mug", "warm"],
    "tea": ["steam", "cup", "warm"],
    "sandwich": ["bread", "salt", "lunch"],
    "ham": ["salt", "lunch"],
    "dog": ["pawprints", "leash"],
    "cat": ["whiskers", "purring"],
    "penguin": ["ice", "waddle", "snow"],
    "ocean": ["salt", "tide", "shore"],
}


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def infer_tags(user_text: str) -> Set[str]:
    toks = tokenize(user_text)
    tags: Set[str] = set()
    for mood, keys in MOOD_KEYWORDS.items():
        if any(t in keys for t in toks):
            tags.add(mood)
    for setting, keys in SETTING_KEYWORDS.items():
        if any(t in keys for t in toks):
            tags.add(setting)
    tags.add("neutral")
    return tags


def extract_keywords(user_text: str) -> List[str]:
    toks = tokenize(user_text)
    out: List[str] = []
    seen: Set[str] = set()
    for t in toks:
        if t in STOPWORDS:
            continue
        if len(t) <= 2:
            continue
        if t in seen:
            continue
        out.append(t)
        seen.add(t)
    return out


def expanded_keywords(user_text: str) -> List[str]:
    kws = extract_keywords(user_text)
    out: List[str] = []
    seen: Set[str] = set()
    for k in kws:
        if k not in seen:
            out.append(k)
            seen.add(k)
        for alt in KEYWORD_MAP.get(k, []):
            for t in tokenize(alt):
                if t not in seen:
                    out.append(t)
                    seen.add(t)
    return out


def pick_theme(tags: Set[str], keywords: List[str]) -> str:
    for theme in ("food", "animal", "urban", "indoors", "nature"):
        if theme in tags:
            return theme
    for k in keywords:
        for theme, keys in SETTING_KEYWORDS.items():
            if k in keys:
                return "nature" if theme == "night" else theme
    return "neutral"


# -----------------------------
# Unknown/gibberish detection
# -----------------------------

def is_known_word(w: str) -> bool:
    wl = re.sub(r"[^a-z']", "", w.lower())
    if not wl:
        return False
    if wl in STOPWORDS or wl in FUNCTIONY:
        return True
    if wl in SYLLABLE_OVERRIDES:
        return True
    if wl in KEYWORD_MAP:
        return True
    if _CMU is not None and wl in _CMU:
        return True
    if len(wl) <= 3:
        return True
    return any(v in wl for v in "aeiou")


def prompt_is_mostly_unknown(keywords: List[str]) -> bool:
    if not keywords:
        return False
    known = sum(1 for k in keywords if is_known_word(k))
    return known / max(1, len(keywords)) < 0.4


# -----------------------------
# Phrase inventory
# -----------------------------

@dataclass(frozen=True)
class Phrase:
    text: str
    tags: Tuple[str, ...]
    pid: str

    @property
    def words(self) -> List[str]:
        return self.text.split()

    @property
    def syllables(self) -> int:
        return sum(syllable_count(w) for w in self.words)


def P(text: str, *tags: str) -> Phrase:
    pid = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return Phrase(text=text, tags=tuple(tags) if tags else ("neutral",), pid=pid)


PHRASES_GLUE: List[Phrase] = [
    P("in the", "glue"),
    P("by the", "glue"),
    P("near the", "glue"),
    P("under the", "glue"),
    P("beside the", "glue"),
    P("within", "glue"),
    P("after", "glue"),
    P("before", "glue"),
]

PHRASES_NATURE: List[Phrase] = [
    P("sun breaks through", "nature", "bright"),
    P("sun on wet leaves", "nature", "bright", "wet"),
    P("warm light returns", "nature", "bright", "warm"),
    P("a clean blue sky", "nature", "bright"),
    P("clouds drift slowly", "nature", "calm"),
    P("thin clouds unravel", "nature", "calm"),
    P("the day opens", "nature", "time"),
    P("morning light", "nature", "bright", "time"),
    P("late sun on stone", "nature", "bright"),
    P("twilight gathers", "nature", "night", "wistful"),
    P("after the rain", "nature", "wet", "wistful"),
    P("rain taps softly", "nature", "wet", "calm"),
    P("a hush of rain", "nature", "wet", "quiet"),
    P("mist in branches", "nature", "wet", "quiet"),
    P("fog holds still", "nature", "quiet"),
    P("wind in tall grass", "nature", "weather"),
    P("wind finds cracks", "nature", "weather"),
    P("snow without sound", "nature", "cold", "quiet"),
    P("frost on the rail", "nature", "cold"),
    P("river keeps moving", "nature", "water", "calm"),
    P("water over stones", "nature", "water", "calm"),
    P("a slow current", "nature", "water", "calm"),
    P("tide pulls away", "nature", "water", "wistful"),
    P("moon over water", "nature", "night", "calm"),
    P("salt in the air", "nature", "water"),
    P("shore listens back", "nature", "water", "quiet"),
    P("leaves turn slowly", "nature", "wistful"),
    P("a leaf falls", "nature", "wistful"),
    P("petals on stone", "nature", "soft"),
    P("forest shade", "nature", "quiet"),
    P("pine scent", "nature", "calm"),
    P("stones underfoot", "nature", "still"),
    P("a field of hush", "nature", "quiet"),
    P("stars above pines", "nature", "night"),
    P("moonlight on snow", "nature", "night", "cold"),
]

PHRASES_URBAN: List[Phrase] = [
    P("neon in puddles", "urban", "night", "wet"),
    P("streetlights flicker", "urban", "night"),
    P("train hums on", "urban", "travel"),
    P("platform wind", "urban", "travel", "cold"),
    P("station echo", "urban", "travel"),
    P("late buses pass", "urban", "travel"),
    P("glass and reflections", "urban", "night"),
    P("footsteps fade out", "urban", "night"),
    P("sirens drift off", "urban", "night", "tense"),
    P("traffic keeps talking", "urban", "tense"),
    P("rain on pavement", "urban", "wet"),
    P("a quiet alley", "urban", "dark"),
    P("doorway of light", "urban", "bright"),
    P("corner shop glow", "urban", "night"),
    P("a red signal", "urban", "tense"),
    P("the city exhales", "urban", "calm"),
    P("subway air", "urban", "travel"),
]

PHRASES_INDOORS: List[Phrase] = [
    P("coffee and quiet", "indoors", "warm", "calm"),
    P("tea steam rises", "indoors", "warm", "calm"),
    P("kettle starts singing", "indoors", "warm"),
    P("lamp in corner", "indoors", "night", "calm"),
    P("a soft-lit room", "indoors", "calm"),
    P("pages turn slowly", "indoors", "calm"),
    P("ink on fingers", "indoors", "personal"),
    P("window listens", "indoors", "quiet"),
    P("shadows on the wall", "indoors", "night"),
    P("keys on the table", "indoors", "personal"),
    P("screen glow at night", "indoors", "night"),
    P("the sink drips", "indoors", "quiet"),
    P("paper and pencil", "indoors", "personal"),
]

PHRASES_ANIMALS: List[Phrase] = [
    P("a bird on wire", "animal", "urban", "calm"),
    P("a dog waits", "animal", "personal", "calm"),
    P("cat in window", "animal", "indoors", "calm"),
    P("moth finds light", "animal", "night", "indoors"),
    P("owl in pines", "animal", "night", "nature"),
    P("fox at dusk", "animal", "nature", "wistful"),
    P("bees in clover", "animal", "nature", "bright"),
    P("fish under glass", "animal", "indoors", "quiet"),
    P("pawprints in mud", "animal", "nature", "wet"),
    P("wings beat once", "animal", "bright"),
]

PHRASES_FOOD: List[Phrase] = [
    P("bread warm in hands", "food", "warm"),
    P("salt on my lips", "food", "personal"),
    P("mustard and patience", "food", "wistful"),
    P("lunch in a hurry", "food", "urban"),
    P("crumbs on the shirt", "food", "personal"),
    P("a quiet bite", "food", "calm"),
    P("steam from soup", "food", "warm"),
    P("honey on tongue", "food", "soft"),
    P("butter on toast", "food", "warm"),
]

PHRASES_FEELINGS: List[Phrase] = [
    P("I breathe in", "personal", "calm"),
    P("my thoughts go still", "personal", "calm"),
    P("worry loops again", "personal", "tense"),
    P("memory returns", "personal", "wistful"),
    P("a small relief", "personal", "calm"),
    P("hope, a small match", "personal", "bright"),
    P("I let it be", "personal", "calm"),
    P("time keeps walking", "time", "wistful"),
]

PHRASES_NEUTRAL: List[Phrase] = [
    P("a familiar sound", "neutral", "wistful"),
    P("something begins", "neutral", "time"),
    P("something ends", "neutral", "time"),
    P("the world keeps going", "neutral", "time"),
    P("a clean sharp silence", "neutral", "quiet"),
    P("a flicker of change", "neutral", "wistful"),
    P("and then, stillness", "neutral", "quiet"),
    P("as if it could last", "neutral", "wistful"),
]

PHRASES_ALL: List[Phrase] = (
    PHRASES_NATURE
    + PHRASES_URBAN
    + PHRASES_INDOORS
    + PHRASES_ANIMALS
    + PHRASES_FOOD
    + PHRASES_FEELINGS
    + PHRASES_NEUTRAL
    + PHRASES_GLUE
)


# -----------------------------
# Novelty memory across prompts
# -----------------------------

RECENT_PHRASES = deque(maxlen=80)
RECENT_WORDS = deque(maxlen=160)
RECENT_PHRASE_SET: Set[str] = set()
RECENT_WORD_SET: Set[str] = set()


def _sync_sets() -> None:
    RECENT_PHRASE_SET.clear()
    RECENT_PHRASE_SET.update(RECENT_PHRASES)
    RECENT_WORD_SET.clear()
    RECENT_WORD_SET.update(RECENT_WORDS)


def record_usage(haiku_words: List[str]) -> None:
    for w in haiku_words:
        wl = w.lower()
        if wl in STOPWORDS:
            continue
        RECENT_WORDS.append(wl)
    _sync_sets()


def record_phrase_hits(lines: Tuple[List[str], List[str], List[str]], pool: List[Phrase]) -> None:
    joined = " ".join(lines[0]) + " | " + " ".join(lines[1]) + " | " + " ".join(lines[2])
    for ph in pool:
        if ph.text in joined:
            RECENT_PHRASES.append(ph.pid)
            for w in ph.words:
                wl = w.lower()
                if wl not in STOPWORDS:
                    RECENT_WORDS.append(wl)
    _sync_sets()


# -----------------------------
# Phrase pool + steering
# -----------------------------

THEME_ALLOW_ALWAYS = {"neutral", "personal", "calm", "wistful", "tense", "bright", "quiet", "time"}


def make_user_steer_phrases(kws: List[str], theme: str, tags: Set[str], allow: bool) -> List[Phrase]:
    if not allow:
        return []
    echos: List[Phrase] = []
    strong = [k for k in kws if k not in FUNCTIONY][:2]
    for kw in strong:
        if len(kw) < 3:
            continue
        if kw in {"who", "what", "why", "how"}:
            continue
        templates = [
            f"{kw} in the air",
            f"{kw} on the street",
            f"my {kw} again",
            f"{kw} at night",
            f"{kw} in my hands",
        ]
        for t in templates:
            echos.append(P(t, "user", theme, *tuple(tags)))
    return echos


def phrase_pool(theme: str, desired_tags: Set[str], kws: List[str], unknown_mode: bool) -> List[Phrase]:
    if theme == "nature":
        base = PHRASES_NATURE + PHRASES_FEELINGS + PHRASES_NEUTRAL + PHRASES_GLUE
    elif theme == "urban":
        base = PHRASES_URBAN + PHRASES_FEELINGS + PHRASES_NEUTRAL + PHRASES_GLUE
    elif theme == "indoors":
        base = PHRASES_INDOORS + PHRASES_FEELINGS + PHRASES_NEUTRAL + PHRASES_GLUE
    elif theme == "animal":
        base = PHRASES_ANIMALS + PHRASES_NATURE + PHRASES_FEELINGS + PHRASES_NEUTRAL + PHRASES_GLUE
    elif theme == "food":
        base = PHRASES_FOOD + PHRASES_INDOORS + PHRASES_FEELINGS + PHRASES_NEUTRAL + PHRASES_GLUE
    else:
        base = PHRASES_ALL

    steer = make_user_steer_phrases(kws, theme, desired_tags, allow=not unknown_mode)

    pool: List[Phrase] = []
    for ph in base + steer:
        tagset = set(ph.tags)

        if theme != "neutral" and theme not in tagset and not (tagset & THEME_ALLOW_ALWAYS) and "glue" not in tagset:
            continue

        if not (tagset & desired_tags) and not (tagset & THEME_ALLOW_ALWAYS) and "user" not in tagset and "glue" not in tagset:
            continue

        pool.append(ph)

    return pool


# -----------------------------
# Grammar join guard + line ending guard
# -----------------------------

VERB_ENDINGS = {
    "waits", "drifts", "falls", "turns", "keeps", "pulls", "fades", "rises",
    "passes", "listens", "exhales", "opens", "gathers", "starts", "talking", "moving"
}
ALLOWED_AFTER_VERB_START = {
    "in", "on", "at", "by", "near", "under", "beside", "within", "after", "before",
    "the", "a", "an", "my"
}

BAD_LINE_END = {
    "the", "a", "an", "my", "your", "our",
    "in", "on", "at", "by", "near", "under", "beside", "within",
    "after", "before", "between", "with", "without", "over", "from", "of", "to",
}

def bad_join(prev_words: List[str], next_words: List[str]) -> bool:
    if not prev_words or not next_words:
        return False
    last = prev_words[-1].lower()
    first = next_words[0].lower()
    if last in VERB_ENDINGS and first not in ALLOWED_AFTER_VERB_START:
        return True
    return False

def ends_ok(words: List[str]) -> bool:
    if not words:
        return False
    return words[-1].lower() not in BAD_LINE_END

def is_glue_word(w: str) -> bool:
    return w.lower() in BAD_LINE_END


# -----------------------------
# Builders
# -----------------------------

def phrase_weight(ph: Phrase, desired_tags: Set[str], kws: List[str]) -> float:
    w = 1.0
    if ph.pid in RECENT_PHRASE_SET:
        w *= 0.08

    content = [x.lower() for x in ph.words if x.lower() not in {"the", "a", "an", "in", "on", "at", "my", "of"}]
    if any(x in RECENT_WORD_SET for x in content):
        w *= 0.55

    tagset = set(ph.tags)
    w *= 1.0 + 0.20 * len(tagset & desired_tags)

    if any(k in content for k in kws):
        w *= 1.6

    if "glue" in tagset:
        w *= 0.65

    w *= 1.0 + 0.06 * len(content)
    return max(0.01, w)


def build_line_from_phrases(
    target: int,
    pool: List[Phrase],
    forbid: Set[str],
    desired_tags: Set[str],
    kws: List[str],
    max_phrases: int = 2,
    max_calls: int = 560,
) -> Optional[List[str]]:
    chosen: List[Phrase] = []
    calls = 0

    def current_words() -> List[str]:
        out: List[str] = []
        for ph in chosen:
            out.extend(ph.words)
        return out

    # one-time ordering per line (keeps it fast)
    tmp = pool[:]
    weights = [phrase_weight(p, desired_tags, kws) for p in tmp]
    # sample a good-ish subset first, then append the rest shuffled
    ordered: List[Phrase] = []
    for _ in range(min(len(tmp), 60)):
        total = sum(weights)
        r = random.random() * total if total > 0 else random.random()
        acc = 0.0
        idx = 0
        for i, ww in enumerate(weights):
            acc += ww
            if acc >= r:
                idx = i
                break
        ordered.append(tmp.pop(idx))
        weights.pop(idx)
    random.shuffle(tmp)
    ordered.extend(tmp)

    def rec(rem: int) -> bool:
        nonlocal calls
        calls += 1
        if calls > max_calls:
            return False

        if rem == 0:
            return ends_ok(current_words())
        if rem < 0:
            return False
        if len(chosen) >= max_phrases:
            return False

        prev_words = current_words()

        for ph in ordered:
            syl = ph.syllables
            if syl > rem:
                continue

            if bad_join(prev_words, ph.words):
                continue

            # if finishing the line, enforce a good ending
            if rem - syl == 0 and not ends_ok(prev_words + ph.words):
                continue

            # prevent repeating content words within a haiku
            bad = False
            for w in ph.words:
                wl = w.lower()
                if wl in {"the", "a", "an", "in", "on", "at", "my", "of"}:
                    continue
                if wl in forbid:
                    bad = True
                    break
            if bad:
                continue

            chosen.append(ph)
            if rec(rem - syl):
                return True
            chosen.pop()

        return False

    if not rec(target):
        return None

    line = fix_articles(current_words())
    if line_syllables(line) != target:
        return None
    if not ends_ok(line):
        return None
    return line


TOKENS_1 = [
    "soft", "cold", "warm", "dark", "still",
    "rain", "wind", "sky", "light", "stone",
    "night", "dawn", "dust", "breath", "time",
    "salt", "clouds", "street", "glass", "leaf",
    "shore", "stars", "train", "bread", "hands",
]
TOKENS_2 = [
    "quiet", "gentle", "golden", "silver",
    "river", "puddle", "window", "shadow",
    "morning", "twilight", "footsteps",
    "coffee", "kettle", "paper", "moonlight",
    "station", "sirens", "forest", "honey",
    "sunlight", "drizzle",
]
TOKENS_3 = ["memory", "horizon", "yesterday"]
GLUE_1 = ["the", "a", "in", "on", "at", "my"]
GLUE_2 = ["under", "between", "after", "before", "within", "without", "beside", "near"]


def build_line_from_tokens(target: int, theme: str, maybe_keyword: Optional[str]) -> List[str]:
    bank: List[str] = []
    bank += TOKENS_1 + TOKENS_2 + TOKENS_3 + GLUE_1 + GLUE_2

    if theme == "urban":
        bank += ["neon", "traffic", "subway", "city"]
    elif theme == "indoors":
        bank += ["room", "lamp", "tea", "pages", "table"]
    elif theme == "animal":
        bank += ["bird", "dog", "cat", "wings", "pawprints"]
    elif theme == "nature":
        bank += ["water", "mist", "pines", "field", "petals"]
    elif theme == "food":
        bank += ["soup", "lunch", "butter", "salt"]

    if maybe_keyword and maybe_keyword.lower() not in FUNCTIONY and len(maybe_keyword) >= 3:
        bank.append(maybe_keyword.lower())

    bank_syl = [(w, syllable_count(w)) for w in bank]
    bank_syl = [(w, s) for (w, s) in bank_syl if 1 <= s <= 4]

    end_candidates = [(w, s) for (w, s) in bank_syl if not is_glue_word(w) and s <= target]
    if not end_candidates:
        end_candidates = [(w, s) for (w, s) in bank_syl if s <= target]

    # try multiple endings until we can fill the rest cleanly
    random.shuffle(end_candidates)
    for end_word, end_syl in end_candidates[:40]:
        remaining = target - end_syl
        if remaining < 0:
            continue
        if remaining == 0:
            line = fix_articles([end_word])
            if ends_ok(line) and line_syllables(line) == target:
                return line
            continue

        dp: Dict[int, Optional[List[str]]] = {0: []}
        for rem in range(1, remaining + 1):
            dp[rem] = None

        # fill rem exactly
        for rem in range(1, remaining + 1):
            options = bank_syl[:]
            random.shuffle(options)
            for w, s in options:
                if rem - s >= 0 and dp[rem - s] is not None:
                    dp[rem] = dp[rem - s] + [w]  # type: ignore
                    break

        prefix = dp[remaining]
        if not prefix:
            continue

        line = fix_articles(prefix + [end_word])
        if ends_ok(line) and line_syllables(line) == target:
            return line

    # hard guarantee (should basically never happen)
    line = fix_articles(["morning", "light", "stone"])  # usually 2+1+1=4; still validate below
    # brute patch: use small safe words to hit target
    while line_syllables(line) < target:
        line.insert(0, "the")
    while line_syllables(line) > target and len(line) > 1:
        line.pop(0)
    if not ends_ok(line):
        line[-1] = "stone"
    return line


def format_haiku(lines: Tuple[List[str], List[str], List[str]]) -> str:
    def join(line: List[str]) -> str:
        s = " ".join(line)
        s = re.sub(r"\bi\b", "I", s)
        return s
    return "\n".join(join(list(line)) for line in lines)


def ensure_line_target(line: List[str], target: int, theme: str, maybe_kw: Optional[str]) -> List[str]:
    if line_syllables(line) == target and ends_ok(line):
        return line
    return build_line_from_tokens(target, theme, maybe_kw)


# -----------------------------
# Main generation
# -----------------------------

def generate_best_haiku(user_text: str, n_candidates: int = 160) -> str:
    random.seed(hash(user_text) ^ random.getrandbits(32))

    tags = infer_tags(user_text)
    kws = expanded_keywords(user_text)
    theme = pick_theme(tags, kws)
    unknown_mode = prompt_is_mostly_unknown(kws)

    echo_kw = None
    if kws:
        known_first = [k for k in kws if is_known_word(k) and k.lower() not in FUNCTIONY]
        echo_kw = known_first[0] if known_first else kws[0]

    pool = phrase_pool(theme, tags, kws, unknown_mode)

    best: Optional[Tuple[List[str], List[str], List[str]]] = None
    best_score = -1e9

    def score(lines: Tuple[List[str], List[str], List[str]]) -> float:
        allw = [w.lower() for line in lines for w in line]
        uniq = set(allw)

        rep_pen = (len(allw) - len(uniq)) * 1.2

        kw_bonus = 0.0
        if echo_kw and echo_kw.lower() in uniq and not unknown_mode:
            kw_bonus = 2.2

        novelty_bonus = 0.03 * sum(1 for w in uniq if w not in RECENT_WORD_SET)

        return kw_bonus + novelty_bonus - rep_pen

    for _ in range(n_candidates):
        forbid: Set[str] = set()

        l1 = build_line_from_phrases(5, pool, forbid, tags, kws, max_phrases=2)
        if not l1:
            l1 = build_line_from_tokens(5, theme, None if unknown_mode else echo_kw)
        l1 = ensure_line_target(l1, 5, theme, None if unknown_mode else echo_kw)
        forbid |= {w.lower() for w in l1 if w.lower() not in STOPWORDS}

        l2 = build_line_from_phrases(7, pool, forbid, tags, kws, max_phrases=2)
        if not l2:
            l2 = build_line_from_tokens(7, theme, None if unknown_mode else echo_kw)
        l2 = ensure_line_target(l2, 7, theme, None if unknown_mode else echo_kw)
        forbid |= {w.lower() for w in l2 if w.lower() not in STOPWORDS}

        l3 = build_line_from_phrases(5, pool, forbid, tags, kws, max_phrases=2)
        if not l3:
            l3 = build_line_from_tokens(5, theme, echo_kw)
        l3 = ensure_line_target(l3, 5, theme, echo_kw)

        haiku = (l1, l2, l3)

        # final hard validation: if anything is off, force token rebuild
        if line_syllables(l1) != 5:
            l1 = build_line_from_tokens(5, theme, None if unknown_mode else echo_kw)
        if line_syllables(l2) != 7:
            l2 = build_line_from_tokens(7, theme, None if unknown_mode else echo_kw)
        if line_syllables(l3) != 5:
            l3 = build_line_from_tokens(5, theme, echo_kw)
        haiku = (l1, l2, l3)

        s = score(haiku)
        if s > best_score:
            best_score = s
            best = haiku

    assert best is not None

    record_phrase_hits(best, pool)
    record_usage([w for line in best for w in line])

    return format_haiku(best)


def repl() -> None:
    print("Welcome to Alexia's Haiku generator. Type something and see what happens, or 'quit'.\n")
    while True:
        try:
            user_text = input("prompt> ").strip()
        except EOFError:
            break
        if not user_text:
            continue
        if user_text.lower() in {"q", "quit", "exit"}:
            break
        print()
        print(generate_best_haiku(user_text))
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        print(generate_best_haiku(prompt))
    else:
        repl()