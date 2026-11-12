"""Every tunable weight and switch of the consultant in one place.

Defaults reproduce context-v1 (best platform score; best of four systems on the independent
holdout) with the reliability fixes. The variant tuned on the development scenarios (TOP_K=5,
PROMPT=v2) lost to it on the holdout and is kept as a switch only (docs/unified-v1.md).
Any value can be overridden without editing code: environment variable ET_<NAME>,
or override(...) inside tests and tuning scripts. A weight of 0 switches a component off.
Standard library only; no data, answers or product facts live here.
"""
import math
import os
from contextlib import contextmanager

DEFAULTS = {
    # --- lexical ranking (retrieval.py) ---
    'BM25_K1': 1.2,            # term-frequency saturation
    'BM25_B': 0.35,            # length normalisation
    'W_TITLE': 0.35,           # query word found in the section heading path
    'W_HISTORY': 0.4,          # earlier client turns (divided by age)
    'W_ASSISTANT': 0.0,        # earlier assistant turns as a search signal only
    'W_ALIAS': 0.35,           # colloquial synonyms
    'W_COLLOQUIAL': 0.0,       # general everyday wording -> document vocabulary (0.5 helps retrieval labels; answers not yet verified)
    'W_PREFIX': 0.65,          # inflection recovery for out-of-vocabulary words
    'W_INPUTS': 0.6,           # "what data do you need" intent words
    'B_INPUTS_TITLE': 8.0,     # section that lists required data for the active product
    'M_TOPIC_MATCH': 1.35,     # multiplier when section topic equals question topic
    'B_TOPIC_MATCH': 4.0,      # additive bonus per shared topic
    'M_TOPIC_MISMATCH': 0.4,   # multiplier for a section of another product
    'B_BODY_TOPIC': 2.0,       # untitled-topic section that mentions the topic in its body
    'B_MULTI_TOPIC': 5.0,      # section covering all topics of a two-topic question
    'B_STAGED': 8.0,           # phased-rollout question -> phased-rollout section
    'B_SUITABILITY': 4.0,      # "is it worth it" question -> limits section
    'B_SCOPE': 0.9,            # scope-and-limits reference for the active product
    # --- context assembly (context.py) ---
    'TOP_K': 12,              # sections given to the model (5 was better on tuning scenarios, worse on the holdout)
    'PLATFORM_TOP_K': 1,       # 1 = the platform's TOP_K variable sets the context size, as in context-v1
    'CONTEXT_CHARS': 40000,    # hard budget of the documents block
    'PRIMARY': 3,              # strongest sections always kept first
    'SUPPORT_EXTRAS': 2,       # limits/conditions sections added beyond TOP_K
    'LIMITS_INJECT': 0,        # 1 = always add the scope-and-limits section of the active product; 2 = also its "what must not be claimed" list
    'NUMBERS_INJECT': 0,       # 1 = add the numbers reference of the active product when the client gives a number
    'NUMBERS_MAX_CHARS': 3000, # skip a numbers table larger than this
    'MIN_SCORE_RATIO': 0.0,    # drop sections scoring below this share of the best one (0 = keep all)
    # --- optional retrieval stages from earlier versions (all measured worse on the platform) ---
    'USE_HIERARCHY': 0,        # hierarchy-v1: whole product document
    'USE_HYBRID': 0,           # bge-m3 embeddings; off: on real answers it broke more scenarios than it fixed
    'HYBRID_POLICY': 'weighted', # 'anchored' (anchored-v1), 'rrf' (hybrid-v1) or 'weighted' (score fusion with W_DENSE)
    'HYBRID_EXTRAS': 2,          # semantic additions in the anchored policy
    'M_DENSE_MISMATCH': 1.0,     # share of the dense score kept by a section of another product
    'EMBED_BUDGET': 240,         # seconds allowed for computing embeddings before falling back to lexical search
    'W_DENSE': 3.0,              # weight of embedding similarity in the weighted policy
    'USE_RERANKER': 0,
    'USE_CORRECTIONS': 0,      # corrections-v1: client's explicit corrections rewrite the search view
    # --- generation (solution.py) ---
    'PROMPT': 'full',         # 'full' = context-v1 instruction, 'short' = prompt-v1, 'v2' = prompt-v1 with question/example edits, 'v3' = 'full' with form edits, 'v4' = 'full' plus four checks against critical errors
    'NUMBER_CHECK': 0,         # 1 = add the arithmetic comparison of the client's quantities with document quantities (checks.py)
    'NUMBER_CHECK_SCOPE': 'dialog',  # 'dialog' = latest client turn naming each quantity; 'last' = quantities of the last client turn only
    'NUMBER_CHECK_VALUES': 3,  # nearest document values listed on each side of the client's quantity
    'ASSISTANT_CHECK': 0,      # 1 = when the dialogue has assistant turns, remind to verify them against the documents
    'USE_EXAMPLES': 1,         # few-shot form examples inside the system prompt
    'USE_REMINDER': 1,         # form reminder after the documents
    'DIALOG_FORMAT': 'raw',    # 'raw' = dialogue as received; 'structured' = history and the last client turn separated
    'LONG_WORDS': 0,          # regenerate once when the answer is longer than this (0 = never, as in context-v1; prompt-v1 used 95)
    'MAX_ATTEMPTS': 5,         # transport retries
    'REQUEST_TIMEOUT': 90,
    'THINKING': '',            # '' = model default; 'on'/'off' = ask the server to reason or not (dropped if the proxy refuses it)
}


def _coerce(name, raw):
    default = DEFAULTS[name]
    if isinstance(default, str):
        return str(raw).strip()
    value = type(default)(float(raw)) if isinstance(default, int) else float(raw)
    if not math.isfinite(value):
        raise ValueError('Invalid setting: ' + name)
    return value


_values = {}
# Legacy plain names of earlier versions still work. context-v1 took the context size from the
# platform's TOP_K variable (12 when it is absent); PLATFORM_TOP_K=1 keeps exactly that behaviour.
# Variants with their own tuned size are built with PLATFORM_TOP_K=0. ET_TOP_K always wins.
_env_aliases = {'USE_HIERARCHY': ('USE_HIERARCHY',), 'USE_HYBRID': ('USE_HYBRID',),
                'USE_RERANKER': ('USE_RERANKER',), 'TOP_K': ('TOP_K',)}


def reload():
    """Read defaults and environment overrides again (ET_<NAME>, plus legacy plain names)."""
    _values.clear()
    for name, default in DEFAULTS.items():
        raw = None
        aliases = _env_aliases.get(name, ())
        if name == 'TOP_K' and os.environ.get('ET_PLATFORM_TOP_K', str(DEFAULTS['PLATFORM_TOP_K'])).strip() == '0':
            aliases = ()
        for key in ('ET_' + name,) + aliases:
            candidate = os.environ.get(key, '').strip()
            if candidate and candidate != key:
                raw = candidate
                break
        try:
            _values[name] = default if raw is None else _coerce(name, raw)
        except ValueError:
            raise ValueError('Invalid setting: ' + name) from None


def get(name):
    return _values[name]


@contextmanager
def override(**changes):
    """Temporarily change weights (tuning, ablation, tests)."""
    saved = dict(_values)
    try:
        for name, value in changes.items():
            _values[name] = _coerce(name, value)
        yield
    finally:
        _values.clear()
        _values.update(saved)


def snapshot():
    return dict(_values)


reload()
