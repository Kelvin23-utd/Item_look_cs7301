#!/usr/bin/env python3
"""Shared code of the workstation VLM naming experiment (analysis/VLM_NAMING_PLAN.md).

Standard library only, so the same file runs on the Mac (preparation, report) and on the lab
workstation (requests to the model server). The design these functions implement is fixed in
analysis/notes/2026-11-17_vlm_naming_design.md; the prompts below are quoted there verbatim.

Contents: the prompts and answer schemas, request bodies for an OpenAI-compatible server, a
streaming HTTP client that times a request, a reader for the server's Prometheus metrics, name
normalisation and matching, and the statistics used in the report.
"""
import http.client
import json
import math
import os
import random
import re
import time

SEED = 20261117

# ----------------------------------------------------------------------------- models (aliases only)
# The repository names the models only by these aliases. Their exact identifiers (repository, pinned
# revision, download size, serving limits, the key of the first workstation runs) are in a private,
# git-ignored file: analysis/vlm_models.private.json (template: vlm_models.example.json), or the file
# named by FB_VLM_MODELS.

CONTESTANTS = ["model_a", "model_b", "model_c", "model_d"]
JUDGES = ["judge_a", "judge_b"]
PRIMARY_JUDGE = "judge_a"
PUBLIC_MODELS = {
    # alias: size class, licence, weights as served (generic), GPUs in the design
    "model_a": {"size": "about 4 B", "licence": "Apache-2.0", "precision": "16-bit", "gpus": 1},
    "model_b": {"size": "about 4–8 B", "licence": "Apache-2.0", "precision": "16-bit", "gpus": 1},
    "model_c": {"size": "about 9 B", "licence": "Apache-2.0", "precision": "16-bit", "gpus": 1},
    "model_d": {"size": "about 12 B", "licence": "Apache-2.0", "precision": "4-bit weights, 16-bit activations",
                "gpus": 1},
    "judge_a": {"size": "about 14 B", "licence": "Apache-2.0", "precision": "8-bit", "gpus": 1},
    "judge_b": {"size": "about 27 B", "licence": "Apache-2.0", "precision": "8-bit", "gpus": 2},
}
MODELS_FILE = "vlm_models.private.json"
PRIVATE_FIELDS = ("repo", "rev", "gb", "limit")
_PRIVATE = {}


def alias_label(key):
    """Display form of an alias (model_a -> Model A, judge_b -> Judge B)."""
    a, _, b = key.partition("_")
    return "%s %s" % (a.capitalize(), b.upper()) if b else key


def models_path():
    return os.environ.get("FB_VLM_MODELS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), MODELS_FILE)


def private_models(required=True):
    """alias -> private entry (repo, rev, gb, limit, ignore, old_key, ...). Read once per path. Without the
    file: SystemExit when required, else {}."""
    p = models_path()
    if p not in _PRIVATE:
        if not os.path.exists(p):
            if required:
                raise SystemExit("the private model file is missing: %s (copy analysis/vlm_models.example.json "
                                 "to analysis/vlm_models.private.json and fill in the identifiers)" % p)
            return {}
        with open(p) as f:
            d = json.load(f)["models"]
        for k in PUBLIC_MODELS:
            missing = [x for x in PRIVATE_FIELDS if x not in d.get(k, {})]
            if missing:
                raise SystemExit("%s: entry %s lacks %s" % (p, k, ", ".join(missing)))
        _PRIVATE[p] = d
    return _PRIVATE[p]


def canonical_key(key):
    """The alias of a model key; a former key (old_key in the private file) maps to its alias. Unknown
    keys are returned unchanged."""
    if key in PUBLIC_MODELS:
        return key
    for a, m in private_models(required=False).items():
        if m.get("old_key") and m["old_key"] == key:
            return a
    return key


def neutral_name(name):
    """A run folder name with any former key replaced by its alias (for ids and anything displayed)."""
    for a, m in private_models(required=False).items():
        if m.get("old_key") and m["old_key"] in name:
            name = name.replace(m["old_key"], a)
    return name


def run_keys(alias):
    """Every key a run folder of this model may carry: the alias and, if there is one, the former key."""
    old = private_models(required=False).get(alias, {}).get("old_key")
    return [alias] + ([old] if old and old != alias else [])

# ----------------------------------------------------------------------------- prompts

CATEGORIES = ["kitchenware", "food or drink", "electronics", "clothing or accessory", "bag or container",
              "paper, book or stationery", "toiletry or cleaning", "toy", "tool", "furniture or fixture",
              "decoration", "other"]
KINDS = ["one item", "part of an item", "several items", "furniture or fixture", "background or unclear"]
VERDICTS = ["right", "close", "wrong", "unclear"]

_CROP_INTRO = ("This photo comes from a phone app that remembers where things are in a home. It shows one "
               "object the app detected; it may be blurry or cut off at the edges.")
_CTX_INTRO = ("The first image is a close-up of one object detected by a phone app that remembers where things "
              "are in a home; it may be blurry or cut off at the edges. The second image is the whole camera "
              "view at that moment, with the same object inside a red box. Use the second image for context only.")
_NAME_TASK = ("Name the object{where} the way its owner would when asking \"where is my ...?\": a short everyday "
              "name of one to four words, specific enough to tell it apart from other things in a home (for "
              "example \"TV remote\", \"reading glasses\", \"car keys\"). Do not use brand names.")
_ANSWER = ("Answer in JSON with these fields: name; category (one of: %s); attributes (up to three short visible "
           "attributes such as colour, material or shape); visible_text (readable text on the object, or an "
           "empty string); kind (one of: %s)." % (", ".join(CATEGORIES), ", ".join(KINDS)))

NAME_PROMPT_CROP = "\n".join([_CROP_INTRO, _NAME_TASK.format(where=""), _ANSWER])
NAME_PROMPT_CTX = "\n".join([_CTX_INTRO, _NAME_TASK.format(where=" in the red box"), _ANSWER])

JUDGE_VIEW_CROP = "The image is a close-up of the object; it may be blurry or cut off at the edges."
JUDGE_VIEW_CTX = ("The first image is a close-up of the object; it may be blurry or cut off at the edges. The "
                  "second image is the whole camera view, with the object inside a red box.")
JUDGE_PROMPT = (
    "A phone app photographed an object in a home, and several systems proposed a name for it.\n"
    "{view}\n"
    "For each proposed name, judge whether it names the object correctly:\n"
    "- right: a correct name for this object that a person could use to find it;\n"
    "- close: related but not quite right (too general, a near miss, or it names a part, the contents or the "
    "container of the object);\n"
    "- wrong: it names a different kind of object;\n"
    "- unclear: the image is too unclear to tell.\n"
    "Proposed names:\n"
    "{names}\n"
    "Answer in JSON with one verdict for each name number.")

NAME_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "category": {"type": "string", "enum": CATEGORIES},
        "attributes": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "visible_text": {"type": "string"},
        "kind": {"type": "string", "enum": KINDS},
    },
    "required": ["name", "category", "attributes", "visible_text", "kind"],
    "additionalProperties": False,
}

VARIANTS = ["crop-full", "crop-320", "ctx-full", "ctx-320"]
PRIMARY_VARIANT = "crop-full"
SIZE_CAP = {"full": 960, "320": 320}
NAME_MAX_TOKENS = 200
JUDGE_MAX_TOKENS = 400


def judge_schema(n):
    keys = ["n%d" % (i + 1) for i in range(n)]
    return {"type": "object", "properties": {k: {"type": "string", "enum": VERDICTS} for k in keys},
            "required": keys, "additionalProperties": False}


def judge_text(names, with_frame):
    lines = "\n".join("n%d: %s" % (i + 1, n) for i, n in enumerate(names))
    return JUDGE_PROMPT.format(view=JUDGE_VIEW_CTX if with_frame else JUDGE_VIEW_CROP, names=lines)


def chat_body(model, text, images_b64, schema, max_tokens, stream=True, schema_name="answer"):
    """OpenAI chat-completions body: the images first, then the text; greedy; JSON schema output."""
    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}} for b in images_b64]
    content.append({"type": "text", "text": text})
    body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.0,
            "max_tokens": max_tokens, "seed": 0, "chat_template_kwargs": {"enable_thinking": False}}
    if schema is not None:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": schema_name, "schema": schema, "strict": True}}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def ping_body(model):
    return {"model": model, "messages": [{"role": "user", "content": "Reply with OK."}], "temperature": 0.0,
            "max_tokens": 1, "chat_template_kwargs": {"enable_thinking": False}}


# ----------------------------------------------------------------------------- HTTP client

def post_stream(host, port, path, body, timeout=600):
    """POST a JSON body to an OpenAI-compatible endpoint and time it.

    Streams when body["stream"] is true (server-sent events). Returns a dict: t_send (perf_counter
    when the request was written), t_first (first content token), t_last (last byte), text, usage,
    finish_reason, request_bytes, response_bytes, status, error.
    """
    raw = json.dumps(body).encode("utf-8")
    out = {"request_bytes": len(raw), "response_bytes": 0, "text": "", "usage": None, "finish_reason": None,
           "t_first": None, "status": None, "error": None}
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        t0 = time.perf_counter()
        conn.request("POST", path, body=raw, headers={"Content-Type": "application/json"})
        out["t_send"] = t0
        resp = conn.getresponse()
        out["status"] = resp.status
        if resp.status != 200:
            data = resp.read()
            out["response_bytes"] = len(data)
            out["error"] = data.decode("utf-8", "replace")[:2000]
            out["t_last"] = time.perf_counter()
            return out
        if not body.get("stream"):
            data = resp.read()
            out["t_last"] = time.perf_counter()
            out["response_bytes"] = len(data)
            d = json.loads(data)
            ch = (d.get("choices") or [{}])[0]
            out["text"] = (ch.get("message") or {}).get("content") or ""
            out["finish_reason"] = ch.get("finish_reason")
            out["usage"] = d.get("usage")
            out["t_first"] = out["t_last"]
            return out
        parts = []
        while True:
            line = resp.readline()
            if not line:
                break
            out["response_bytes"] += len(line)
            s = line.strip()
            if not s.startswith(b"data:"):
                continue
            payload = s[5:].strip()
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                out["usage"] = d["usage"]
            for ch in d.get("choices") or []:
                delta = (ch.get("delta") or {}).get("content")
                if delta:
                    if out["t_first"] is None:
                        out["t_first"] = time.perf_counter()
                    parts.append(delta)
                if ch.get("finish_reason"):
                    out["finish_reason"] = ch["finish_reason"]
        out["t_last"] = time.perf_counter()
        out["text"] = "".join(parts)
        return out
    except Exception as e:  # noqa: BLE001 - record and carry on; the caller counts failures
        out["error"] = "%s: %s" % (type(e).__name__, e)
        out.setdefault("t_send", time.perf_counter())
        out["t_last"] = time.perf_counter()
        return out
    finally:
        conn.close()


def http_get(host, port, path, timeout=10):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path)
        r = conn.getresponse()
        return r.status, r.read().decode("utf-8", "replace")
    except OSError as e:
        return None, str(e)
    finally:
        conn.close()


_PROM = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)\s*$")


def parse_prometheus(text, prefix="vllm:"):
    """Sum of every sample of each metric family whose name starts with `prefix` (labels summed)."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _PROM.match(line)
        if not m or not m.group(1).startswith(prefix):
            continue
        if "le=" in (m.group(2) or ""):
            continue              # histogram buckets: only _sum and _count are kept
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + v
    return out


def metric_deltas(before, after, suffixes=("_sum", "_count")):
    return {k: after[k] - before.get(k, 0.0) for k in after if k.endswith(suffixes)}


# ----------------------------------------------------------------------------- answers

def parse_answer(text, required=("name",)):
    """The JSON object in a model's answer, or None. Tolerates code fences and leading text."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        d = json.loads(s)
    except ValueError:
        i, j = s.find("{"), s.rfind("}")
        if i < 0 or j <= i:
            return None
        try:
            d = json.loads(s[i:j + 1])
        except ValueError:
            return None
    if not isinstance(d, dict) or any(k not in d for k in required):
        return None
    return d


def answer_name(ans):
    """The name of a parsed answer, or None when it is missing or empty."""
    if not isinstance(ans, dict):
        return None
    n = ans.get("name")
    if not isinstance(n, str):
        return None
    n = " ".join(n.split())
    return n or None


# ----------------------------------------------------------------------------- names

STOPWORDS = {"a", "an", "the", "my", "your", "some", "this", "that"}
_IRREGULAR = {"knives": "knife", "leaves": "leaf", "shelves": "shelf", "loaves": "loaf", "scarves": "scarf",
              "halves": "half", "wolves": "wolf", "men": "man", "women": "woman", "children": "child",
              "feet": "foot", "teeth": "tooth", "mice": "mouse", "geese": "goose", "people": "person",
              "dice": "die", "potatoes": "potato", "tomatoes": "tomato", "lenses": "lens"}
_NO_SINGULAR = {"glass", "glasses", "pants", "scissors", "tongs", "jeans", "shorts", "headphones", "earbuds",
                "chopsticks", "sunglasses", "goggles", "binoculars", "pliers", "tweezers", "clothes", "series",
                "news", "lens", "bus", "gas", "dress", "mattress", "canvas", "cactus", "virus", "iris", "chess",
                "this", "is", "was", "has", "its", "us", "yes", "plus", "hummus", "asparagus", "octopus"}
_ES_ENDINGS = ("ches", "shes", "sses", "xes", "zes")
REF_TYPO_FIXES = {"suger": "sugar"}
HUMAN_SYNONYMS = {
    "phone": ["phone", "smartphone", "cellphone", "mobile", "iphone"],
    "cup": ["cup", "mug", "teacup"],
    "chip": ["chip", "crisp"],
}
PLACEHOLDER_NAME = re.compile(r"^\s*(item\s*\d+|object)\s*$", re.I)
HEAD_CUT = (" of ", " for ", " with ", " in ", " on ")


def singular(w):
    """Simple English singular of one lower-case word (good enough for object names)."""
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if w in _NO_SINGULAR or len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(_ES_ENDINGS):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def name_words(name):
    """Normalised words: lower case, non-letters to spaces, stop words dropped, singular."""
    if not name:
        return []
    s = re.sub(r"[^a-z]+", " ", name.lower())
    return [singular(w) for w in s.split() if w not in STOPWORDS]


def normalize(name):
    return " ".join(name_words(name))


def head_word(name):
    """Last normalised word of the name after cutting it at ' of ', ' for ', ... ('bag of chips' -> bag)."""
    if not name:
        return None
    s = " " + re.sub(r"[^a-z]+", " ", name.lower()) + " "
    cut = len(s)
    for c in HEAD_CUT:
        i = s.find(c, 1)
        if 0 < i < cut:
            cut = i
    ws = name_words(s[:cut])
    if not ws:
        ws = name_words(name)
    return ws[-1] if ws else None


def same_name(a, b):
    """Label-free agreement: equal normalised names or equal head words. False when either is None."""
    if not a or not b:
        return False
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    return na == nb or head_word(a) == head_word(b)


def reference_name(typed):
    """The typed name as used for scoring (typos fixed), or None for placeholders."""
    if not typed or PLACEHOLDER_NAME.match(typed):
        return None
    ws = [REF_TYPO_FIXES.get(w, w) for w in re.sub(r"[^a-z]+", " ", typed.lower()).split()]
    return " ".join(ws)


def _contains(seq, sub):
    n = len(sub)
    return n > 0 and any(seq[i:i + n] == sub for i in range(len(seq) - n + 1))


def human_match(candidate, reference):
    """True when the words of the reference, or of one of its synonyms, appear consecutively in the
    candidate (both normalised). `reference` is the scoring form (reference_name)."""
    if not candidate or not reference:
        return False
    cw = name_words(candidate)
    ref = normalize(reference)
    alts = HUMAN_SYNONYMS.get(ref, [ref])
    return any(_contains(cw, name_words(alt)) for alt in alts)


# ----------------------------------------------------------------------------- judge pool

def item_rng(item_id, seed=SEED):
    return random.Random("%s:%s" % (seed, item_id))


def build_pool(item_id, names):
    """Distinct normalised names in a per-item shuffled order: list of (normalised, shown text).

    The shown text is the first spelling seen, lower case, whitespace collapsed; the order of
    `names` does not change the result (sorted before shuffling)."""
    first = {}
    for n in names:
        if not n:
            continue
        k = normalize(n)
        if not k:
            continue
        shown = " ".join(n.lower().split())
        if k not in first or shown < first[k]:
            first[k] = shown
    keys = sorted(first)
    item_rng(item_id).shuffle(keys)
    return [(k, first[k]) for k in keys]


def pool_names(extra_names, named_runs):
    """item -> every proposed name: the non-contestant names (baselines and typed names, from
    pool_inputs.json) plus the sequential-phase names of each primary contestant run. `named_runs`:
    run id -> {item: [names]}. Items outside extra_names (which lists every R and S item) are ignored."""
    names = {k: list(v) for k, v in extra_names.items()}
    for _, by_item in sorted(named_runs.items()):
        for item, ns in by_item.items():
            if item in names:
                names[item].extend(ns)
    return names


def judge_pool(manifest, names_by_item):
    """The blind judging pool: for every R and S item of the manifest (sorted by id), the deduplicated
    names in a per-item shuffled order, and the images the judge sees (crop and marked keyframe when
    there is one, else the crop). `names_by_item`: item -> every proposed name, in any order."""
    out = []
    for it in sorted((i for i in manifest["items"] if i["set"] in ("R", "S")), key=lambda i: i["id"]):
        with_frame = "ctx-full" in it["variants"]
        entries = build_pool(it["id"], names_by_item.get(it["id"], []))
        out.append({"item": it["id"], "with_frame": with_frame,
                    "images": it["variants"]["ctx-full" if with_frame else "crop-full"],
                    "names": [s for _, s in entries]})
    return out


def parse_verdicts(text, n):
    """{index 0..n-1: verdict} from a judge answer; missing or invalid entries are left out."""
    d = parse_answer(text, required=())
    out = {}
    if not d:
        return out
    for i in range(n):
        v = d.get("n%d" % (i + 1))
        if isinstance(v, str) and v.strip().lower() in VERDICTS:
            out[i] = v.strip().lower()
    return out


# ----------------------------------------------------------------------------- statistics

def wilson(k, n, z=1.959963984540054):
    """Wilson score interval (lo, hi) for k successes out of n; (nan, nan) when n = 0."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2 * tail)


def cohen_kappa(a, b):
    """Cohen's kappa of two equal-length label lists; nan when expected agreement is 1."""
    if len(a) != len(b) or not a:
        return float("nan")
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return float("nan") if pe >= 1 else (po - pe) / (1 - pe)


def quantile(xs, q):
    """Linear-interpolation quantile (numpy's default) of a non-empty list."""
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def median(xs):
    return quantile(xs, 0.5)


def energy_trapezoid(ts, watts, t0=None, t1=None):
    """Joules from power samples (seconds, watts) between t0 and t1 by the trapezoid rule; the ends are
    interpolated. Returns nan with fewer than two samples."""
    pts = sorted(zip(ts, watts))
    if len(pts) < 2:
        return float("nan")
    t0 = pts[0][0] if t0 is None else t0
    t1 = pts[-1][0] if t1 is None else t1
    if t1 <= t0:
        return 0.0

    def at(t):
        if t <= pts[0][0]:
            return pts[0][1]
        if t >= pts[-1][0]:
            return pts[-1][1]
        for (ta, wa), (tb, wb) in zip(pts, pts[1:]):
            if ta <= t <= tb:
                return wa if tb == ta else wa + (wb - wa) * (t - ta) / (tb - ta)
        return pts[-1][1]

    inner = [(t, w) for t, w in pts if t0 < t < t1]
    seq = [(t0, at(t0))] + inner + [(t1, at(t1))]
    return sum((tb - ta) * (wa + wb) / 2 for (ta, wa), (tb, wb) in zip(seq, seq[1:]))


def share(flags):
    """(k, n) of a list of booleans, None entries left out."""
    v = [bool(f) for f in flags if f is not None]
    return sum(v), len(v)
