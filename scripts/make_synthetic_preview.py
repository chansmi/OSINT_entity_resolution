#!/usr/bin/env python3
"""Generate the synthetic preview: the real record schema with fabricated values.

The released corpus names real people and is gated on Hugging Face. This preview
lets anyone inspect the format without access: every value is invented (names are
built from random syllables, ids and source datasets are prefixed "synthetic"),
so no record refers to a real person or organisation. Pairs mimic the patterns
of the real data: spelling and transliteration variants, sparse versus rich
records, and same-name negatives separated by a conflicting attribute.

    python scripts/make_synthetic_preview.py      # -> data/preview/synthetic_pairs.json
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "preview" / "synthetic_pairs.json"
SEED = 7

SYL = ["ka", "ro", "mi", "tel", "var", "do", "sen", "lu", "bri", "nak", "zo", "fen",
       "ta", "gor", "vel", "is", "mun", "dra", "pe", "lom", "su", "kir", "ban", "ost"]
CYR = {"ka": "ка", "ro": "ро", "mi": "ми", "tel": "тел", "var": "вар", "do": "до",
       "sen": "сен", "lu": "лу", "bri": "бри", "nak": "нак", "zo": "зо", "fen": "фен",
       "ta": "та", "gor": "гор", "vel": "вел", "is": "ис", "mun": "мун", "dra": "дра",
       "pe": "пе", "lom": "лом", "su": "су", "kir": "кир", "ban": "бан", "ost": "ост"}
COUNTRIES = ["zz", "xa", "xb", "xc"]  # user-assigned ISO codes, not real countries
TOPICS = ["sanction", "role.pep", "debarment"]
SUFFIX = ["Holdings", "Trading", "Logistics", "Group", "Industries"]


def word(rng, n=None):
    parts = [rng.choice(SYL) for _ in range(n or rng.randint(2, 3))]
    return parts


def latin(parts):
    return "".join(parts).capitalize()


def cyrillic(parts):
    s = "".join(CYR[p] for p in parts)
    return s[0].upper() + s[1:]


def date(rng):
    return f"{rng.randint(1945, 1995)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def stamp(rng):
    return f"20{rng.randint(22, 25)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:00:00"


def entity(rng, n, schema, caption, props, datasets):
    fs = stamp(rng)
    return {
        "id": f"synthetic-{n:04d}",
        "caption": caption,
        "schema": schema,
        "referents": [],
        "datasets": datasets,
        "first_seen": fs,
        "last_seen": "2025-12-01T00:00:00",
        "last_change": fs,
        "properties": props,
    }


def person_pair(rng, n, positive, variant):
    first, last = word(rng, 2), word(rng)
    name = f"{latin(first)} {latin(last)}"
    bd, nat, gender = date(rng), rng.choice(COUNTRIES), rng.choice(["male", "female"])
    left = entity(rng, n, "Person", name, {
        "name": [name], "firstName": [latin(first)], "lastName": [latin(last)],
        "birthDate": [bd], "nationality": [nat], "gender": [gender],
        "topics": [rng.choice(TOPICS)],
    }, ["synthetic_list_a"])
    if variant == "transliteration":
        rname = f"{cyrillic(first)} {cyrillic(last)}"
        rprops = {"name": [rname], "alias": [name.upper()], "birthDate": [bd], "nationality": [nat]}
    elif variant == "sparse":
        rname = f"{latin(last).upper()}, {latin(first)}"
        rprops = {"name": [rname]}
    else:  # spelling
        rname = f"{latin(first)} {latin(last)[:-1]}{rng.choice('aeiou')}"
        rprops = {"name": [rname], "birthDate": [bd], "gender": [gender]}
    if not positive:  # same-looking name, conflicting attribute
        other = date(rng)
        while other[:4] == bd[:4]:
            other = date(rng)
        rprops["birthDate"] = [other]
    rprops["topics"] = [rng.choice(TOPICS)]
    right = entity(rng, n + 1, "Person", rprops["name"][0], rprops, ["synthetic_list_b"])
    return left, right


def company_pair(rng, n, positive):
    base, suffix = latin(word(rng)), rng.choice(SUFFIX)
    name = f"{base} {suffix}"
    reg, juris = f"SYN{rng.randint(100000, 999999)}", rng.choice(COUNTRIES)
    left = entity(rng, n, "Company", name, {
        "name": [name], "jurisdiction": [juris], "registrationNumber": [reg],
        "incorporationDate": [date(rng)[:4]], "topics": ["sanction"],
    }, ["synthetic_list_a"])
    other = rng.choice([x for x in SUFFIX if x != suffix])
    rname = f"{base.upper()} {suffix.upper()}" if positive else f"{base} {other}"
    rprops = {"name": [rname], "jurisdiction": [juris],
              "registrationNumber": [reg if positive else f"SYN{rng.randint(100000, 999999)}"]}
    right = entity(rng, n + 1, "Company", rname, rprops, ["synthetic_list_c"])
    return left, right


def main():
    rng = random.Random(SEED)
    pairs, n = [], 1
    plan = ([("person", True, v) for v in ("spelling", "transliteration", "sparse")] * 5
            + [("person", False, v) for v in ("spelling", "transliteration", "sparse")] * 3
            + [("company", True, None)] * 6 + [("company", False, None)] * 5)
    for kind, positive, variant in plan:
        if kind == "person":
            left, right = person_pair(rng, n, positive, variant)
        else:
            left, right = company_pair(rng, n, positive)
        pairs.append({"left": left, "right": right,
                      "judgement": "positive" if positive else "negative", "synthetic": True})
        n += 2
    rng.shuffle(pairs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    meta = {"description": "Synthetic preview of OpenSanctions Pairs: real schema, fabricated values. "
                           "No record refers to a real person or organisation.",
            "generator": "scripts/make_synthetic_preview.py", "seed": SEED, "n_pairs": len(pairs)}
    OUT.write_text(json.dumps({"metadata": meta, "pairs": pairs}, indent=2, ensure_ascii=False) + "\n")
    pos = sum(p["judgement"] == "positive" for p in pairs)
    print(f"{len(pairs)} synthetic pairs ({pos} positive) -> {OUT}")


if __name__ == "__main__":
    main()
