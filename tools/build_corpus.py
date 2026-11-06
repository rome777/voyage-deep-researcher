"""대항해시대 지명 코퍼스를 만든다: 영어 위키백과 문서의 'History' 절 (LLM 호출 없음).

    ..\\simulator\\.venv\\Scripts\\python tools\\build_corpus.py

수업과 같은 형식으로 data/corpus.json 을 쓴다.
    {"docs": {"한국어 지명": "History 절 본문"}, "links": {"한국어 지명": ["가리키는 다른 지명", ...]}}
지도·데모용 부가 정보(영어 제목, 좌표, 권역, 묶음, 출처 URL)는 data/places.json 에 둔다.

규칙
- 지명 표는 tools/places_table.py 하나에서 관리한다.
- 본문은 'History' 2단계 절 전체(하위 절 포함). 3,000자 미만이면 문서 전체를 쓴다.
- 링크는 그 문서 본문 링크 중 코퍼스 안의 지명만 남긴다 (prop=links + pltitles 로 한꺼번에 조회).
- 출처: 영어 위키백과, CC BY-SA 4.0. 출처 URL 을 places.json 에 남긴다.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from places_table import GROUPS, PLACES, REGIONS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "aiffel-deep-researcher-corpus/0.2 (course project)"}  # 아스키만
MIN_CHARS = 3000


def get(params: dict) -> dict:
    params = {"format": "json", "formatversion": "2", "redirects": "1", **params}
    for attempt in range(8):
        try:
            r = requests.get(API, params=params, headers=UA, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** min(attempt, 4))
            continue
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After") or 5 * (attempt + 1)))
            continue
        r.raise_for_status()
        time.sleep(1.0)
        return r.json()
    raise RuntimeError("위키백과 응답을 받지 못했습니다")


def history_section(text: str) -> str:
    m = re.search(r"^==\s*History\s*==\s*$", text, re.M)
    if not m:
        return ""
    rest = text[m.end():]
    nxt = re.search(r"^==[^=].*==\s*$", rest, re.M)
    return (rest[: nxt.start()] if nxt else rest).strip()


def clean(text: str) -> str:
    text = re.sub(r"^(=+)\s*(.*?)\s*\1\s*$", r"\2", text, flags=re.M)  # 하위 절 제목은 한 줄 글로
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main() -> None:
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    docs, meta, title_of = {}, {}, {}
    missing = []
    for name, title, lat, lon, region, group, inland in PLACES:
        d = get({"action": "query", "prop": "extracts", "explaintext": "1", "exsectionformat": "wiki", "titles": title})
        page = d["query"]["pages"][0]
        full = page.get("extract", "")
        if page.get("missing") or len(full) < 500:
            missing.append(f"{name} ({title})")
            print(f"  ✗ {name} ({title}): 문서 없음", file=sys.stderr)
            continue
        hist = history_section(full)
        source = "History 절"
        if len(hist) < MIN_CHARS:
            hist, source = full, "문서 전체 (History 절이 짧음)"
        body = clean(hist)
        if len(body) < MIN_CHARS:  # 문서 전체도 3,000자 미만인 토막글은 넣지 않는다 (나눠 읽을 이유가 없다)
            missing.append(f"{name} ({page['title']}, {len(body):,}자 토막글)")
            print(f"  ✗ {name}: {len(body):,}자 토막글이라 제외", file=sys.stderr)
            continue
        docs[name] = body
        title_of[name] = page["title"]
        meta[name] = {"title": page["title"], "url": "https://en.wikipedia.org/wiki/" + quote(page["title"].replace(" ", "_")),
                      "lat": lat, "lon": lon, "region": region, "group": group, "inland": inland, "source": source,
                      "chars": len(body), "tokens": len(enc.encode(body))}
        print(f"  ✓ {name} ({page['title']}): {len(body):,}자 · {source}", file=sys.stderr)

    # 링크: 문서 50개씩 × 대상 제목 50개씩 묶어 조회한다
    name_of = {t: n for n, t in title_of.items()}
    titles = list(name_of)
    links = {n: set() for n in docs}
    for i in range(0, len(titles), 50):
        for j in range(0, len(titles), 50):
            cont = {}
            while True:
                d = get({"action": "query", "prop": "links", "titles": "|".join(titles[i:i + 50]),
                         "pltitles": "|".join(titles[j:j + 50]), "pllimit": "max", **cont})
                for p in d["query"]["pages"]:
                    src = name_of.get(p["title"])
                    for l in p.get("links", []):
                        dst = name_of.get(l["title"])
                        if src and dst and dst != src:
                            links[src].add(dst)
                if "continue" not in d:
                    break
                cont = d["continue"]
    links = {k: sorted(v) for k, v in links.items()}

    (ROOT / "data" / "corpus.json").write_text(json.dumps({"docs": docs, "links": links}, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "data" / "places.json").write_text(json.dumps({"regions": REGIONS, "groups": GROUPS, "places": meta}, ensure_ascii=False, indent=1), encoding="utf-8")

    chars = sum(len(v) for v in docs.values())
    tokens = sum(m["tokens"] for m in meta.values())
    counts = sorted(len(v) for v in links.values())
    print(f"문서 {len(docs)}건 · {chars:,}자 · {tokens:,}토큰(o200k) · 컨텍스트 128,000 대비 {tokens/128000:.2f}배")
    print(f"가장 짧은 문서 {min(len(v) for v in docs.values()):,}자 · 링크 수 중앙값 {counts[len(counts)//2]}개 · 링크 총 {sum(counts)}개")
    print("링크가 0개인 문서:", [k for k, v in links.items() if not v])
    print("문서 전체를 쓴 곳:", [k for k, m in meta.items() if m["source"] != "History 절"])
    print("못 찾은 문서:", missing)


if __name__ == "__main__":
    main()
