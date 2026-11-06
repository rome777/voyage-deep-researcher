"""대항해시대 지명 문서 후보를 한국어 위키백과에서 실측한다 (LLM 호출 없음).

문서마다 본문 글자 수 · '역사' 절 글자 수 · 좌표 · 후보끼리의 링크를 재서
data/places_probe.json 에 저장하고, 요약을 출력한다.

    ..\\simulator\\.venv\\Scripts\\python tools\\probe_places.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

LANG = sys.argv[1] if len(sys.argv) > 1 else "ko"
API = f"https://{LANG}.wikipedia.org/w/api.php"
UA = {"User-Agent": "aiffel-deep-researcher-probe/0.1 (course project)"}  # 아스키만
OUT = Path(__file__).resolve().parent.parent / "data" / ("places_probe.json" if LANG == "ko" else f"places_probe_{LANG}.json")
HISTORY = {"ko": "역사", "en": "History"}[LANG]

CANDIDATES = [
    # 유럽
    "리스본", "세비야", "카디스", "팔로스데라프론테라", "암스테르담", "안트베르펜", "베네치아", "제노바", "세우타",
    # 대서양 · 아프리카
    "마데이라 제도", "카나리아 제도", "아소르스 제도", "카보베르데", "엘미나", "상투메 프린시페", "루안다",
    "희망봉", "모잠비크섬", "소팔라", "몸바사", "말린디", "잔지바르",
    # 인도양
    "호르무즈섬", "아덴", "코지코드", "고아주", "코치 (인도)", "콜롬보",
    # 동남아 · 동아시아
    "믈라카", "말루쿠 제도", "반다 제도", "테르나테섬", "자카르타", "마닐라", "세부섬", "마카오", "광저우",
    "나가사키시", "히라도시", "괌",
    # 아메리카
    "산살바도르섬", "히스파니올라섬", "산토도밍고", "아바나", "베라크루스", "멕시코시티", "아카풀코",
    "파나마시티", "카르타헤나 (콜롬비아)", "포토시", "쿠스코", "리마", "사우바도르 (바이아주)",
    "리우데자네이루", "마젤란 해협",
]
# 같은 후보의 영어 위키백과 제목 (한국어 표시 이름과 짝을 맞춘다)
EN = dict(zip(CANDIDATES, [
    "Lisbon", "Seville", "Cádiz", "Palos de la Frontera", "Amsterdam", "Antwerp", "Venice", "Genoa", "Ceuta",
    "Madeira", "Canary Islands", "Azores", "Cape Verde", "Elmina", "São Tomé and Príncipe", "Luanda",
    "Cape of Good Hope", "Island of Mozambique", "Sofala", "Mombasa", "Malindi", "Zanzibar",
    "Hormuz Island", "Aden", "Kozhikode", "Goa", "Kochi", "Colombo",
    "Malacca City", "Maluku Islands", "Banda Islands", "Ternate", "Jakarta", "Manila", "Cebu", "Macau", "Guangzhou",
    "Nagasaki", "Hirado, Nagasaki", "Guam",
    "San Salvador Island", "Hispaniola", "Santo Domingo", "Havana", "Veracruz (city)", "Mexico City", "Acapulco",
    "Panama City", "Cartagena, Colombia", "Potosí", "Cusco", "Lima", "Salvador, Bahia",
    "Rio de Janeiro", "Strait of Magellan",
]))
if LANG == "en":
    CANDIDATES = [EN[c] for c in CANDIDATES]
KO_NAME = {v: k for k, v in EN.items()}


def get(params: dict) -> dict:
    params = {"format": "json", "formatversion": "2", "redirects": "1", **params}
    for attempt in range(6):
        try:
            r = requests.get(API, params=params, headers=UA, timeout=30)
            if r.status_code == 429:  # 속도 제한 — Retry-After 만큼(없으면 점점 길게) 쉰다
                time.sleep(float(r.headers.get("Retry-After") or 5 * (attempt + 1)))
                continue
            r.raise_for_status()
            time.sleep(1.0)
            return r.json()
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def history_chars(text: str) -> int:
    """'== 역사 ==' 로 시작하는 2단계 절의 길이 (하위 절 포함)."""
    m = re.search(rf"^==\s*{HISTORY}\s*==\s*$", text, re.M)
    if not m:
        return 0
    rest = text[m.end():]
    nxt = re.search(r"^==[^=].*==\s*$", rest, re.M)
    return len(rest[: nxt.start()] if nxt else rest)


def links_of(title: str) -> list[str]:
    out, cont = [], {}
    while True:
        d = get({"action": "query", "prop": "links", "titles": title, "plnamespace": "0", "pllimit": "max", **cont})
        for p in d["query"]["pages"]:
            out += [l["title"] for l in p.get("links", [])]
        if "continue" not in d:
            return out
        cont = d["continue"]
        time.sleep(0.3)


def main() -> None:
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")  # gpt-4o-mini 계열 토크나이저
    docs, missing = {}, []
    for cand in CANDIDATES:
        d = get({"action": "query", "prop": "extracts|coordinates", "explaintext": "1",
                 "exsectionformat": "wiki", "titles": cand})
        page = d["query"]["pages"][0]
        time.sleep(0.3)
        if page.get("missing") or not page.get("extract"):
            missing.append(cand)
            print(f"  ✗ {cand}: 문서 없음", file=sys.stderr)
            continue
        title, text = page["title"], page["extract"]
        coord = (page.get("coordinates") or [{}])[0]
        docs[title] = {
            "asked": cand,
            "ko": KO_NAME.get(cand, cand),
            "chars": len(text),
            "tokens": len(enc.encode(text)),
            "history_chars": history_chars(text),
            "lat": coord.get("lat"),
            "lon": coord.get("lon"),
            "raw_links": links_of(title),
        }
        time.sleep(0.3)
        print(f"  ✓ {title}: {len(text):,}자 · 역사 {docs[title]['history_chars']:,}자", file=sys.stderr)

    titles = set(docs)
    for t, v in docs.items():
        v["links"] = sorted({l for l in v.pop("raw_links") if l in titles and l != t})

    total = sum(v["chars"] for v in docs.values())
    tokens = sum(v["tokens"] for v in docs.values())
    hist = sum(v["history_chars"] for v in docs.values())
    big = [t for t, v in docs.items() if v["chars"] >= 3000]
    nolink = [t for t, v in docs.items() if not v["links"]]
    nocoord = [t for t, v in docs.items() if v["lat"] is None]
    linkcounts = sorted(len(v["links"]) for v in docs.values())
    summary = {
        "문서수": len(docs), "3000자이상": len(big), "총글자": total, "총토큰_o200k": tokens,
        "역사절_총글자": hist, "링크수_중앙값": linkcounts[len(linkcounts) // 2] if linkcounts else 0,
        "링크0개": nolink, "좌표없음": nocoord, "없는후보": missing,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "docs": docs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
