"""사람 비교용 화면 데이터: 같은 질문의 보고서를 나란히 놓고, 인용을 원문과 대조할 재료를 만든다.

    .venv\\Scripts\\python tools\\build_compare.py

- output/ablation.json 의 실행(질문 × 조건 × 반복)마다 보고서를 절·문장으로 나누고,
  절마다 어느 함대가 무엇을 읽었는지(관련 없음 판정 포함) 실행 기록에서 붙인다.
- 문장마다 연도(1000~1999)를 뽑아, 인용한 문서의 '함대가 실제로 읽은 범위'(read_cap_chars) 안에 그 연도가 있는지 본다.
  없으면 '의심'으로 표시하고, 같은 실행에서 읽은 다른 문서에 그 연도가 있으면 그 문서 이름을 단서로 단다.
  → 이건 판정이 아니라 사람이 원문을 펼쳐 볼 곳을 좁혀 주는 단서다. 한글로 옮긴 고유명사는 대조하지 못한다.
- concepts/compare.src.html 의 __DATA__ 자리에 넣어 concepts/compare.html 을 쓴다.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import metrics  # noqa: E402

CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
CORPUS = json.loads((ROOT / "data" / "corpus.json").read_text(encoding="utf-8"))
PLACES = json.loads((ROOT / "data" / "places.json").read_text(encoding="utf-8"))["places"]
QUESTIONS = {q["id"]: q for q in json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8"))["questions"]}
OUT = ROOT / "output"
YEAR = re.compile(r"(?<!\d)(1\d{3})(?!\d)")
LATIN = re.compile(r"[A-Za-z][A-Za-z'\-]{3,}")
# 나라 단서: 문장이 나라를 말하는데 인용 문서가 그 나라를 거의 언급하지 않으면(2회 미만) 오귀속 후보 (예: «마닐라»를 네덜란드 항구라고 씀)
NATIONS = [("포르투갈", r"포르투갈", r"Portug"), ("스페인", r"스페인|에스파냐", r"Spani|Spain|Castil"),
           ("네덜란드", r"네덜란드|VOC", r"Dutch|VOC|Netherlands"), ("영국", r"영국|잉글랜드", r"English|British|Britain|England"),
           ("프랑스", r"프랑스", r"French|France")]


def doc_sents(name: str) -> list[str]:
    text = CORPUS["docs"][name][: CFG["read_cap_chars"]]
    flat = re.sub(r"\[\d+\]", "", re.sub(r"\s+", " ", text))
    return [s for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]


SENTS: dict[str, list[str]] = {}


def sents_of(name: str) -> list[str]:
    if name not in SENTS:
        SENTS[name] = doc_sents(name)
    return SENTS[name]


def check(sentence: str, read_docs: list[str]) -> dict:
    """문장 하나: 인용 · 연도 · 원문 근거 문장 · 의심 단서."""
    cites = [c for c in metrics.cited(sentence) if c in CORPUS["docs"]]
    years = sorted(set(YEAR.findall(sentence)))
    latin = [w for w in LATIN.findall(sentence) if w.upper() != w or len(w) > 4]
    out = {"t": sentence, "c": cites, "y": years}
    if not cites:
        return out
    ev = []
    for d in cites:
        ss = sents_of(d)
        scored = []
        for i, s in enumerate(ss):
            sc = sum(2 for y in years if y in s) + sum(1 for w in latin if w.lower() in s.lower())
            if sc:
                scored.append((sc, i))
        for sc, i in sorted(scored, reverse=True)[:2]:
            ev.append([d, i])
    out["e"] = ev
    text_of = lambda d: " ".join(sents_of(d))
    miss = [y for y in years if not any(y in text_of(d) for d in cites)]
    weak = [k for k, ko, en in NATIONS if re.search(ko, sentence) and all(len(re.findall(en, text_of(d))) < 2 for d in cites)]
    if weak:
        out["n"] = weak
    if miss:
        out["m"] = miss
        alt = sorted({d for d in read_docs if d not in cites and any(y in text_of(d) for y in miss)})
        if alt:
            out["a"] = alt
    return out


def split_report(md: str) -> tuple[str, list[str], list[dict], list[str]]:
    """제목 · 머리말 블록 · 절(제목, 본문) · 맺음말 블록 — metrics.body_text 와 같은 규칙."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", md.strip()) if b.strip()]
    title = next((b[2:].strip() for b in blocks if b.startswith("# ")), "")
    blocks = [b for b in blocks if not b.startswith("# ")]
    heads = [i for i, b in enumerate(blocks) if b.startswith("## ")]
    if not heads:
        return title, [], [{"h": "", "body": "\n".join(blocks)}], []
    intro = blocks[: heads[0]]
    last = len(blocks) - 1
    outro = [blocks[last]] if last > heads[-1] and not blocks[last].startswith("#") else []
    body_blocks = blocks[heads[0]: last if outro else len(blocks)]
    secs: list[dict] = []
    for b in body_blocks:
        if b.startswith("## "):
            head, _, rest = b.partition("\n")
            secs.append({"h": head[3:].strip(), "body": rest.strip()})
        elif secs:
            secs[-1]["body"] += "\n" + b
    return title, intro, secs, outro


def build_run(r: dict) -> dict:
    rid = r["run_id"]
    ev = [json.loads(l) for l in (OUT / "runs" / f"{rid}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    md = (OUT / "reports" / f"{rid}.md").read_text(encoding="utf-8")
    plan = next((e for e in ev if e["type"] == "plan"), {})
    toc = plan.get("toc") or []
    reads = [e for e in ev if e["type"] == "read"]
    read_docs = list(dict.fromkeys(e["doc"] for e in reads))
    title, intro, secs, outro = split_report(md)
    single = bool(r.get("agent"))
    out_secs = []
    for s in secs:
        # 절 제목 '1. 네덜란드의 …' → 목차 번호 찾기 (번호 떼고 제목으로 맞춘다)
        name = re.sub(r"^\d+\.\s*", "", s["h"])
        k = next((i for i, t in enumerate(toc) if t.get("절") == name), None)
        if single:
            k = None
        sec_reads = [{"d": e["doc"], "x": bool(e.get("irrelevant")), "r": e.get("round", 1), "a": bool(e.get("assigned"))}
                     for e in reads if (k is not None and e.get("sec") == k)]
        recite = [{"b": e["before"], "f": e["after"], "k": e["kept"], "r": e.get("round", 1)} for e in ev if e["type"] == "recite" and k is not None and e.get("sec") == k]
        writes = [{"r": e.get("round", 1), "n": e.get("chars"), "c": e.get("cites"), "ok": e.get("enough"), "gap": e.get("gap", "")}
                  for e in ev if e["type"] == "write" and k is not None and e.get("sec") == k]
        out_secs.append({"h": s["h"], "k": k, "role": toc[k].get("역할") if k is not None else "", "brief": toc[k].get("지시") if k is not None else "",
                         "seed": toc[k].get("시작문서") if k is not None else "", "reads": sec_reads, "recite": recite, "writes": writes,
                         "s": [check(x, read_docs) for x in metrics.split_sentences(s["body"])]})
    m = r["metrics"]
    return {"id": rid, "qid": r["qid"], "cond": r["condition"], "rep": r["rep"], "single": single, "title": title,
            "intro": metrics.split_sentences("\n".join(intro)), "outro": metrics.split_sentences("\n".join(outro)), "secs": out_secs,
            "reads": [{"d": e["doc"], "x": bool(e.get("irrelevant"))} for e in reads] if single else [],
            "review": [{"round": e["round"], "gaps": e["gaps"], "reason": e.get("reason", "")} for e in ev if e["type"] == "review"],
            "fix": [{"what": e.get("what"), "given": e.get("given"), "used": e.get("used")} for e in ev if e["type"] == "fix"],
            "m": {"근거율": m["근거율"], "허위": len(m["허위인용"]), "엉뚱": len(m["엉뚱한인용"]), "읽은문서": m["읽은문서"],
                  "읽고안씀": len(m["읽고안쓴"]), "편중": m["편중"], "중복률": m["중복률"], "격리율": m["격리율"],
                  "호출": m["LLM호출"], "원": m["원"], "문장": m["문장수"], "자수": m["보고서자수"]}}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    results = json.loads((OUT / "ablation.json").read_text(encoding="utf-8"))
    runs = [build_run(r) for r in results]
    used = sorted({c for run in runs for s in run["secs"] for x in s["s"] for c in x["c"]}
                  | {rd["d"] for run in runs for s in run["secs"] for rd in s["reads"]} | {rd["d"] for run in runs for rd in run["reads"]})
    docs = {d: {"title": PLACES[d]["title"], "url": PLACES[d]["url"], "s": sents_of(d)} for d in used}
    qids = list(dict.fromkeys(r["qid"] for r in runs))
    data = {"questions": [{"id": q, "q": QUESTIONS[q]["question"], "type": QUESTIONS[q].get("type", ""), "why": QUESTIONS[q].get("why", "")} for q in qids],
            "runs": runs, "docs": docs, "cap": CFG["read_cap_chars"], "version": results[0].get("version", "")}
    src = (ROOT / "concepts" / "compare.src.html").read_text(encoding="utf-8")
    out = ROOT / "concepts" / "compare.html"
    out.write_text(src.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    flagged = sum(1 for run in runs for s in run["secs"] for x in s["s"] if x.get("m"))
    alt = sum(1 for run in runs for s in run["secs"] for x in s["s"] if x.get("a"))
    nat = sum(1 for run in runs for s in run["secs"] for x in s["s"] if x.get("n"))
    total = sum(len(s["s"]) for run in runs for s in run["secs"])
    print(f"실행 {len(runs)}개 · 본문 문장 {total} · 연도 의심 {flagged} (그중 다른 읽은 문서에 있음 {alt}) · 나라 의심 {nat} · 원문 {len(docs)}건 → {out.name} ({out.stat().st_size/1024:,.0f} KB)")


if __name__ == "__main__":
    main()
