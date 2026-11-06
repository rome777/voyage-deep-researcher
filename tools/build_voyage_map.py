"""해도 시뮬레이션 HTML 을 만든다: 템플릿 + 육지 윤곽 + 코퍼스(지명 문서) 수치.

    ..\\simulator\\.venv\\Scripts\\python tools\\build_voyage_map.py

- concepts/vendor/land-50m.json (world-atlas, Natural Earth) 을 위경도 고리로 풀어 싣는다
- data/corpus.json · data/places.json (tools/build_corpus.py 결과) 을 합친다
- concepts/voyage-map.src.html 의 __DATA__ 자리에 넣어 concepts/voyage-map.html 을 쓴다
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # 루트의 metrics.py


def decode_land(path: Path, step: float = 0.04) -> list[list[list[float]]]:
    """TopoJSON(양자화) → 위경도 고리 목록. 너무 촘촘한 점은 step 격자로 솎는다."""
    topo = json.loads(path.read_text(encoding="utf-8"))
    sx, sy = topo["transform"]["scale"]
    tx, ty = topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:
            x += dx
            y += dy
            pts.append((x * sx + tx, y * sy + ty))
        arcs.append(pts)

    def arc_pts(i: int) -> list[tuple[float, float]]:
        return arcs[i] if i >= 0 else arcs[~i][::-1]

    rings = []
    for geom in topo["objects"]["land"]["geometries"]:
        polys = geom["arcs"] if geom["type"] == "MultiPolygon" else [geom["arcs"]]
        for poly in polys:
            for ring in poly:
                pts: list[tuple[float, float]] = []
                for i in ring:
                    seg = arc_pts(i)
                    pts.extend(seg[1:] if pts else seg)
                # 날짜변경선을 넘는 고리(유라시아 동쪽 끝, 피지 등)는 경도를 이어 붙인 뒤
                # ±360° 옮긴 사본을 하나 더 둔다. 그래야 가로줄도 없고 대륙도 지도 안에 온전히 남는다.
                unwrapped, shift, prev = [], 0.0, None
                for lon, lat in pts:
                    if prev is not None and lon + shift - prev > 180:
                        shift -= 360
                    elif prev is not None and lon + shift - prev < -180:
                        shift += 360
                    prev = lon + shift
                    unwrapped.append((prev, lat))
                lons = [p[0] for p in unwrapped]
                copies = [0.0]
                if max(lons) > 180:
                    copies.append(-360.0)
                if min(lons) < -180:
                    copies.append(360.0)
                for off in copies:
                    out, last = [], None
                    for lon, lat in unwrapped:
                        q = (round((lon + off) / step) * step, round(lat / step) * step)
                        if q != last:
                            out.append([round(q[0], 2), round(q[1], 2)])
                            last = q
                    if len(out) >= 4:
                        rings.append(out)
    return rings


def excerpt(text: str, limit: int = 260) -> str:
    """보고서 예시에 붙일 원문 발췌: 앞에서부터 문장 단위로 limit 자 안쪽."""
    import re
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    while lines and len(lines[0]) < 60 and not lines[0].endswith((".", "!", "?")):  # 맨 앞의 소제목 줄은 뺀다
        lines.pop(0)
    flat = re.sub(r"\s+", " ", " ".join(lines)).strip()
    out = ""
    for s in re.split(r"(?<=[.!?])\s+", flat):
        if len(out) + len(s) > limit and out:
            break
        out = (out + " " + s).strip()
    return out[:limit + 40]


KEEP = {"run_start", "plan", "fix", "read", "write", "review", "synth", "evaluate", "run_end"}
OFF_NAME = {"roles": "역할", "assign": "배정", "zones": "구역", "redelegate": "재위임"}


def load_questions() -> list[dict]:
    qs = json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8"))["questions"]
    return [{"id": q["id"], "q": q["question"], "type": q.get("type", "")} for q in qs]


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def condition(start: dict) -> str:
    if start.get("agent"):
        return "단일 에이전트"
    off = [OFF_NAME.get(k, k) for k, v in (start.get("switches") or {}).items() if not v]
    return "전체" if not off else " · ".join(off) + " 끔"


def run_record(path: Path, summaries: dict, score=None) -> dict | None:
    """실행 기록 하나 → 해도 재생용. 중간에 멈춘 실행이면 None. 프롬프트·문서 원문·메모·원고는 싣지 않는다."""
    events = _rows(path)
    start = next((e for e in events if e["type"] == "run_start"), None)
    end = next((e for e in events if e["type"] == "run_end"), None)
    if not start or not end:
        return None
    report_path = ROOT / "output" / "reports" / f"{path.stem}.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    reads = [(e.get("sec", 0), e["doc"]) for e in events if e["type"] == "read"]
    ground = score(report, reads) if (score and report and reads) else None  # 지금의 metrics.py 규칙으로 다시 잰 근거율
    s = summaries.get(path.stem, {})
    stamp = path.stem[:15]
    return {"run_id": path.stem, "qid": start.get("qid"), "question": start["question"], "mode": start["mode"],
            "cond": condition(start), "single": bool(start.get("agent")), "version": s.get("version") or "이전",
            "time": f"{stamp[4:6]}/{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}", "won": end.get("won"), "ground": ground,
            "events": [{k: v for k, v in e.items() if k not in ("note", "body")} for e in events if e["type"] in KEEP],
            "report": report}


def load_runs() -> list[dict]:
    """graph.py · baseline.py 가 남긴 실제 LLM 실행 기록 전부 (최신순). 잘못된 것으로 판정해 둔 실행(output/invalid.jsonl)은 뺀다."""
    import metrics
    names = set(json.loads((ROOT / "data" / "corpus.json").read_text(encoding="utf-8"))["docs"])
    score = lambda report, reads: metrics.evaluate(report, reads, names)["근거율"]
    out = ROOT / "output"
    invalid = {r["run_id"] for r in _rows(out / "invalid.jsonl")}
    summaries = {r["run_id"]: r for r in _rows(out / "runs.jsonl") + _rows(out / "baseline.jsonl")}
    runs = []
    for path in sorted((out / "runs").glob("*-live*.jsonl"), reverse=True):
        if path.stem in invalid:
            continue
        rec = run_record(path, summaries, score)
        if rec:
            runs.append(rec)
    return runs


def main() -> None:
    corpus = json.loads((ROOT / "data" / "corpus.json").read_text(encoding="utf-8"))
    pj = json.loads((ROOT / "data" / "places.json").read_text(encoding="utf-8"))
    places = []
    for name, m in pj["places"].items():
        places.append({"id": name, "name": name, "title": m["title"], "url": m["url"], "lat": m["lat"], "lon": m["lon"],
                       "region": m["region"], "group": m["group"], "inland": m["inland"], "source": m["source"],
                       "chars": m["chars"], "tokens": m["tokens"], "links": corpus["links"].get(name, []),
                       "excerpt": excerpt(corpus["docs"][name])})
    counts = sorted(len(p["links"]) for p in places)
    summary = {"문서수": len(places), "3000자이상": sum(p["chars"] >= 3000 for p in places),
               "총글자": sum(p["chars"] for p in places), "총토큰_o200k": sum(p["tokens"] for p in places),
               "링크수_중앙값": counts[len(counts) // 2], "링크총수": sum(counts),
               "링크0개": [p["name"] for p in places if not p["links"]],
               "문서전체사용": [p["name"] for p in places if p["source"] != "History 절"], "출처": "History 절"}
    sea = json.loads((ROOT / "data" / "seagrid.json").read_text(encoding="utf-8"))  # tools/build_seagrid.mjs 결과
    data = {"corpus": summary, "regions": pj["regions"], "groups": pj["groups"], "places": places,
            "land": decode_land(ROOT / "concepts" / "vendor" / "land-50m.json"), "sea": sea,
            "questions": load_questions(), "runs": load_runs()}
    src = (ROOT / "concepts" / "voyage-map.src.html").read_text(encoding="utf-8")
    out = ROOT / "concepts" / "voyage-map.html"
    out.write_text(src.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    pts = sum(len(r) for r in data["land"])
    print(f"지명 {len(places)}곳 · 토큰 {summary['총토큰_o200k']:,} · 육지 고리 {len(data['land'])}개 · 점 {pts:,}개 → {out.name} ({out.stat().st_size/1024:,.0f} KB)")


if __name__ == "__main__":
    main()
