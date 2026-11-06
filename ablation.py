"""소거 실험: 장치를 하나씩 끄고, 같은 예산의 단일 에이전트와도 견준다.

    .venv\\Scripts\\python ablation.py --fake                              # 0원으로 표 모양까지 점검
    .venv\\Scripts\\python ablation.py --env ..\\simulator\\.env            # 예상 비용만 보여 주고 멈춘다
    .venv\\Scripts\\python ablation.py --env ..\\simulator\\.env --yes      # 실제로 돌린다
    .venv\\Scripts\\python ablation.py --qids Q4 --only 전체,구역 끔 --env ..\\simulator\\.env --yes

조건 (질문마다 하나씩 돌린다)
    전체           스위치 전부 켬
    역할 끔        모든 함대가 '큰그림 담당' — 역할 프롬프트가 차이를 만드는가
    배정 끔        시작 문서 없이 함대가 스스로 첫 문서를 고름 — 배정이 겹침·헛읽기를 줄이는가
    구역 끔        남의 시작 문서·남이 읽은 문서를 피하지 않음 — 구역이 겹침을 줄이는가
    재위임 끔      부족 신고가 와도 2라운드를 돌리지 않음 — 재위임이 근거를 늘리는가
    단일 에이전트  baseline.py, '전체' 실행과 읽기 횟수가 같음 — 나눠 맡기는 것 자체가 이득인가

조건마다 --repeat 번(기본 2) 돌린다. 이미 같은 조건·같은 코드 판(graph.VERSION)으로 돌린 실행이 output/ 에 있으면
다시 부르지 않고 그 기록을 쓴다 (예산 아끼기).
--fresh 를 주면 전부 새로 돌린다. 결과: output/ablation.md · output/ablation.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

import baseline
import graph
import metrics
from graph import CFG, OUT, QUESTIONS

SWITCHES = list(CFG["switches"])
CONDITIONS = [("전체", None), ("역할 끔", "roles"), ("배정 끔", "assign"), ("구역 끔", "zones"), ("재위임 끔", "redelegate"),
              ("단일 에이전트", "single")]
DEFAULT_QIDS = ["Q3", "Q4", "Q6", "Q8"]  # 다갈래 둘 · 추적 · 비교 — 함대가 여럿 나가야 장치 차이가 드러난다
# 표에 싣는 지표: (열 이름, 뽑는 법, 높을수록 좋은가 — None 이면 방향 없음)
COLS = [("근거율", lambda m: m["근거율"], True), ("허위", lambda m: len(m["허위인용"]), False),
        ("엉뚱", lambda m: len(m["엉뚱한인용"]), False), ("읽은 문서", lambda m: m["읽은문서"], None),
        ("읽고 안 씀", lambda m: len(m["읽고안쓴"]), False), ("편중", lambda m: m["편중"], False),
        ("중복률", lambda m: m["중복률"], False), ("격리율", lambda m: m["격리율"], False),
        ("인용", lambda m: m["인용수"], None), ("자수", lambda m: m["보고서자수"], None),
        ("호출", lambda m: m["LLM호출"], None), ("₩", lambda m: m["원"], None)]


def rows(path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def switches_for(off: str | None) -> dict:
    return {k: k != off for k in SWITCHES}


def find_multi(qid: str, mode: str, sw: dict) -> list[dict]:
    """같은 질문·모델·스위치·예산·코드 판의 멀티 에이전트 실행 (오래된 것부터)."""
    return [r for r in rows(OUT / "runs.jsonl") if r.get("qid") == qid and r["mode"] == mode and r["switches"] == sw
            and r["sections"] == CFG["max_sections"] and r["budget"] == CFG["section_budget"] and r.get("agent") is None
            and r.get("version") == graph.VERSION]


def find_single(qid: str, mode: str, ref_id: str) -> dict | None:
    hit = [r for r in rows(OUT / "baseline.jsonl") if r.get("qid") == qid and r["mode"] == mode and r.get("reference") == ref_id
           and r.get("version") == graph.VERSION]
    return hit[-1] if hit else None


def estimate(qid: str, mode: str) -> float:
    """실제 호출 예상 비용: 같은 질문의 지난 실제 실행 평균, 없으면 ₩15 (함대 다섯 척 Q4 가 ₩15.6 이었다 — 넉넉히 잡는다)."""
    if mode == "fake":
        return 0.0
    same = [r["metrics"]["원"] for r in rows(OUT / "runs.jsonl") + rows(OUT / "baseline.jsonl")
            if r["mode"] == mode and r.get("qid") == qid and r["metrics"].get("원")]
    return st.mean(same) if same else 15.0


def plan_jobs(qids: list[str], only: list[str], mode: str, fresh: bool, repeat: int) -> list[tuple]:
    """(qid, 조건, 끈 스위치, 재사용할 기록 또는 None, 반복 번호) 목록.
    단일 에이전트 i 번째는 '전체' i 번째 실행과 짝을 지어 그 실행의 읽기 횟수를 예산으로 받는다."""
    jobs = []
    for qid in qids:
        fulls = [] if fresh else find_multi(qid, mode, switches_for(None))[-repeat:]
        for name, off in CONDITIONS:
            if only and name not in only:
                continue
            if off is None:
                olds = fulls
            elif off == "single":
                olds = [find_single(qid, mode, f["run_id"]) for f in fulls]
            else:
                olds = [] if fresh else find_multi(qid, mode, switches_for(off))[-repeat:]
            for i in range(repeat):
                jobs.append((qid, name, off, olds[i] if i < len(olds) else None, i))
    return jobs


def reads_of(r: dict) -> int:
    if r.get("agent"):
        return r["reads"]
    ev = rows(OUT / "runs" / f"{r['run_id']}.jsonl")
    return sum(e["type"] == "read" for e in ev)


def rescore(r: dict) -> dict:
    """저장된 보고서와 읽기 기록으로 지표를 다시 잰다 (LLM 호출 없음). metrics.py 를 고친 뒤에도 같은 잣대로 비교하려고.
    격리율 · 호출 수 · 비용은 실행 중에만 알 수 있으므로 기록된 값을 그대로 둔다."""
    rep = OUT / "reports" / f"{r['run_id']}.md"
    if not rep.exists():
        return r
    ev = rows(OUT / "runs" / f"{r['run_id']}.jsonl")
    reads = [(e.get("sec", 0), e["doc"]) for e in ev if e["type"] == "read"]
    m = metrics.evaluate(rep.read_text(encoding="utf-8"), reads, set(graph.DOCS))
    for k in ("격리율", "LLM호출", "원"):
        m[k] = r["metrics"].get(k, m.get(k))
    return {**r, "metrics": m}


def table(results: list[dict]) -> str:
    head = "| 조건 | " + " | ".join(c for c, _, _ in COLS) + " | 읽기 횟수 | 실행 |"
    sep = "|" + "---|" * (len(COLS) + 3)
    out = ["# 소거 실험", "",
           f"모델 `{results[0]['mode']}` · 절 상한 {CFG['max_sections']} · 절 예산 {CFG['section_budget']} · 라운드 상한 {CFG['max_rounds']}. "
           "정답표·판정 모델 없이 보고서와 읽은 문서 목록만으로 잰다 (metrics.py). "
           "허위·엉뚱 인용 경보가 울린 실행은 ⚠ 를 달고 평균에서 뺀다.", ""]
    for qid in dict.fromkeys(r["qid"] for r in results):
        q = next(x["question"] for x in QUESTIONS if x["id"] == qid)
        out += [f"## {qid} {q}", "", head, sep]
        for r in (x for x in results if x["qid"] == qid):
            m = r["metrics"]
            flag = " ⚠" if metrics.alarms(m) else ""
            out.append(f"| {r['condition']} #{r['rep']}{flag} | " + " | ".join(f"{f(m)}" for _, f, _ in COLS) + f" | {r['reads']} | `{r['run_id']}`{' (재사용)' if r['reused'] else ''} |")
        out.append("")
    # 질문 평균과 '전체' 대비 차이
    out += ["## 질문 평균 (전체 대비 차이)", "", "| 조건 | 실행 수 | " + " | ".join(c for c, _, _ in COLS) + " |", "|" + "---|" * (len(COLS) + 2)]
    means: dict[str, dict] = {}
    for name, _ in CONDITIONS:
        ok = [r for r in results if r["condition"] == name and not metrics.alarms(r["metrics"])]
        if ok:
            means[name] = {c: round(st.mean(f(r["metrics"]) for r in ok), 1) for c, f, _ in COLS} | {"_n": len(ok), "_all": sum(r["condition"] == name for r in results)}
    base = means.get("전체")
    for name, mv in means.items():
        cells = []
        for c, _, good in COLS:
            v = mv[c]
            if base and name != "전체":
                d = round(v - base[c], 1)
                mark = "" if good is None or d == 0 else (" ▲" if (d > 0) == good else " ▼")
                cells.append(f"{v} ({d:+g}{mark})")
            else:
                cells.append(f"{v}")
        out.append(f"| {name} | {mv['_n']}/{mv['_all']} | " + " | ".join(cells) + " |")
    out += ["", "▲ 는 전체보다 나아짐, ▼ 는 나빠짐 (방향이 정해진 지표만). 지표 뜻:", ""]
    out += [f"- **{k}**: {v[0]} — {v[1]} · {v[2]}" for k, v in metrics.INFO.items()]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qids", default=",".join(DEFAULT_QIDS), help="쉼표로 구분한 질문 번호")
    ap.add_argument("--only", default="", help="이 조건만 (쉼표 구분, 예: 전체,구역 끔)")
    ap.add_argument("--fake", action="store_true", help="가짜 모델 (0원)")
    ap.add_argument("--env", help="OPENAI_API_KEY 가 든 .env 경로")
    ap.add_argument("--fresh", action="store_true", help="지난 기록을 쓰지 않고 전부 새로 돌린다")
    ap.add_argument("--repeat", type=int, default=2, help="조건마다 몇 번 돌릴지 (한 번이면 집필 실수 하나가 장치 효과보다 커진다)")
    ap.add_argument("--yes", action="store_true", help="실제 호출일 때 예상 비용을 보고 진행에 동의")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    mode = "fake" if a.fake else CFG["model"]
    qids = [q.strip() for q in a.qids.split(",") if q.strip()]
    only = [o.strip() for o in a.only.split(",") if o.strip()]
    jobs = plan_jobs(qids, only, mode, a.fresh, a.repeat)
    todo = [j for j in jobs if j[3] is None]
    est = sum(estimate(j[0], mode) for j in todo)
    spent = json.loads((OUT / "spend.json").read_text(encoding="utf-8"))["total_won"] if (OUT / "spend.json").exists() else 0.0
    print(f"조건 {len(jobs)}개 중 새로 돌릴 것 {len(todo)}개 · 재사용 {len(jobs) - len(todo)}개 · 예상 ₩{est:.0f} "
          f"(누적 ₩{spent:.1f} / 상한 ₩{CFG['budget_won']})")
    if not a.fake and todo:
        if not a.yes:
            print("실제 호출입니다. 예상 비용을 확인했으면 --yes 를 붙여 다시 실행하세요.")
            return
        graph.load_key(a.env)
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY 가 없습니다. --env 로 .env 경로를 주세요.")
    results = []
    full_by_q: dict[tuple, dict] = {}
    for qid, name, off, old, i in jobs:
        q = next(x["question"] for x in QUESTIONS if x["id"] == qid)
        try:
            if old:
                r = old
            elif off == "single":
                full = full_by_q.get((qid, i))
                r = baseline.run_single(q, qid, a.fake, ref_id=full["run_id"] if full else None)
            else:
                r = graph.run(q, qid, a.fake, switches_for(off), CFG["max_sections"], CFG["section_budget"])
        except graph.BudgetExceeded as e:
            print(f"  {qid} {name}: 비용 상한에 걸려 멈춤 — {e}")
            break
        if off is None:
            full_by_q[(qid, i)] = r
        r = rescore(r)
        results.append({**r, "condition": name, "reads": reads_of(r), "reused": bool(old), "rep": i + 1})
        print(f"  {qid} {name:<8} #{i + 1} {'(재사용) ' if old else ''}{metrics.line(r['metrics'])} · ₩{r['metrics']['원']}")
    if not results:
        return
    md = table(results)
    (OUT / "ablation.md").write_text(md, encoding="utf-8")
    (OUT / "ablation.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ output/ablation.md · output/ablation.json ({len(results)}행)")


if __name__ == "__main__":
    main()
