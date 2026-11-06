"""단일 에이전트 기준선: 코디네이터·서브에이전트 없이 에이전트 하나가 같은 읽기 예산으로 끝까지 한다.

    .venv\\Scripts\\python baseline.py --qid Q4 --fake
    .venv\\Scripts\\python baseline.py --qid Q4 --env ..\\simulator\\.env     # 실제 호출
    .venv\\Scripts\\python baseline.py --qid Q4 --budget 12 --fake          # 예산을 직접 줄 때

같은 예산이란 '문서를 읽는 호출 수'가 같다는 뜻이다. --budget 을 안 주면 같은 질문·같은 모델로 돌린
멀티 에이전트 실행(스위치 전부 켬) 가운데 가장 최근 것을 찾아, 그 실행이 실제로 읽은 횟수(중복 포함)를 그대로 쓴다.
기준선은 받은 예산을 끝까지 쓴다. 모델이 그만 읽겠다고 하거나 엉뚱한 이름을 대면 코드가 다음 문서를 골라 준다
(예산을 덜 쓰고 진 기준선은 비교가 안 되므로).

멀티 에이전트와 다른 점은 셋뿐이다.
- 목차·배정이 없다: 에이전트가 문서 카드를 보고 첫 문서를 스스로 고른다.
- 컨텍스트가 하나다: 고르기·읽기·쓰기를 모두 같은 에이전트가 하므로 격리율이 100% 가 된다.
- 쓰는 사람이 하나다: 모은 메모 전부로 보고서 전체를 한 번에 쓴다.
읽기 프롬프트·지표·비용 상한은 graph.py 와 똑같은 것을 쓴다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import metrics
from graph import (CFG, DOCS, LINKS, OUT, QUESTIONS, BudgetExceeded, FakeModel, Meter, OpenAIModel,
                   VERSION, CARD_HEAD, Recorder, append_line, cards, in_scope_hint, jload, load_key, new_run_id, read_prompt, recite_hint, uncited)

AGENT = "단일 에이전트"


# ───────────────────────── 프롬프트 ─────────────────────────
def start_prompt(q: str) -> tuple[str, str]:
    return ("너는 대항해시대 질문에 혼자 답하는 리서처다. 문서 카드를 보고 가장 먼저 읽을 문서를 하나 골라라. "
            "카드는 각 문서에서 뽑은 문장 하나뿐이고 본문에는 대항해시대 이야기가 훨씬 많다. 질문이 대항해시대의 항해·무역·식민이나 "
            "이 지명들과 아예 무관할 때만 문서를 null 로 두고 '범위밖'에 이유를 한 문장으로 적어라.\n"
            'JSON으로만: {"문서":"목록의 한국어 지명","범위밖":""}',
            f"{CARD_HEAD}{cards()}\n\n[질문] {q}")  # 카드(매번 같음)를 앞에 → 캐시에 맞아 반값


def next_prompt(q: str, notes: list[tuple[str, str]], cand: list[str], left: int) -> tuple[str, str]:
    mat = "\n".join(f"- {d}: {n[:240]}" for d, n in notes)
    return ("너는 대항해시대 질문에 혼자 답하는 리서처다. 지금까지의 메모를 보고, 질문에 답하는 데 아직 빈 곳을 채울 문서를 후보에서 정확히 하나 골라라.\n"
            'JSON으로만: {"문서":"후보에 있는 지명"}',
            f"[질문] {q}\n[남은 읽기 횟수] {left}\n[지금까지의 메모]\n{mat}\n[후보] {', '.join(cand[:CFG['max_candidates']])}")


def report_prompt(q: str, notes: list[tuple[str, str]], sections: int) -> tuple[str, str]:
    mat = "\n\n".join(f"[{d}]\n{n}" for d, n in notes) or "(읽은 자료 없음)"
    return (f"너는 대항해시대 질문에 혼자 답하는 리서처다. 아래 자료만 근거로 한국어 보고서를 마크다운으로 써라.\n"
            f"첫 줄은 '# 보고서 제목', 그다음 머리말 한두 문장, 그다음 '## 절 제목' 으로 시작하는 절 {sections}개, 마지막에 맺음말 한두 문장.\n"
            "절마다 여섯 문장 이상 열두 문장 이하, 600자 이상.\n"
            "본문 문장마다 끝에 근거 문서의 지명을 «지명» 으로 붙여라 (자료에 대괄호로 적힌 이름 그대로). 자료에 없는 내용은 쓰지 마라. "
            "머리말·맺음말에는 새 사실도 인용도 넣지 마라.\n"
            "마크다운 본문만 답하라.",
            f"[질문] {q}\n[모은 자료]\n{mat}")


class FakeSingle(FakeModel):
    """가짜 모델에 단일 에이전트용 답 두 가지(start · report)를 더한다."""

    def __call__(self, kind: str, system: str, user: str, coord: bool, info: dict) -> str:
        if kind == "start":
            self.meter.add(coord, len(system) + len(user), (len(system) + len(user)) // 3, 30)
            q = info["question"]
            hit = [n for n in DOCS if n in q]
            return json.dumps({"문서": hit[0] if hit else max(DOCS, key=lambda n: len(LINKS.get(n, [])))}, ensure_ascii=False)
        if kind == "report":
            self.meter.add(coord, len(system) + len(user), (len(system) + len(user)) // 3, 600)
            body = "\n".join(f"«{d}» 문서에 따르면 {n[:80]}… «{d}»" for d, n in info["notes"])
            return f"# {info['question']}\n\n이 보고서는 한 에이전트가 혼자 조사한 결과다.\n\n## 조사 결과\n{body}\n\n각 문장의 근거는 «»로 표시한 문서다."
        return super().__call__(kind, system, user, coord, info)


# ───────────────────────── 같은 예산 찾기 ─────────────────────────
def reference(qid: str | None, question: str, mode: str, run_id: str | None = None) -> dict | None:
    """짝지을 멀티 에이전트 실행. run_id 를 주면 그 실행, 아니면 같은 질문·같은 모델·스위치 전부 켠 실행 중 가장 최근 것."""
    path = OUT / "runs.jsonl"
    if not path.exists():
        return None
    best = None
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if run_id:
            if r["run_id"] == run_id:
                best = r
            continue
        if r.get("agent") == AGENT or r["mode"] != mode or not all(r["switches"].values()):
            continue
        if (qid and r.get("qid") == qid) or (not qid and r["question"] == question):
            best = r
    if not best:
        return None
    ev = [json.loads(l) for l in (OUT / "runs" / f"{best['run_id']}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    plan = next((e for e in ev if e["type"] == "plan"), {})
    return {"run_id": best["run_id"], "reads": sum(e["type"] == "read" for e in ev), "sections": len(plan.get("toc") or []),
            "metrics": best["metrics"]}


# ───────────────────────── 실행 ─────────────────────────
def run_single(question: str, qid: str | None, fake: bool, budget: int | None = None, sections: int | None = None,
               ref_id: str | None = None) -> dict:
    mode = "fake" if fake else CFG["model"]
    ref = reference(qid, question, mode, ref_id)
    if budget is None:
        budget = ref["reads"] if ref and ref["reads"] else CFG["max_sections"] * CFG["section_budget"]
    sections = sections or (ref["sections"] if ref and ref["sections"] else 3)
    run_id = new_run_id(("-fake" if fake else "-live") + (f"-{qid}" if qid else "") + "-single")
    meter = Meter(real=not fake)
    rec = Recorder(run_id)
    model = FakeSingle(meter) if fake else OpenAIModel(meter)
    ask = lambda kind, p, **info: model(kind, p[0], p[1], True, info)  # 컨텍스트가 하나뿐이므로 모든 호출이 '본부' 쪽이다
    rec("run_start", run_id=run_id, qid=qid, question=question, mode=mode, agent=AGENT, budget=budget,
        sections=sections, reference=ref["run_id"] if ref else None)
    t0 = time.time()
    reads: list[str] = []
    notes: list[tuple[str, str]] = []
    forced = 0
    try:
        obj = jload(ask("start", start_prompt(question), question=question), {})
        first = str(obj.get("문서") or "").strip()
        if first.lower() in ("null", "none", "없음"):
            first = ""
        if first and first not in DOCS:
            fixed = next((n for n in DOCS if n.replace(" ", "") in first.replace(" ", "")), None)
            rec("fix", what="첫 문서", given=first, used=fixed)
            first = fixed or ""
        if not first and in_scope_hint(question):  # 범위 안이 분명한데 고르지 않았으면 질문에 나온 지명, 없으면 링크가 가장 많은 문서
            first = next((n for n in DOCS if n in question), "") or max(DOCS, key=lambda n: len(LINKS.get(n, [])))
            rec("fix", what="첫 문서", given=str(obj.get("범위밖") or "없음"), used=first)
            obj["범위밖"] = ""
        out_of_scope = str(obj.get("범위밖") or "") if not first else ""
        if not first and not out_of_scope:
            out_of_scope = "첫 문서를 고르지 못함"
        nxt = first or None
        while nxt and len(reads) < budget:
            note = ask("read", read_prompt(nxt, DOCS[nxt], question), title=nxt, text=DOCS[nxt])
            rec("read", sec=0, doc=nxt, chars=min(len(DOCS[nxt]), CFG["read_cap_chars"]), assigned=not reads, round=1,
                irrelevant=note.strip().startswith("관련 없음"))
            reads.append(nxt)
            notes.append((nxt, note))
            if len(reads) >= budget:
                break
            # 후보: 읽은 문서들이 많이 가리키는 문서부터. 다 막히면 코퍼스 나머지 전부
            score: dict[str, int] = {}
            for v in reads:
                for x in LINKS.get(v, []):
                    if x not in reads:
                        score[x] = score.get(x, 0) + 1
            cand = sorted(score, key=lambda x: (-score[x], x)) or [x for x in DOCS if x not in reads]
            pick = str(jload(ask("pick", next_prompt(question, notes, cand, budget - len(reads)), candidates=cand), {}).get("문서") or "")
            ok = pick in DOCS and pick not in reads
            rec("pick", sec=0, cand=len(cand), frontier=bool(score), chose=pick if ok else None, forced=not ok)
            if not ok:  # 예산을 남기고 멈추지 않는다
                forced += 1
                pick = cand[0]
            nxt = pick
        if out_of_scope:
            report = f"# {question}\n\n이 질문은 코퍼스(대항해시대 지명 문서 {len(DOCS)}건)로 답할 수 없어 조사하지 않았다. 이유: {out_of_scope}"
        else:
            def draft(p):
                r = ask("report", p, question=question, notes=notes).strip()
                return r.removeprefix("```markdown").removeprefix("```").removesuffix("```").strip()
            prompt = report_prompt(question, notes, sections)
            report = draft(prompt)
            bad, n = uncited(report)
            if notes and n and len(bad) / n > 0.3:  # 멀티 에이전트 집필과 같은 인용 규율 점검 (한 번 되돌려 보냄)
                report2 = draft((prompt[0], prompt[1] + recite_hint(bad)))
                bad2, n2 = uncited(report2)
                keep = n2 and len(bad2) / n2 < len(bad) / n
                rec("recite", sec=0, before=f"{n - len(bad)}/{n}", after=f"{n2 - len(bad2)}/{n2}", kept="새 원고" if keep else "첫 원고", round=1)
                if keep:
                    report = report2
        rec("write", sec=0, chars=len(report), cites=len(metrics.cited(report)), enough=True, gap="", round=1)
    except BudgetExceeded as e:
        rec("abort", reason=str(e))
        meter.save(run_id)
        raise
    m = metrics.evaluate(report, [(AGENT, d) for d in reads], set(DOCS), meter.coord_chars, meter.sub_chars)
    m["LLM호출"], m["원"] = meter.calls, round(meter.won, 2)
    rec("evaluate", metrics=m, alarms=metrics.alarms(m))
    secs = time.time() - t0
    rec("run_end", seconds=round(secs, 1), calls=meter.calls, won=round(meter.won, 2), tok_in=meter.tok_in, tok_cached=meter.tok_cached, tok_out=meter.tok_out)
    meter.save(run_id)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / f"{run_id}.md").write_text(report, encoding="utf-8")
    summary = {"run_id": run_id, "qid": qid, "question": question, "mode": mode, "agent": AGENT,
               "switches": {k: False for k in CFG["switches"]}, "sections": sections, "budget": budget,
               "reads": len(reads), "forced_picks": forced, "reference": ref["run_id"] if ref else None,
               "seconds": round(secs, 1), "version": VERSION, "metrics": m}
    append_line(OUT / "baseline.jsonl", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qid", help="data/questions.json 의 질문 번호 (예: Q4)")
    ap.add_argument("--question", help="직접 질문")
    ap.add_argument("--all", action="store_true", help="질문 전부")
    ap.add_argument("--fake", action="store_true", help="가짜 모델 (0원)")
    ap.add_argument("--env", help="OPENAI_API_KEY 가 든 .env 경로 (실제 호출일 때)")
    ap.add_argument("--budget", type=int, help="읽기 횟수. 안 주면 같은 질문의 멀티 에이전트 실행에서 가져온다")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    jobs = [(q["question"], q["id"]) for q in QUESTIONS] if a.all else [(a.question, None)] if a.question else \
        [(next(q["question"] for q in QUESTIONS if q["id"] == a.qid), a.qid)]
    if not a.fake:
        load_key(a.env)
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY 가 없습니다. --env 로 .env 경로를 주거나 --fake 로 돌리세요.")
    for q, qid in jobs:
        s = run_single(q, qid, a.fake, a.budget)
        m = s["metrics"]
        src = f"멀티 실행 {s['reference']} 과 같은 예산" if s["reference"] else "기준 실행 없음 → config 기본 예산"
        print(f"[{s['run_id']}] {q}\n  읽기 {s['reads']}/{s['budget']}회 ({src}) · 코드가 대신 고름 {s['forced_picks']}회\n"
              f"  {metrics.line(m)} · LLM {m['LLM호출']}회 · ₩{m['원']} · {s['seconds']}초")
        for al in metrics.alarms(m):
            print(f"  ⚠ {al}")


if __name__ == "__main__":
    main()
