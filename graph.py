"""대항해시대 딥리서처: 기획 → 배치 → 서브에이전트(동시) → 점검(재위임) → 종합 → 평가.

    .venv\\Scripts\\python graph.py --qid Q3 --fake            # 가짜 모델 (0원) 으로 끝까지
    .venv\\Scripts\\python graph.py --qid Q3 --env ..\\simulator\\.env   # 실제 호출 (gpt-4o-mini)
    .venv\\Scripts\\python graph.py --all --fake                # 질문 전부
    .venv\\Scripts\\python graph.py --qid Q4 --fake --off assign,zones   # 스위치 끄기 (소거 실험)

- 코퍼스: data/corpus.json (영어 위키백과 지명 문서의 History 절, 키는 한국어 지명)
- 설정: config.json (절 수 · 절 예산 · 라운드 상한 · 역할 명단 · 스위치 · 비용 상한)
- 결과: output/runs/<run_id>.jsonl (해도 재생용 사건 기록) · output/reports/<run_id>.md · output/runs.jsonl (요약)
- 비용: 실제 호출은 응답의 토큰 사용량으로 원화를 누적하고, config 의 상한(누적 budget_won, 1회 run_cap_won)을 넘기 전에 멈춘다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import operator
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

import metrics

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
CORPUS = json.loads((ROOT / "data" / "corpus.json").read_text(encoding="utf-8"))
PLACES = json.loads((ROOT / "data" / "places.json").read_text(encoding="utf-8"))
QUESTIONS = json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8"))["questions"]
DOCS: dict[str, str] = CORPUS["docs"]
LINKS: dict[str, list[str]] = CORPUS["links"]
META: dict[str, dict] = PLACES["places"]
REGION: dict[str, str] = PLACES["regions"]
OUT = ROOT / "output"
# 실행 기록에 남기는 코드 판. 소거 실험은 같은 판의 기록만 재사용한다 (프롬프트를 고친 뒤의 실행과 섞이지 않게).
VERSION = "2026-11-06f"  # f: 여러 곳을 묻는 질문은 최소 2절 (e: 읽기 프롬프트 — 관련 없음 판정 완화)


class BudgetExceeded(RuntimeError):
    pass


# 질문이 코퍼스 범위 안이라는 확실한 단서: 지명이나 이 말들이 나오면 모델이 '범위 밖'이라 해도 받아들이지 않는다
SCOPE_WORDS = re.compile(r"항구|항로|항해|무역|교역|향신료|은광|은화|노예|원정|식민|거점|선단|포르투갈|스페인|에스파냐|네덜란드|영국|잉글랜드|프랑스|"
                         r"아시아|아메리카|아프리카|인도|대서양|태평양|카리브|대항해")


def in_scope_hint(q: str) -> bool:
    return any(n in q for n in DOCS) or bool(SCOPE_WORDS.search(q))


# 여러 곳·여러 갈래를 묻는 말. 이런 말이 있거나 지명이 하나로 좁혀지지 않으면 목차를 한 절로 끝내지 않는다.
MANY_WORDS = re.compile(r"들|각각|어디어디|어느 어느|여러|모두|비교|차이|거쳐|경로|길로|어떻게 다르")


_NATION_KO = [(r"포르투갈", r"Portug"), (r"스페인|에스파냐", r"Spani|Spain|Castil"), (r"네덜란드|VOC|동인도 회사", r"Dutch|VOC|Netherlands"),
              (r"영국|잉글랜드", r"English|British|Britain"), (r"프랑스", r"French|France")]


def region_hint(q: str, per_region: int = 3, regions: int = 5) -> str:
    """질문에 나온 나라를 자주 언급하는 문서를 권역별로 몇 건씩 짚어 준다 (되묻기용). 나라가 없으면 빈 문자열."""
    pats = [en for ko, en in _NATION_KO if re.search(ko, q)]
    if not pats:
        return ""
    by_region: dict[str, list[tuple[int, str]]] = {}
    for name, text in DOCS.items():
        n = sum(len(re.findall(p, text)) for p in pats)
        if n >= 3:
            by_region.setdefault(META[name]["region"], []).append((n, name))
    ranked = sorted(by_region.items(), key=lambda kv: -sum(n for n, _ in kv[1]))[:regions]
    lines = [f"- {REGION[r]}: " + ", ".join(nm for _, nm in sorted(v, reverse=True)[:per_region]) for r, v in ranked]
    return "\n[권역별 후보 — 질문의 나라를 자주 언급하는 문서]\n" + "\n".join(lines)


def single_ok(q: str) -> bool:
    """한 절로 답해도 되는 질문인가: 코퍼스 지명이 딱 하나 나오고, 여러 곳을 묻는 말이 없을 때만 (예: '믈라카는 어떤 항구였나?')."""
    return sum(n in q for n in DOCS) == 1 and not MANY_WORDS.search(q)


# ───────────────────────── 기록 · 비용 ─────────────────────────
class Recorder:
    """해도가 재생할 사건 기록. 스레드(동시 서브에이전트)에서 함께 쓰므로 잠근다."""

    def __init__(self, run_id: str):
        self.path = OUT / "runs" / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.lock = threading.Lock()

    def __call__(self, kind: str, **data: Any) -> None:
        row = {"t": round(time.time() - self.t0, 3), "type": kind, **data}
        with self.lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@contextmanager
def file_lock(name: str = "output"):
    """프로세스 사이 잠금 (output/.output.lock). 소거 실험을 질문별로 동시에 돌릴 때 spend.json · runs.jsonl 을 지킨다.
    30초 넘게 남은 잠금 파일은 죽은 프로세스가 남긴 것으로 보고 지운다."""
    path = OUT / f".{name}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    for _ in range(1200):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > 30:
                    path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            time.sleep(0.05)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
            path.unlink(missing_ok=True)


def append_line(path: Path, row: dict) -> None:
    with file_lock(), path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Meter:
    """호출 수 · 입력 글자(코디네이터/서브) · 토큰 · 원화. 실제 호출이면 상한을 넘기 전에 멈춘다."""

    def __init__(self, real: bool):
        self.real = real
        self.lock = threading.Lock()
        self.calls = self.coord_chars = self.sub_chars = self.tok_in = self.tok_out = self.tok_cached = 0
        self.won = 0.0
        spend = OUT / "spend.json"
        self.spent_before = json.loads(spend.read_text(encoding="utf-8"))["total_won"] if spend.exists() else 0.0

    def price(self, tin: int, tout: int, cached: int = 0) -> float:
        """cached: 입력 중 프롬프트 캐시에 맞은 토큰 (캐시 단가로 계산)."""
        p = CFG["price_usd_per_mtok"]
        return ((tin - cached) * p["input"] + cached * p.get("cached_input", p["input"]) + tout * p["output"]) / 1e6 * CFG["won_per_usd"]

    def guard(self, est_in_tokens: int) -> None:
        if not self.real:
            return
        nxt = self.price(est_in_tokens, 800)
        if self.won + nxt > CFG["run_cap_won"]:
            raise BudgetExceeded(f"이번 실행 상한 ₩{CFG['run_cap_won']} 초과 직전 (지금 ₩{self.won:.1f})")
        if self.spent_before + self.won + nxt > CFG["budget_won"]:
            raise BudgetExceeded(f"누적 예산 ₩{CFG['budget_won']} 초과 직전 (누적 ₩{self.spent_before + self.won:.1f})")

    def add(self, coord: bool, chars: int, tin: int, tout: int, cached: int = 0) -> None:
        with self.lock:
            self.calls += 1
            if coord:
                self.coord_chars += chars
            else:
                self.sub_chars += chars
            self.tok_in += tin
            self.tok_out += tout
            self.tok_cached += cached
            if self.real:
                self.won += self.price(tin, tout, cached)

    def save(self, run_id: str) -> None:
        if not self.real:
            return
        spend = OUT / "spend.json"
        with file_lock():  # 여러 실험을 동시에 돌려도 누적 비용이 덮어써지지 않게
            data = json.loads(spend.read_text(encoding="utf-8")) if spend.exists() else {"total_won": 0.0, "runs": []}
            data["total_won"] = round(data["total_won"] + self.won, 2)
            data["runs"].append({"run_id": run_id, "won": round(self.won, 2), "calls": self.calls, "tok_in": self.tok_in,
                                 "tok_cached": self.tok_cached, "tok_out": self.tok_out})
            spend.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# ───────────────────────── 모델 ─────────────────────────
class OpenAIModel:
    def __init__(self, meter: Meter):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(model=CFG["model"], temperature=CFG["temperature"], timeout=90, max_retries=0)
        self.meter = meter
        import tiktoken
        self.enc = tiktoken.get_encoding("o200k_base")

    def __call__(self, kind: str, system: str, user: str, coord: bool, info: dict) -> str:
        est = len(self.enc.encode(system + user))
        self.meter.guard(est)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(4):
            try:
                res = self.llm.invoke(msgs)
                break
            except Exception as e:  # 동시에 부르면 속도 제한에 걸릴 수 있다
                transient = any(k in type(e).__name__ for k in ("RateLimit", "APIConnection", "Timeout", "InternalServer"))
                if attempt == 3 or not transient:
                    raise
                time.sleep(2 ** attempt)
        u = getattr(res, "usage_metadata", None) or {}
        cached = (u.get("input_token_details") or {}).get("cache_read", 0) or 0
        self.meter.add(coord, len(system) + len(user), u.get("input_tokens", est), u.get("output_tokens", 0), cached)
        return res.content


class FakeModel:
    """LLM 없이 같은 모양의 답을 돌려준다 (0원). 그래프 · 기록 · 지표 · 해도 연결을 점검하는 용도."""

    def __init__(self, meter: Meter):
        self.meter = meter

    def __call__(self, kind: str, system: str, user: str, coord: bool, info: dict) -> str:
        self.meter.add(coord, len(system) + len(user), (len(system) + len(user)) // 3, 200)
        if kind == "plan":
            q, names = info["question"], info["names"]
            mentioned = [n for n in names if n in q]
            if not in_scope_hint(q):
                return json.dumps({"제목": q, "범위밖": "질문에 코퍼스의 지명·권역과 이어지는 말이 없음", "목차": []}, ensure_ascii=False)
            regions: list[str] = []
            for n in mentioned:
                r = META[n]["region"]
                if r not in regions:
                    regions.append(r)
            for r in sorted(REGION, key=lambda r: -sum(META[n]["region"] == r for n in names)):
                if len(regions) >= (1 if len(mentioned) == 1 else info["max"]):
                    break
                if r not in regions:
                    regions.append(r)
            roles = list(CFG["roster"])
            toc = []
            for i, r in enumerate(regions):
                pick = next((n for n in mentioned if META[n]["region"] == r), None) or max(
                    (n for n in names if META[n]["region"] == r), key=lambda n: len(LINKS.get(n, [])))
                toc.append({"절": f"{REGION[r]}의 거점", "지시": f"{REGION[r]}에서 질문과 관련된 항구를 정리한다", "역할": roles[i % len(roles)], "시작문서": pick})
            return json.dumps({"제목": q, "목차": toc}, ensure_ascii=False)
        if kind == "pick":
            return json.dumps({"문서": info["candidates"][0] if info["candidates"] else None}, ensure_ascii=False)
        if kind == "read":
            first = re.split(r"(?<=[.!?])\s+", info["text"].strip())[:2]
            return " ".join(first)[:300]
        if kind == "write":
            body = " ".join(f"«{d}» 문서에 따르면 {note[:80]}…" for d, note in info["notes"]) or "읽은 자료가 없다."
            return json.dumps({"본문": body, "충분": len(info["notes"]) >= 2, "부족": "" if len(info["notes"]) >= 2 else "읽은 문서가 하나뿐"}, ensure_ascii=False)
        if kind == "edit":
            return json.dumps({"머리말": "이 보고서는 질문을 권역별로 나눠 조사한 결과다.", "맺음말": "각 절의 근거는 «»로 표시한 문서다."}, ensure_ascii=False)
        return ""


def jload(raw: str, default):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return default
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, dict) else default
    except json.JSONDecodeError:
        return default


# ───────────────────────── 프롬프트 ─────────────────────────
def role_line(role: str) -> str:
    return f"너는 {role}이다. {CFG['roster'].get(role, CFG['roster'][CFG['default_role']])}."


_YEAR = re.compile(r"\b(1[4-7]\d\d)(?!\s*(?:–|-)\s*\d+\s*BC)(?!\s*BC)\b")
_AGE = re.compile(r"Portug|Spani|Spain|Dutch|VOC|English|British|Britain|East India|French|France|trade|trading|spice|silver|slave|"
                  r"fort|colon|merchant|ships?\b|fleet|voyage|conquer|captur|settle|port\b|galleon|pepper|cloves|nutmeg|sugar", re.I)
_NATION_TAGS = [("포", r"Portug"), ("스", r"Spani|Spain|Castil"), ("네", r"Dutch|VOC|Netherlands"),
                ("영", r"English|British|Britain|East India Company"), ("프", r"French|France")]


def card_line(text: str) -> str:
    """문서 하나의 카드 (LLM 없이 코드로 뽑음): [자주 나오는 나라] + 대항해시대다운 문장 하나.
    History 절 맨 앞을 자르면 '선사시대 암각화…' 같은 문장이 나와 코디네이터가 판단을 못 했다.
    1480~1799년 연도가 있고 무역·식민 낱말이 많은 문장을 고른다."""
    flat = re.sub(r"\[\d+\]", "", re.sub(r"\s+", " ", text))
    sents = [s for s in re.split(r"(?<=[.!?])\s+", flat) if 40 < len(s) < 400]
    score = lambda s: (2 if any(1480 <= int(y) <= 1799 for y in _YEAR.findall(s)) else 0) + min(3, len(_AGE.findall(s)))
    best = max(sents, key=score) if sents else flat
    n = CFG["card_chars"]
    tags = "".join(k for k, pat in _NATION_TAGS if len(re.findall(pat, text)) >= 2)
    return (f"[{tags}] " if tags else "") + best[:n].rstrip() + ("…" if len(best) > n else "")


_CARDS: str | None = None


def cards() -> str:
    """코디네이터가 보는 '문서 카드' 134장. 본문 전체는 보지 않는다. 한 번 만들어 두고 글자 하나 안 바꾸고 재사용한다
    (프롬프트 맨 앞에 똑같이 놓여야 OpenAI 프롬프트 캐싱으로 반값이 된다)."""
    global _CARDS
    if _CARDS is None:
        _CARDS = "\n".join(f"- {name} ({REGION[META[name]['region']]}): {card_line(txt)}" for name, txt in DOCS.items())
    return _CARDS


CARD_HEAD = ("[문서 카드 — 지명 (권역): [그 문서에 자주 나오는 나라: 포르투갈·스페인·네덜란드·영국·프랑스 머리글자] "
             "대항해시대 문장 하나. 본문은 영어 위키백과 History 절이고 카드보다 훨씬 길다]\n")


def plan_prompt(q: str, k: int) -> tuple[str, str]:
    roster = "\n".join(f"- {r}: {d}" for r, d in CFG["roster"].items())
    sys_ = ("너는 대항해시대 리서치 팀의 코디네이터다. 질문에 답하는 보고서의 목차를 짜고 절마다 리서처 에이전트 하나를 맡긴다.\n"
            f"절은 1개 이상 {k}개 이하. 답이 문서 한 건에 거의 다 있는 질문(항구 하나를 묻는 질문)이면 절을 하나만 둬라. "
            "'항구들', '어디어디', '각각'처럼 여러 곳을 묻거나, 한 대상을 여러 구간에 걸쳐 따라가거나, 둘 이상을 견주는 질문이면 반드시 2절 이상으로 나눠라. "
            "여러 곳을 묻는 질문은 권역(바다)별로 나누고, 절마다 서로 다른 권역의 시작 문서를 배정하라. 절 제목은 서로 달라야 한다.\n"
            "문서 카드는 각 문서에서 뽑은 문장 하나뿐이다. 본문에는 대항해시대의 항해·무역·식민 이야기가 훨씬 많이 들어 있으니, "
            "카드에 질문의 말이 그대로 안 보인다는 이유로 범위 밖이라 판단하지 마라. 질문이 대항해시대(15~18세기)의 항해·무역·식민이나 "
            "이 지명들과 아예 무관할 때만(예: 다른 지역·다른 시대·현대 문제) 목차를 빈 배열로 두고 '범위밖'에 이유를 한 문장으로 적어라.\n"
            "목차는 반드시 내용 단위(권역·항구 무리·구간·쟁점)로 나눠라. '개요·연표·인물·분석' 같은 형식 단위로 나누면 "
            "리서처들이 같은 문서를 겹쳐 읽는다.\n"
            "절마다 가장 먼저 읽을 문서를 문서 카드 목록에서 하나 배정하라. 절끼리 다른 문서를 주어라. 문서 이름은 목록의 한국어 지명을 그대로 써라.\n"
            f"절 성격에 맞는 리서처를 명단에서 골라라.\n[리서처 명단]\n{roster}\n"
            'JSON으로만: {"제목":"보고서 제목","범위밖":"","목차":[{"절":"절 제목","지시":"이 절에서 밝힐 것 한두 문장","역할":"명단의 이름","시작문서":"지명"}]}')
    # 카드(매번 똑같음)를 앞에, 질문(매번 다름)을 맨 뒤에 둔다 → 앞부분이 캐시에 맞아 반값
    return sys_, f"{CARD_HEAD}{cards()}\n\n[질문] {q}"


def pick_prompt(task: dict, read: list[str], cand: list[str]) -> tuple[str, str]:
    return (f"{role_line(task['역할'])} 맡은 절을 쓰려고 다음에 읽을 문서를 후보에서 정확히 하나 고른다. 더 읽을 것이 없으면 그만둔다.\n"
            'JSON으로만: {"문서":"후보에 있는 지명"} 또는 {"문서":null}',
            f"[맡은 절] {task['절']}\n[지시] {task['지시']}\n[이미 읽음] {', '.join(read) or '없음'}\n[후보] {', '.join(cand[:CFG['max_candidates']])}")


def read_prompt(name: str, text: str, brief: str) -> tuple[str, str]:
    # 고정 지시 → 문서 원문 → 이번 지시 순서. 같은 문서를 다른 함대·다른 실행이 또 읽으면 원문까지가 캐시에 맞는다.
    # '관련 없음' 규칙을 느슨하게: 첫 판은 향신료 질문에 «말루쿠 제도»(향신료 제도)까지 '관련 없음'으로 버려 빈 원고가 나왔다.
    return ("너는 자료 조사 담당이다. 영어 문서와 그 뒤의 [지시]를 받는다. 문서에서 지시에 쓸 만한 사실을 한국어로 여섯 문장 이내로 간추려라. "
            "지시가 묻는 것과 같은 지역·나라·상품·항로·시대(15~18세기)의 사실이면 지시를 정면으로 다루지 않아도 쓸 만한 사실이다. "
            "연도와 고유명사는 살려라. 문서 전체가 지시와 완전히 다른 지역·시대 이야기일 때만 '관련 없음'이라고만 답하라.",
            f"[문서: {name}]\n{text[:CFG['read_cap_chars']]}\n\n[지시] {brief}\n위 문서에서 이 지시에 쓸 만한 사실을 간추려라.")


def write_prompt(task: dict, notes: list[tuple[str, str]]) -> tuple[str, str]:
    mat = "\n\n".join(f"[{d}]\n{n}" for d, n in notes) or "(읽은 자료 없음)"
    return (f"{role_line(task['역할'])} 아래 자료만 근거로 보고서의 한 절을 한국어로 써라. 여섯 문장 이상 열두 문장 이하, 600자 이상.\n"
            "문장마다 끝에 근거 문서의 지명을 «지명» 으로 붙여라 (자료에 대괄호로 적힌 이름 그대로). 자료에 없는 내용은 쓰지 마라.\n"
            "지시를 자료로 다 채우지 못했으면 충분을 false 로 하고, 무엇이 비었는지(근거가 비었을 때만, 문장 수 같은 형식 문제는 빼고) 적어라.\n"
            'JSON으로만: {"본문":"...","충분":true,"부족":""}',
            f"[맡은 절] {task['절']}\n[지시] {task['지시']}\n[모은 자료]\n{mat}")


def uncited(text: str) -> tuple[list[str], int]:
    """인용 «» 이 없는 문장과 전체 문장 수. 지표(metrics.근거율)와 같은 문장 분리를 쓴다."""
    sents = metrics.sentences(text)
    return [x for x in sents if not metrics.cited(x)], len(sents)


def recite_hint(bad: list[str]) -> str:
    """원고를 한 번 되돌려 보낼 때 덧붙이는 말: 인용이 빠진 문장을 짚어 준다."""
    shown = "\n".join(f"- {x[:120]}" for x in bad[:8])
    return ("\n[지난 원고에서 문장 끝 인용 «지명» 이 빠진 문장]\n" + shown +
            "\n지난 원고는 문단 끝에만 인용을 붙이거나 아예 빠뜨렸다. 같은 자료로 다시 쓰되, 모든 문장 끝에 그 문장의 근거 문서 «지명» 을 붙여라.")


def edit_prompt(q: str, title: str, secs: list[dict]) -> tuple[str, str]:
    outline = "\n".join(f"{i}. {s['절']}: {s['본문'][:90]}…" for i, s in enumerate(secs, 1))
    return ("너는 보고서를 마무리하는 편집자다. 아래는 리서처들이 쓴 절의 제목과 첫머리다. 본문은 고치지 말고 머리말과 맺음말만 각각 세 문장 이내로 써라. "
            "새 사실을 지어내지 말고 인용 «» 도 붙이지 마라.\n"
            'JSON으로만: {"머리말":"...","맺음말":"..."}',
            f"[질문] {q}\n[보고서 제목] {title}\n[절 목록]\n{outline}")


# ───────────────────────── 그래프 ─────────────────────────
class Research(TypedDict):
    question: str
    plan: dict
    sections: Annotated[list, operator.add]
    visited: Annotated[list, operator.add]
    report: str
    metrics: dict
    task: dict
    prior: dict


class Pipeline:
    def __init__(self, model, meter: Meter, rec: Recorder, switches: dict, k: int, budget: int):
        self.m, self.meter, self.rec, self.sw, self.k, self.budget = model, meter, rec, switches, k, budget

    def ask(self, kind: str, prompt: tuple[str, str], coord: bool, **info) -> str:
        return self.m(kind, prompt[0], prompt[1], coord, info)

    # ① 기획
    def plan(self, s: dict) -> dict:
        names = list(DOCS)
        prompt = plan_prompt(s["question"], self.k)
        raw = self.ask("plan", prompt, True, question=s["question"], names=names, max=self.k)
        obj = jload(raw, {})
        if not obj.get("목차") and in_scope_hint(s["question"]):  # 지명·나라·무역이 나오는 질문을 '범위 밖'이라 하면 한 번 되묻는다
            self.rec("fix", what="범위밖", given=str(obj.get("범위밖") or "빈 목차"), used="되묻기")
            prompt = (prompt[0], prompt[1] + "\n[다시 확인] 이 질문은 코퍼스의 지명·나라·무역을 다루므로 범위 안이다. 범위밖을 비우고 목차를 짜라.")
            obj = jload(self.ask("plan", prompt, True, question=s["question"], names=names, max=self.k), {})
        spread = lambda o: len({META[str(t.get("시작문서"))]["region"] for t in (o.get("목차") or []) if str(t.get("시작문서")) in DOCS})
        n_sec = len(obj.get("목차") or [])
        if n_sec and self.k > 1 and self.sw["assign"] and not single_ok(s["question"]) and (n_sec == 1 or spread(obj) <= 1):
            # 여러 곳을 묻는데 한 절로 짜거나 모든 절을 한 권역에서 시작하면 한 번 되묻는다. 그대로 두면 서브에이전트가 하나뿐이거나
            # 모두 같은 바다만 읽어 분업이 일어나지 않는다 (판 d 소거 실험에서 멀티 실행 40회 중 20회가 한 절로 끝났다).
            # 프롬프트만으로는 gpt-4o-mini 가 잘 따르지 않아서, 질문에 나온 나라를 자주 언급하는 문서를 권역별로 짚어 준다.
            self.rec("fix", what="절 수", given=f"{n_sec}절 · 권역 {spread(obj)}곳", used="되묻기")
            prompt = (prompt[0], prompt[1] + f"\n[다시 확인] 이 질문은 여러 항구·권역·나라·구간에 걸친 질문이다. 권역(바다)별로 2절 이상 {self.k}절 이하로 나누고, "
                      f"절마다 서로 다른 권역의 시작 문서와 서로 다른 제목을 줘라. '맥락·배경·의의' 같은 형식으로 나누지 마라.{region_hint(s['question'])}")
            again = jload(self.ask("plan", prompt, True, question=s["question"], names=names, max=self.k), {})
            if (len(again.get("목차") or []), spread(again)) > (n_sec, spread(obj)):
                obj = again
        toc, taken = [], set()
        for item in (obj.get("목차") or [])[: self.k]:
            seed = str(item.get("시작문서") or "")
            if not self.sw["assign"]:
                seed = ""
            elif seed not in DOCS or seed in taken:  # 모델이 지어낸 이름이나 겹친 배정은 코드가 바로잡는다
                fixed = next((n for n in names if n not in taken and n.replace(" ", "") in seed.replace(" ", "")), None)
                self.rec("fix", what="시작문서", given=seed, used=fixed)
                seed = fixed or ""
            if seed:
                taken.add(seed)
            role = item.get("역할", "")
            title, n = str(item.get("절") or "무제"), 2
            while any(t["절"] == title for t in toc):  # 같은 제목이 둘이면 원고가 서로 덮어쓴다 (latest 는 제목으로 묶는다)
                title = f"{item.get('절') or '무제'} ({n})"; n += 1
            toc.append({"절": title, "지시": item.get("지시", s["question"]),
                        "역할": role if (self.sw["roles"] and role in CFG["roster"]) else CFG["default_role"],
                        "시작문서": seed, "예산": self.budget})
        if not toc:  # 코퍼스로 답할 수 없는 질문: 아무 문서나 읽혀 그럴듯한 글을 짓지 않고, 범위 밖이라고 보고한다
            why = str(obj.get("범위밖") or "문서 카드에서 질문과 이어지는 지명·사건을 찾지 못함")
            p = {"제목": obj.get("제목") or s["question"], "목차": [], "배치": [], "바퀴": 1, "범위밖": why}
            self.rec("plan", title=p["제목"], toc=[], switches=self.sw, out_of_scope=why)
            return {"plan": p}
        p = {"제목": obj.get("제목") or s["question"], "목차": toc, "배치": list(range(len(toc))), "바퀴": 1}
        self.rec("plan", title=p["제목"], toc=toc, switches=self.sw)
        return {"plan": p}

    # ② 배치 (갈림길)
    def dispatch(self, s: dict) -> dict:
        self.rec("dispatch", round=s["plan"]["바퀴"], sections=s["plan"]["배치"])
        return {}

    def fanout(self, s: dict):
        idxs = s["plan"].get("배치", [])
        if not idxs:
            return "review"
        latest = self.latest(s)
        toc = s["plan"]["목차"]
        sends = []
        for i in idxs:
            mine = toc[i]["절"]
            avoid: set[str] = set()
            if self.sw["zones"]:  # 남의 구역: 다른 절의 시작 문서 + 다른 절이 이미 읽은 문서
                avoid = {t["시작문서"] for j, t in enumerate(toc) if j != i and t["시작문서"]}
                avoid |= {d for name, sec in latest.items() if name != mine for d in sec["읽은문서"]}
            sends.append(Send("researcher", {"question": s["question"], "task": {**toc[i], "번호": i, "피하기": sorted(avoid), "바퀴": s["plan"]["바퀴"]},
                                             "prior": latest.get(mine, {})}))
        return sends

    # ③ 조사 (서브에이전트)
    def researcher(self, s: dict) -> dict:
        t, prior = s["task"], s.get("prior") or {}
        read, notes = list(prior.get("읽은문서", [])), [tuple(x) for x in prior.get("메모", [])]
        avoid = set(t.get("피하기") or [])
        nxt = t["시작문서"] if t["시작문서"] in DOCS and t["시작문서"] not in read else None
        new_reads = []
        for _ in range(t["예산"]):
            if nxt is None:
                frontier = sorted({x for v in read for x in LINKS.get(v, [])} - set(read) - avoid)
                cand = frontier or [x for x in DOCS if x not in read and x not in avoid]
                raw = self.ask("pick", pick_prompt(t, read, cand), False, candidates=cand)
                pick = (jload(raw, {}).get("문서") or "")
                self.rec("pick", sec=t["번호"], cand=len(cand), frontier=bool(frontier), chose=pick if pick in DOCS else None)
                if pick not in DOCS or pick in read:
                    break
                nxt = pick
            note = self.ask("read", read_prompt(nxt, DOCS[nxt], t["지시"]), False, title=nxt, text=DOCS[nxt])
            assigned = nxt == t["시작문서"]
            self.rec("read", sec=t["번호"], doc=nxt, chars=min(len(DOCS[nxt]), CFG["read_cap_chars"]), assigned=assigned, round=t["바퀴"], irrelevant=note.strip().startswith("관련 없음"), note=note)  # 메모: 사람 비교에서 원고와 대조
            read.append(nxt)
            new_reads.append(nxt)
            notes.append((nxt, note))
            nxt = None
        def draft(prompt):
            raw = self.ask("write", prompt, False, notes=notes)
            obj = jload(raw, {})
            body = str(obj.get("본문") or "")
            if len(body) < 100:  # JSON 이 깨졌으면 원문에서 건진다
                body = re.sub(r'^\s*\{.*?"본문"\s*:\s*"?', "", raw, flags=re.S)[:2500].strip()
            return obj, body
        prompt = write_prompt(t, notes)
        obj, body = draft(prompt)
        bad, n = uncited(body)
        if notes and n and len(bad) / n > 0.3:  # 인용 규율 점검: 문장 30% 넘게 인용이 없으면 한 번 되돌려 보낸다 (코드 판정, 모델 호출 1회)
            obj2, body2 = draft((prompt[0], prompt[1] + recite_hint(bad)))
            bad2, n2 = uncited(body2)
            keep = n2 and len(bad2) / n2 < len(bad) / n
            self.rec("recite", sec=t["번호"], before=f"{n - len(bad)}/{n}", after=f"{n2 - len(bad2)}/{n2}", kept="새 원고" if keep else "첫 원고", round=t["바퀴"])
            if keep:
                obj, body = obj2, body2
        cites = metrics.cited(body)
        sec = {"번호": t["번호"], "절": t["절"], "역할": t["역할"], "본문": body, "읽은문서": read, "메모": notes,
               "인용": cites, "허위인용": sorted({c for c in cites if c in DOCS and c not in read}), "충분": bool(obj.get("충분")),
               "부족": str(obj.get("부족") or ""), "바퀴": t["바퀴"]}
        self.rec("write", sec=t["번호"], chars=len(body), cites=len(cites), enough=sec["충분"], gap=sec["부족"], round=t["바퀴"], body=body)  # 재위임으로 버려진 원고도 남긴다
        return {"sections": [sec], "visited": [[t["절"], d] for d in new_reads]}

    @staticmethod
    def latest(s: dict) -> dict:
        """같은 절이 두 번 올라오면(재위임) '유효 인용이 더 많은 원고'를 남긴다. 같으면 새 원고.
        무조건 덮어쓰면, 인용을 빠뜨린 두 번째 원고가 멀쩡한 첫 원고를 지운다 (수업의 소거 실험에서 확인된 문제)."""
        best: dict[str, dict] = {}
        for sec in s["sections"]:
            old = best.get(sec["절"])
            score = lambda x: len([c for c in x["인용"] if c in x["읽은문서"]])
            if old is None or score(sec) >= score(old):
                best[sec["절"]] = sec
        return best

    # ④ 점검
    def review(self, s: dict) -> dict:
        latest, toc, wheel = self.latest(s), s["plan"]["목차"], s["plan"].get("바퀴", 1)
        gaps = [] if (not self.sw["redelegate"] or wheel >= CFG["max_rounds"]) else [i for i, t in enumerate(toc) if not latest.get(t["절"], {}).get("충분", False)]
        if not gaps:
            why = "재위임 꺼짐" if not self.sw["redelegate"] else ("라운드 상한" if wheel >= CFG["max_rounds"] else "빈 칸 없음")
            self.rec("review", gaps=[], reason=why, round=wheel)
            return {"plan": {**s["plan"], "배치": [], "종료": why}}
        newtoc = [dict(t) for t in toc]
        for i in gaps:
            prev = latest.get(toc[i]["절"], {})
            newtoc[i]["지시"] = (f"{toc[i]['지시']} (재위임: 지난번에 «{'», «'.join(prev.get('읽은문서', [])) or '없음'}»를 읽었지만 "
                                 f"{prev.get('부족') or '근거가 모자랐다'}. 그 빈 칸을 채울 아직 안 본 문서를 찾아라)")
        self.rec("review", gaps=gaps, reason="자기신고 부족", round=wheel, why={i: latest.get(toc[i]["절"], {}).get("부족", "") for i in gaps})
        return {"plan": {**s["plan"], "목차": newtoc, "배치": gaps, "바퀴": wheel + 1}}

    @staticmethod
    def route(s: dict) -> str:
        return "more" if s["plan"].get("배치") else "done"

    # ⑤ 종합
    def synthesize(self, s: dict) -> dict:
        if s["plan"].get("범위밖"):
            report = f"# {s['plan']['제목']}\n\n이 질문은 코퍼스(대항해시대 지명 문서 {len(DOCS)}건)로 답할 수 없어 조사하지 않았다. 이유: {s['plan']['범위밖']}"
            self.rec("synth", chars=len(report), sections=0, out_of_scope=True)
            return {"report": report}
        latest = self.latest(s)
        secs = [latest[t["절"]] for t in s["plan"]["목차"] if t["절"] in latest]
        title = s["plan"]["제목"]
        obj = jload(self.ask("edit", edit_prompt(s["question"], title, secs), True), {})
        parts = [f"# {title}", obj.get("머리말", "")]
        parts += [f"## {i}. {sec['절']}\n{sec['본문']}" for i, sec in enumerate(secs, 1)]
        parts.append(obj.get("맺음말", ""))
        report = "\n\n".join(p for p in parts if p)
        self.rec("synth", chars=len(report), sections=len(secs))
        return {"report": report}

    # ⑥ 평가
    def evaluate(self, s: dict) -> dict:
        m = metrics.evaluate(s["report"], [tuple(v) for v in s["visited"]], set(DOCS), self.meter.coord_chars, self.meter.sub_chars)
        m["LLM호출"], m["원"] = self.meter.calls, round(self.meter.won, 2)
        self.rec("evaluate", metrics=m, alarms=metrics.alarms(m))
        return {"metrics": m}

    def build(self):
        g = StateGraph(Research)
        for name in ("plan", "dispatch", "researcher", "review", "synthesize", "evaluate"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "plan")
        g.add_edge("plan", "dispatch")
        g.add_conditional_edges("dispatch", self.fanout, ["researcher", "review"])
        g.add_edge("researcher", "review")
        g.add_conditional_edges("review", self.route, {"more": "dispatch", "done": "synthesize"})
        g.add_edge("synthesize", "evaluate")
        g.add_edge("evaluate", END)
        return g.compile()


# ───────────────────────── 실행 ─────────────────────────
def load_key(env: str | None) -> None:
    """--env 로 받은 파일에서 OPENAI_API_KEY 한 줄만 읽어 환경 변수로. 키는 출력하지 않는다."""
    if os.getenv("OPENAI_API_KEY") or not env:
        return
    for line in Path(env).read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")


def new_run_id(tag: str) -> str:
    """초 단위 시각 + 꼬리표. 소거 실험처럼 1초 안에 여러 번 돌면 -2, -3 을 붙여 기록 파일이 섞이지 않게 한다."""
    base = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + tag
    rid, n = base, 1
    while (OUT / "runs" / f"{rid}.jsonl").exists():
        n += 1
        rid = f"{base}-{n}"
    return rid


def run(question: str, qid: str | None, fake: bool, switches: dict, k: int, budget: int, run_id: str | None = None) -> dict:
    off = [k for k, v in switches.items() if not v]
    # app.py 처럼 진행 상황을 기록 파일에서 읽어야 하는 쪽은 run_id 를 미리 정해 넘긴다
    run_id = run_id or new_run_id(("-fake" if fake else "-live") + (f"-{qid}" if qid else "") + (f"-off-{'+'.join(off)}" if off else ""))
    meter = Meter(real=not fake)
    rec = Recorder(run_id)
    model = FakeModel(meter) if fake else OpenAIModel(meter)
    rec("run_start", run_id=run_id, qid=qid, question=question, mode="fake" if fake else CFG["model"], switches=switches, sections=k, budget=budget)
    pipe = Pipeline(model, meter, rec, switches, k, budget)
    t0 = time.time()
    try:
        state = pipe.build().invoke({"question": question, "plan": {}, "sections": [], "visited": [], "report": "", "metrics": {}, "task": {}, "prior": {}})
    except BudgetExceeded as e:
        rec("abort", reason=str(e))
        meter.save(run_id)
        raise
    secs = time.time() - t0
    rec("run_end", seconds=round(secs, 1), calls=meter.calls, won=round(meter.won, 2), tok_in=meter.tok_in, tok_cached=meter.tok_cached, tok_out=meter.tok_out)
    meter.save(run_id)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / f"{run_id}.md").write_text(state["report"], encoding="utf-8")
    summary = {"run_id": run_id, "qid": qid, "question": question, "mode": "fake" if fake else CFG["model"], "switches": switches,
               "sections": k, "budget": budget, "seconds": round(secs, 1), "version": VERSION, "metrics": state["metrics"]}
    append_line(OUT / "runs.jsonl", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qid", help="data/questions.json 의 질문 번호 (예: Q3)")
    ap.add_argument("--question", help="직접 질문")
    ap.add_argument("--all", action="store_true", help="질문 전부")
    ap.add_argument("--fake", action="store_true", help="가짜 모델 (0원)")
    ap.add_argument("--env", help="OPENAI_API_KEY 가 든 .env 경로 (실제 호출일 때)")
    ap.add_argument("--off", default="", help="끌 스위치: roles,assign,zones,redelegate")
    ap.add_argument("--sections", type=int, default=CFG["max_sections"])
    ap.add_argument("--budget", type=int, default=CFG["section_budget"])
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    switches = {k: (k not in a.off.split(",")) for k in CFG["switches"]}
    jobs = [(q["question"], q["id"]) for q in QUESTIONS] if a.all else [(a.question, None)] if a.question else [(next(q["question"] for q in QUESTIONS if q["id"] == a.qid), a.qid)]
    if not a.fake:
        load_key(a.env)
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY 가 없습니다. --env 로 .env 경로를 주거나 --fake 로 돌리세요.")
    for q, qid in jobs:
        s = run(q, qid, a.fake, switches, a.sections, a.budget)
        m = s["metrics"]
        print(f"[{s['run_id']}] {q}\n  {metrics.line(m)} · LLM {m['LLM호출']}회 · ₩{m['원']} · {s['seconds']}초")
        for al in metrics.alarms(m):
            print(f"  ⚠ {al}")


if __name__ == "__main__":
    main()
