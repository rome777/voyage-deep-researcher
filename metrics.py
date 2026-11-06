"""정답표도 판정 모델도 쓰지 않는 지표.

만들어진 보고서와 '실제로 읽은 문서 목록'만 본다. 그래서 질문·코퍼스가 바뀌어도 같은 잣대가 돈다.
지표마다 (1) 어느 장치를 보는 신호인지 (2) 높낮이를 견주는 신호인지, 0이어야 하는 경보인지를 적어 둔다.
설명이 한 줄로 안 되는 지표는 넣지 않았다.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

# 인용 하나는 한 줄 안, 40자 이내. 닫는 » 를 빠뜨린 '«고아, 무스카트…' 가 다음 » 까지 수백 자를 삼키지 않게 한다.
CITE = re.compile(r"«([^»«\n]{1,40})»")
CITE_GROUP = re.compile(r"^((?:«[^»«\n]{1,40}»[\s,·]*)+)")

# 이름: (한 줄 설명, 보는 장치, 종류)
INFO = {
    "근거율": ("본문(절) 문장 중 인용(«지명»)이 붙은 문장의 비율. 인용 금지인 머리말·맺음말은 뺌", "조사 깊이 · 재위임", "비교"),
    "허위인용": ("아무 함대도 읽지 않은 문서를 인용한 수", "집필의 근거 규율", "경보(0이어야 함)"),
    "엉뚱한인용": ("코퍼스에 없는 이름을 «»로 인용한 수", "집필 형식", "경보(0이어야 함)"),
    "읽고안쓴": ("읽었지만 보고서에 한 번도 인용되지 않은 문서 수", "배정 품질 (엉뚱한 문서를 읽혔나)", "비교"),
    "편중": ("가장 많이 인용된 문서가 전체 인용에서 차지하는 비율", "배정 · 구역", "비교"),
    "중복률": ("여러 함대가 같은 문서를 다시 읽은 비율", "배정 · 구역", "비교"),
    "격리율": ("코디네이터가 본 글자가 전체 입력 글자에서 차지하는 비율", "컨텍스트 격리", "비교"),
}


def split_sentences(text: str) -> list[str]:
    """글 한 덩어리를 문장으로 나눈다. 마침표 뒤에 붙은 인용(예: '…했다. «리스본» 그 뒤…')은 앞 문장 끝에 붙인다.
    수업 노트북 규칙은 이걸 다음 문장으로 넘겼고, 첫 판 규칙('«' 앞에서는 안 나눔)은 인용을 문장 앞에 쓰는 원고에서
    문단 전체를 한 문장으로 세었다. 둘 다 근거율을 틀리게 만든다."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for p in re.split(r"(?<=[.!?。])\s+", line):
            m = CITE_GROUP.match(p)
            if m and out:  # 문장 머리에 온 인용 묶음 → 앞 문장 것
                out[-1] += " " + m.group(1).strip(" ,·")
                p = p[m.end():]
            if p.strip():
                out.append(p.strip())
    return [s for s in out if len(s) > 15]


def body_text(report: str) -> str:
    """근거율을 잴 본문: '## 절' 아래 글만. 첫 '##' 앞의 머리말과, 마지막 절 뒤에 따로 떨어진 맺음말 문단은 뺀다
    (편집자에게 인용을 금지했으므로 분모에 넣으면 머리말을 길게 쓴 쪽이 손해를 본다).
    '##' 가 없는 보고서(범위 밖 등)는 전체를 본문으로 본다."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", report.strip()) if b.strip()]
    heads = [i for i, b in enumerate(blocks) if b.startswith("## ")]
    if not heads:
        return "\n".join(b for b in blocks if not b.startswith("# "))
    body = blocks[heads[0]:]
    last = len(blocks) - 1
    if last > heads[-1] and not blocks[last].startswith("#"):
        body = blocks[heads[0]:last]  # 마지막 절 뒤의 맺음말
    return "\n".join(body)


def sentences(report: str) -> list[str]:
    """본문(절) 문장. 제목 줄과 15자 이하 조각은 세지 않는다."""
    return split_sentences(body_text(report))


def cited(text: str) -> list[str]:
    """«A» 와 «A, B» 를 모두 이름 목록으로 (한 괄호에 여러 이름을 쉼표로 넣는 원고가 있다)."""
    return [n.strip() for c in CITE.findall(text) for n in re.split(r"[,·]", c) if n.strip()]


def evaluate(report: str, reads: Iterable[tuple[str, str]], corpus_names: set[str],
             coord_chars: int = 0, sub_chars: int = 0) -> dict:
    """reads: (함대·절, 문서) 쌍 목록 — 읽기 시도를 모두 담는다(중복 포함)."""
    reads = list(reads)
    read_docs = {d for _, d in reads}
    sents = sentences(report)
    grounded = [s for s in sents if any(c in corpus_names for c in cited(s))]
    all_cites = cited(report)
    valid = [c for c in all_cites if c in corpus_names]
    use = Counter(valid)
    return {
        "문장수": len(sents),
        "근거율": round(100 * len(grounded) / max(len(sents), 1), 1),
        "허위인용": sorted(set(valid) - read_docs),
        "엉뚱한인용": sorted({c for c in all_cites if c not in corpus_names}),
        "읽은문서": len(read_docs),
        "읽고안쓴": sorted(read_docs - set(use)),
        "편중": round(100 * max(use.values()) / sum(use.values()), 1) if use else 0.0,
        "중복률": round(100 * (1 - len(read_docs) / len(reads)), 1) if reads else 0.0,
        "인용수": len(valid),
        "보고서자수": len(report),
        "격리율": round(100 * coord_chars / max(coord_chars + sub_chars, 1), 1),
    }


def alarms(m: dict) -> list[str]:
    """0이어야 하는 경보만 모은다. 하나라도 있으면 그 보고서는 비교 대상에서 뺀다."""
    out = []
    if m.get("허위인용"):
        out.append(f"허위 인용 {len(m['허위인용'])}건: {', '.join(m['허위인용'])}")
    if m.get("엉뚱한인용"):
        out.append(f"코퍼스에 없는 인용 {len(m['엉뚱한인용'])}건: {', '.join(m['엉뚱한인용'][:5])}")
    return out


def line(m: dict) -> str:
    return (f"근거율 {m['근거율']}% · 허위 {len(m['허위인용'])} · 엉뚱 {len(m['엉뚱한인용'])} · "
            f"읽고안쓴 {len(m['읽고안쓴'])} · 편중 {m['편중']}% · 중복 {m['중복률']}% · "
            f"격리율 {m['격리율']}% · {m['보고서자수']:,}자")
