"""데모 서버: 해도를 띄우고, 새 질문을 실제 LLM(graph.py)으로 돌린다.

    .venv\\Scripts\\python app.py --env path\\to\\.env        # 로컬 → http://127.0.0.1:8000
    OPENAI_API_KEY=... HOST=0.0.0.0 PORT=8000 python app.py  # 서버에 배포할 때 (키는 호스팅의 비밀 환경 변수로)

- 화면은 concepts/ 의 정적 파일(voyage-map.html 등)이다. 이 서버로 열면 질문 칸이 켜지고, GitHub Pages 처럼 정적으로 열면 꺼진다.
- 키는 서버에만 있고 화면으로는 나가지 않는다. 화면은 질문만 보내고 실행 기록만 받는다.
- 비용 보호: 한 번에 하나만 실행 · 실행 1회 상한(config.run_cap_won) · 하루 상한(DAILY_WON) · 접속자(IP)당 하루 횟수(RUNS_PER_IP).
  누적 상한(config.budget_won)은 output/spend.json 기준이다. 서버를 새로 띄우면 이 파일이 비워질 수 있으니 OpenAI 쪽 사용 한도도 함께 걸어 둔다.

API
    GET  /api/health          → {live, busy, today_won, daily_won, runs_left}
    GET  /api/runs            → 실제 LLM 실행 기록 전부 (해도 재생용)
    POST /api/ask {question}  → {run_id}  (202)
    GET  /api/job?run_id=…    → {status, events, error, record}
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
import graph  # noqa: E402
import build_voyage_map as B  # noqa: E402

DAILY_WON = float(os.getenv("DAILY_WON", "300"))
RUNS_PER_IP = int(os.getenv("RUNS_PER_IP", "3"))
QUESTIONS = {q["question"]: q["id"] for q in graph.QUESTIONS}

LOCK = threading.Lock()
JOBS: dict[str, dict] = {}          # run_id → {status, error, question, ip}
BUSY: dict[str, str | None] = {"run_id": None}
PER_IP: dict[tuple[str, str], int] = {}


def today() -> str:
    return dt.datetime.now().strftime("%Y%m%d")


APP_RUNS = graph.OUT / "app_runs.jsonl"   # 이 서버가 받은 실행만 (실험 스크립트로 쓴 돈은 하루 상한에 넣지 않는다)


def today_won() -> float:
    """오늘 이 서버가 돌린 실행에 쓴 돈 (spend.json 에서 app_runs.jsonl 에 적힌 run_id 만 더한다)."""
    ids = {r["run_id"] for r in B._rows(APP_RUNS) if r.get("day") == today()}
    path = graph.OUT / "spend.json"
    if not ids or not path.exists():
        return 0.0
    runs = json.loads(path.read_text(encoding="utf-8")).get("runs", [])
    return round(sum(r.get("won", 0) for r in runs if r.get("run_id") in ids), 2)


def live() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def worker(run_id: str, question: str) -> None:
    try:
        sw = {k: True for k in graph.CFG["switches"]}
        graph.run(question, QUESTIONS.get(question), False, sw, graph.CFG["max_sections"], graph.CFG["section_budget"], run_id=run_id)
        JOBS[run_id]["status"] = "done"
    except graph.BudgetExceeded as e:
        JOBS[run_id].update(status="error", error=f"비용 상한에 걸려 멈췄습니다 — {e}")
    except Exception as e:  # 모델 오류·네트워크 오류는 화면에 그대로 알린다 (키는 메시지에 들어가지 않는다)
        JOBS[run_id].update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        with LOCK:
            BUSY["run_id"] = None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT / "concepts"), **kw)

    def log_message(self, fmt, *args):  # 요청 로그는 짧게
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def client_ip(self) -> str:
        fwd = self.headers.get("X-Forwarded-For", "")
        return fwd.split(",")[0].strip() if fwd else self.client_address[0]

    def send_json(self, obj, status=HTTPStatus.OK) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self.path = "/voyage-map.html"
            return super().do_GET()
        if u.path == "/api/health":
            used = PER_IP.get((self.client_ip(), today()), 0)
            return self.send_json({"live": live(), "busy": bool(BUSY["run_id"]), "today_won": today_won(), "daily_won": DAILY_WON,
                                   "runs_left": max(0, RUNS_PER_IP - used), "run_cap_won": graph.CFG["run_cap_won"]})
        if u.path == "/api/runs":
            return self.send_json(B.load_runs())
        if u.path == "/api/job":
            run_id = (parse_qs(u.query).get("run_id") or [""])[0]
            job = JOBS.get(run_id)
            if not job:
                return self.send_json({"error": "그런 실행이 없습니다"}, HTTPStatus.NOT_FOUND)
            path = graph.OUT / "runs" / f"{run_id}.jsonl"
            events = [{k: v for k, v in e.items() if k not in ("note", "body")} for e in B._rows(path) if e["type"] in B.KEEP | {"pick", "recite", "abort"}]
            rec = None
            if job["status"] == "done":
                summaries = {r["run_id"]: r for r in B._rows(graph.OUT / "runs.jsonl")}
                names = set(graph.DOCS)
                import metrics
                rec = B.run_record(path, summaries, lambda rep, reads: metrics.evaluate(rep, reads, names)["근거율"])
            return self.send_json({"status": job["status"], "error": job.get("error", ""), "events": events, "record": rec})
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/ask":
            return self.send_json({"error": "없는 주소"}, HTTPStatus.NOT_FOUND)
        if not live():
            return self.send_json({"error": "이 서버에는 OpenAI 키가 없어 새 질문을 돌릴 수 없습니다."}, HTTPStatus.SERVICE_UNAVAILABLE)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            q = str(json.loads(self.rfile.read(min(n, 4096)) or b"{}").get("question", "")).strip()
        except (ValueError, json.JSONDecodeError):
            return self.send_json({"error": "질문을 읽지 못했습니다"}, HTTPStatus.BAD_REQUEST)
        if not 2 <= len(q) <= 200:
            return self.send_json({"error": "질문은 2~200자로 적어 주세요"}, HTTPStatus.BAD_REQUEST)
        ip, day = self.client_ip(), today()
        with LOCK:
            if BUSY["run_id"]:
                return self.send_json({"error": "다른 질문을 조사하는 중입니다. 끝나면 다시 보내 주세요.", "run_id": BUSY["run_id"]}, HTTPStatus.CONFLICT)
            if PER_IP.get((ip, day), 0) >= RUNS_PER_IP:
                return self.send_json({"error": f"오늘은 이 접속에서 {RUNS_PER_IP}번까지 돌릴 수 있습니다. 기록 재생은 계속 됩니다."}, HTTPStatus.TOO_MANY_REQUESTS)
            spent = today_won()
            if spent + graph.CFG["run_cap_won"] > DAILY_WON:
                return self.send_json({"error": f"오늘 비용 상한(₩{DAILY_WON:.0f})에 가까워 새 실행을 받지 않습니다 (오늘 ₩{spent})."}, HTTPStatus.TOO_MANY_REQUESTS)
            qid = QUESTIONS.get(q)
            run_id = graph.new_run_id("-live" + (f"-{qid}" if qid else ""))
            BUSY["run_id"] = run_id
            PER_IP[(ip, day)] = PER_IP.get((ip, day), 0) + 1
            JOBS[run_id] = {"status": "running", "question": q, "ip": ip}
            graph.append_line(APP_RUNS, {"run_id": run_id, "day": day})
        threading.Thread(target=worker, args=(run_id, q), daemon=True).start()
        return self.send_json({"run_id": run_id, "question": q}, HTTPStatus.ACCEPTED)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", help="OPENAI_API_KEY 가 든 .env 경로 (로컬). 서버에서는 환경 변수로 준다")
    ap.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    graph.load_key(a.env)
    (graph.OUT / "runs").mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"해도: http://{a.host}:{a.port}/  · 새 질문 실행 {'켜짐' if live() else '꺼짐 (키 없음 — 기록 재생만)'} · 하루 상한 ₩{DAILY_WON:.0f} · 접속당 {RUNS_PER_IP}회")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
