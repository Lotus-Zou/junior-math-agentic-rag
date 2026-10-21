#!/usr/bin/env python3
"""Manage an isolated, unprivileged reproduction stack with real dependencies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
STATE = RUNTIME / "stack.json"


def configuration() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"secret": secrets.token_hex(32), "database_password": secrets.token_hex(24), "processes": {}}


def save(state: dict) -> None:
    RUNTIME.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2) + "\n")
    STATE.chmod(0o600)


def environment(state: dict) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "PYTHON_DOTENV_DISABLED": "1", "PYTHONUNBUFFERED": "1",
        "FORCE_LLM_EVERY_TURN": "true" if state.get("force_llm_every_turn", True) else "false",
        "AUTH_SHARED_SECRET": state["secret"], "REQUIRE_USER_IDENTITY": "true",
        "AUTH_BROWSER_SESSIONS": "true", "AUTH_COOKIE_SECURE": "false",
        "REDIS_URL": "redis://127.0.0.1:26379/0", "GATEWAY_REDIS_URL": "redis://127.0.0.1:26379/1",
        "MYSQL_URL": f"mysql://math_agent:{state['database_password']}@127.0.0.1:23306/math_agent",
        "OPENAI_API_KEY": "isolated-local-model", "OPENAI_API_BASE": "http://127.0.0.1:21434/v1",
        "MODEL_PROVIDER": "custom", "MODEL_WIRE_API": "chat_completions",
        "LLM_MODEL_NAME": "qwen2.5:7b-instruct", "CRITIC_MODEL_NAME": "qwen2.5:7b-instruct", "RERANK_MODEL_NAME": "qwen2.5:7b-instruct",
        "EMBEDDING_PROVIDER": "openai", "EMBEDDING_API_BASE": "http://127.0.0.1:21434/v1", "EMBEDDING_MODEL_NAME": "bge-m3:567m",
        "MATH_EMBEDDING_MODEL_PATH": "bge-m3:567m", "ENABLE_DENSE_RETRIEVAL": "true",
        "RUN_TIMEOUT_SECONDS": "40", "LLM_CALL_TIMEOUT_SECONDS": "15", "LLM_MAX_RETRIES": "0",
        "AI_MAX_INFLIGHT": "4", "AI_QUEUE_MAXSIZE": "512", "MYSQL_POOL_SIZE": "8",
        "GATEWAY_MAX_INFLIGHT": "512",
        "CHROMA_PATH": str(RUNTIME / "chroma"), "TRACE_PATH": str(RUNTIME / "traces"),
        "PUBLIC_CASE_CORPUS_PATH": str(RUNTIME / "public-corpus/train-only.sqlite3") if (RUNTIME / "public-corpus/train-only.sqlite3").is_file() else "",
        "ANSWER_CACHE_VERSION": "target-verified-local-explanation-v5",
        "TASK_CONSUMER_NAME": "reproduction-worker", "OPERATIONS_METRICS_TOKEN": state["secret"],
        "AI_PROTOCOL": "grpc", "AI_GRPC_TARGET": "127.0.0.1:25051", "AI_UPSTREAM": "http://127.0.0.1:28001",
        "GRPC_PORT": "25051", "GATEWAY_PORT": "28080",
        "OLLAMA_HOST": "127.0.0.1:21434", "OLLAMA_MODELS": str(RUNTIME / "models"),
        "OLLAMA_NUM_PARALLEL": "4", "OLLAMA_MAX_LOADED_MODELS": "2", "OLLAMA_KEEP_ALIVE": "30m",
        "OLLAMA_CONTEXT_LENGTH": "4096", "OLLAMA_MAX_QUEUE": "16",
    })
    return env


def listening(port: int) -> bool:
    with socket.socket() as connection:
        connection.settimeout(.2)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def launch(name: str, command: list[str], state: dict, env: dict, port: int | None = None) -> None:
    entry = state["processes"].get(name, {})
    if entry and alive(entry["pid"]):
        print(f"{name}: already running", flush=True)
        return
    if port and listening(port):
        raise RuntimeError(f"Port {port} is already occupied; no existing process was changed")
    log_path = RUNTIME / f"{name}.log"
    with log_path.open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state["processes"][name] = {"pid": process.pid, "command": command, "port": port, "log": str(log_path)}
    save(state)
    if port:
        for _ in range(300):
            if listening(port):
                print(f"{name}: listening on 127.0.0.1:{port}", flush=True)
                return
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited: {log_path.read_text()[-3000:]}")
            time.sleep(.1)
        raise TimeoutError(f"{name} startup exceeded 30 seconds; inspect {log_path}")


def start_dependencies(state: dict, env: dict) -> None:
    manifest = json.loads((RUNTIME / "resources/manifest.json").read_text())
    mysql_root = Path(manifest["mysql"]["directory"])
    env["LD_LIBRARY_PATH"] = str(RUNTIME / "resources/libaio/usr/lib/x86_64-linux-gnu") + ":" + env.get("LD_LIBRARY_PATH", "")
    data = RUNTIME / "mysql-data"
    if not (data / "auto.cnf").exists():
        data.mkdir(exist_ok=True)
        subprocess.run([str(mysql_root / "bin/mysqld"), "--no-defaults", "--initialize-insecure",
                        f"--basedir={mysql_root}", f"--datadir={data}"], env=env, check=True)
    launch("mysql", [str(mysql_root / "bin/mysqld"), "--no-defaults", f"--basedir={mysql_root}",
                    f"--datadir={data}", "--bind-address=127.0.0.1", "--port=23306", "--mysqlx=OFF",
                    f"--socket={RUNTIME / 'mysql.sock'}", f"--pid-file={RUNTIME / 'mysql.pid'}",
                    "--innodb-buffer-pool-size=256M", "--max-connections=128"], state, env, 23306)
    import mysql.connector

    marker = RUNTIME / "mysql-initialized"
    connection = mysql.connector.connect(host="127.0.0.1", port=23306, user="root", password=state["database_password"] if marker.exists() else "")
    try:
        cursor = connection.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS math_agent CHARACTER SET utf8mb4")
        cursor.execute("CREATE USER IF NOT EXISTS 'math_agent'@'127.0.0.1' IDENTIFIED BY %s", (state["database_password"],))
        cursor.execute("GRANT ALL ON math_agent.* TO 'math_agent'@'127.0.0.1'")
        if not marker.exists():
            cursor.execute("ALTER USER 'root'@'localhost' IDENTIFIED BY %s", (state["database_password"],))
        connection.commit()
        marker.touch()
    finally:
        connection.close()
    redis_binary = Path(os.getenv("REPRO_REDIS_BINARY", str(RUNTIME / "resources/redis/usr/bin/redis-server")))
    if not redis_binary.exists():
        raise RuntimeError("Set REPRO_REDIS_BINARY to a downloaded Redis 7 server")
    (RUNTIME / "redis-data").mkdir(exist_ok=True)
    launch("redis", [str(redis_binary), "--bind", "127.0.0.1", "--port", "26379", "--dir", str(RUNTIME / "redis-data"),
                    "--save", "", "--appendonly", "yes"], state, env, 26379)
    launch("model", ["ollama", "serve"], state, env, 21434)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("dependencies", "start", "status", "stop", "restart-apps", "run"))
    parser.add_argument("--answer-cache", choices=("enabled", "disabled"))
    args, command = parser.parse_known_args()
    if args.action != "run" and command:
        parser.error("Unexpected arguments: " + " ".join(command))
    state = configuration()
    if args.answer_cache:
        state["force_llm_every_turn"] = args.answer_cache == "disabled"
    save(state)
    env = environment(state)
    if args.action in {"dependencies", "start"}:
        start_dependencies(state, env)
    if args.action == "restart-apps":
        for name in ("gateway", "worker", "grpc", "app"):
            entry = state["processes"].get(name)
            if entry and alive(entry["pid"]):
                os.killpg(entry["pid"], signal.SIGTERM)
        for _ in range(100):
            if not any(listening(port) for port in (28080, 25051, 28001)):
                break
            time.sleep(.1)
        for name in ("gateway", "worker", "grpc", "app"):
            state["processes"].pop(name, None)
        save(state)
    if args.action in {"start", "restart-apps"}:
        go = os.getenv("REPRO_GO", "/tmp/codex-go/go/bin/go")
        subprocess.run([go, "build", "-o", str(RUNTIME / "gateway"), "."], cwd=ROOT / "gateway", check=True)
        launch("app", [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "28001"], state, env, 28001)
        launch("grpc", [sys.executable, "grpc_server.py"], state, env, 25051)
        launch("worker", [sys.executable, "worker.py"], state, env)
        launch("gateway", [str(RUNTIME / "gateway")], state, env, 28080)
    if args.action == "status":
        print(json.dumps({name: {"alive": alive(entry["pid"]), "port": entry.get("port"), "listening": listening(entry["port"]) if entry.get("port") else None}
                          for name, entry in state["processes"].items()}, indent=2))
    if args.action == "stop":
        for name, entry in reversed(list(state["processes"].items())):
            if alive(entry["pid"]):
                os.killpg(entry["pid"], signal.SIGTERM)
                print(f"{name}: termination requested")
    if args.action == "run":
        command = command[1:] if command[:1] == ["--"] else command
        if not command:
            parser.error("run requires a command")
        raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
