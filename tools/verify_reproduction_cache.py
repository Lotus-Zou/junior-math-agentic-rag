#!/usr/bin/env python3
"""Measure uncached, same-process L1 and cross-process Redis L2 responses."""

import os
from pathlib import Path
import random
import sys
import time
import uuid

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.reproduce_experiments import headers, post_question, write


def main():
    if os.getenv("FORCE_LLM_EVERY_TURN", "true").lower() != "false":
        raise SystemExit("Enable the answer cache before this experiment")
    index = 120000+random.randrange(10000)
    query = f"解方程 17(x-{index})=51。"
    first = post_question(query,index=index,key=uuid.uuid4().hex)
    second = post_question(query,index=index,key=uuid.uuid4().hex)
    user, session = f"experiment-user-{index%100}", f"experiment-session-{index}"
    body = {"query":query,"user_id":user,"session_id":session,"language":"zh","conversation_history":[],"request_id":uuid.uuid4().hex}
    start = time.perf_counter()
    response = httpx.post("http://127.0.0.1:28001/ask",json=body,headers=headers(user,session),timeout=60,trust_env=False)
    third = {"status":response.status_code,"latency_ms":(time.perf_counter()-start)*1000,"response":response.json()}
    checks = {"first_uncached":first["business_success"] and not first["response"].get("cached"),
              "same_process_hit":second["business_success"] and second["response"].get("cached") is True,
              "cross_process_hit":third["status"]==200 and third["response"].get("cached") is True,
              "same_answer":first.get("response",{}).get("answer")==second.get("response",{}).get("answer")==third["response"].get("answer")}
    write("cache-path",{"summary":{"passed":all(checks.values()),"checks":checks,
            "uncached_ms":first["latency_ms"],"local_hit_ms":second["latency_ms"],"cross_process_hit_ms":third["latency_ms"]},
            "scope":"A unique request through gRPC then a different HTTP process; transport request IDs and idempotency keys differ. Cache hits carry original generation metrics, not new model calls.","requests":[first,second,third]})


if __name__ == "__main__":
    main()
