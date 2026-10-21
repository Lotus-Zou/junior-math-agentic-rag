#!/usr/bin/env python3
"""Isolated storage measurements and explicitly labelled capacity simulation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import signal
import statistics
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.reproduce_experiments import BASE, PREFIX, REPORTS, headers, post_question, write


def storage():
    from tools.migrate_mysql import connection_from_url
    from agentic_rag import memory
    from agentic_rag.cache import AnswerCache

    connection = connection_from_url(os.environ["MYSQL_URL"])
    table = "repro_index_"+uuid.uuid4().hex[:12]
    try:
        cursor = connection.cursor()
        cursor.execute(f"CREATE TABLE {table} (id BIGINT PRIMARY KEY AUTO_INCREMENT, user_id BIGINT, session_key VARCHAR(128), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, content VARCHAR(200), INDEX idx_owner(user_id,session_key,created_at))")
        cursor.executemany(f"INSERT INTO {table}(user_id,session_key,content) VALUES (%s,%s,%s)",
                           [(i%200, f"session-{i%17}", "synthetic-index-benchmark") for i in range(20000)])
        connection.commit()
        index_results = {}
        for label, hint in (("without_index", "IGNORE INDEX(idx_owner)"), ("with_index", "FORCE INDEX(idx_owner)")):
            query = f"SELECT * FROM {table} {hint} WHERE user_id=42 AND session_key='session-8' ORDER BY created_at DESC LIMIT 50"
            cursor.execute("EXPLAIN FORMAT=JSON "+query)
            plan = json.loads(cursor.fetchone()[0])
            samples = []
            for _ in range(40):
                start = time.perf_counter()
                cursor.execute(query)
                rows = cursor.fetchall()
                samples.append((time.perf_counter()-start)*1000)
            index_results[label] = {"mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples), "returned_rows": len(rows), "explain": plan, "samples_ms": samples}
    finally:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
        connection.close()
    held = []
    pool_result = {}
    try:
        for _ in range(memory.MYSQL_POOL_SIZE):
            held.append(memory._mysql_connection())
        start = time.perf_counter()
        try:
            extra = memory._mysql_connection()
        except Exception as exc:
            pool_result = {"bounded": True, "overflow_error": type(exc).__name__, "overflow_ms": (time.perf_counter()-start)*1000}
        else:
            extra.close()
            pool_result = {"bounded": False}
    finally:
        for item in held:
            item.close()
    connection = memory._mysql_connection()
    connection.close()
    pool_result.update(size=memory.MYSQL_POOL_SIZE, recovered=True)
    cache = AnswerCache()
    key = {"query": "isolated-cache-benchmark-"+uuid.uuid4().hex}
    cache.set(key, {"answer": "validated"})
    cache_results = {}
    for name in ("local_l1", "redis_l2", "miss"):
        values = []
        for i in range(100):
            if name == "redis_l2":
                cache.local.clear()
            start = time.perf_counter()
            value = cache.get(key if name != "miss" else {"query": key["query"]+str(i)})
            values.append((time.perf_counter()-start)*1000)
        cache_results[name] = {"mean_ms": statistics.mean(values), "hit": value is not None}
    write("storage", {"summary": {"index_speedup": index_results["without_index"]["mean_ms"]/index_results["with_index"]["mean_ms"], "pool": pool_result, "cache": cache_results},
                      "index_benchmark": index_results, "scope": "20,000 synthetic rows in a temporary MySQL table; pool and L1/L2 tests use actual isolated services. Not production traffic."})


def simulate():
    import simpy

    measured = json.loads((REPORTS / f"{PREFIX}-quality.json").read_text())
    samples = [row["latency_ms"]/1000 for row in measured["results"] if row.get("verified") and row.get("response", {}).get("metrics", {}).get("model_successes", 0) > 0]
    if not samples:
        raise RuntimeError("Real model latency samples are required before simulation")

    def run(seed, replicas, cache_hit, fault):
        rng = random.Random(seed)
        env = simpy.Environment()
        resource = simpy.Resource(env, capacity=4*replicas)
        results = []
        clients = 0

        def request(arrival):
            if rng.random() < cache_hit:
                yield env.timeout(.005)
                results.append((arrival, 200, .005, True))
                return
            if resource.count+len(resource.queue) >= 12*replicas:
                results.append((arrival, 429, .002, False))
                return
            with resource.request() as ticket:
                yield ticket
                service = rng.choice(samples)
                # A 30-second model disturbance adds a timeout before verified
                # deterministic fallback; this is a scenario assumption.
                if fault and 500 <= env.now < 530:
                    service += 15
                yield env.timeout(service)
                elapsed = env.now-arrival
                results.append((arrival, 200 if elapsed <= 40 else 504, elapsed, False))

        def arrivals():
            nonlocal clients
            while env.now < 1080:
                rate = 1.5 if env.now < 900 else 30
                yield env.timeout(rng.expovariate(rate))
                if env.now >= 1080:
                    break
                clients += 1
                env.process(request(env.now))

        env.process(arrivals())
        env.run()
        phases = {}
        for phase, selected in (("steady_15min", [row for row in results if row[0] < 900]), ("burst_3min", [row for row in results if row[0] >= 900])):
            latency = sorted(row[2] for row in selected if row[1] == 200)
            counts = dict(Counter(str(row[1]) for row in selected))
            phases[phase] = {"requests": len(selected), "status_counts": counts,
                             "answer_success_rate": counts.get("200", 0)/max(1,len(selected)),
                             "successful_p95_seconds": latency[int((len(latency)-1)*.95)] if latency else None}
        return phases

    scenarios = []
    for replicas in (1,2,4):
        for cache_hit in (.1,.6):
            for fault in (False, True):
                runs = [run(20261020+seed, replicas, cache_hit, fault) for seed in range(5)]
                aggregate = {}
                for phase in ("steady_15min", "burst_3min"):
                    rates = [item[phase]["answer_success_rate"] for item in runs]
                    mean = statistics.mean(rates)
                    margin = 2.776*statistics.stdev(rates)/(len(rates)**.5)
                    aggregate[phase] = {"mean_success_rate": mean, "seed_mean_95pct_ci": [max(0,mean-margin),min(1,mean+margin)]}
                scenarios.append({"service_replicas": replicas, "assumed_cache_hit": cache_hit, "assumed_model_fault": fault, "aggregate": aggregate, "runs": runs})
    write("simulation", {"summary": {"scenarios": len(scenarios), "runs": len(scenarios)*5, "calibration_samples": len(samples), "calibration_mean_seconds": statistics.mean(samples)},
        "kind": "SimPy discrete-event simulation, NOT measured multi-node capacity", "seed": 20261020,
        "assumptions": {"steady_arrivals_per_second": 1.5, "burst_arrivals_per_second": 30, "steady_seconds": 900, "burst_seconds": 180,
            "worker_slots_per_replica": 4, "queued_slots_per_replica": 8, "deadline_seconds": 40,
            "replicas": "Independent whole-service capacity with independent model capacity. Linear scale is hypothetical and not validated on this one GPU.",
            "service_time": "Empirical end-to-end model-enabled quality-run latency resampling at concurrency 2. Queue, network, and cache interactions are approximations; a sensitivity study, not a capacity guarantee.",
            "cache_hit": "Scenario input, not an observed production cache-hit rate.", "confidence": "Student-t interval across five simulation-seed means; excludes model and calibration uncertainty."}, "scenarios": scenarios})


def faults():
    import httpx
    from tools.reproduction_stack import configuration, environment, launch, listening, save
    state = configuration()
    observations = []
    # Pause only processes owned by the isolated reproduction manager. Always
    # resume them, even if a request fails or the experiment is interrupted.
    for name in ("redis", "model"):
        pid = state["processes"][name]["pid"]
        os.kill(pid, signal.SIGSTOP)
        children = []
        if name == "model":
            import subprocess
            output = subprocess.run(["pgrep", "-P", str(pid)], text=True, capture_output=True)
            children = [int(value) for value in output.stdout.split()]
            for child in children:
                os.kill(child, signal.SIGSTOP)
        try:
            row = post_question("解方程 7(x-301)=35。", index=99001 if name == "redis" else 99002, key=uuid.uuid4().hex)
            observations.append({"dependency": name, "during_fault": row})
        finally:
            for child in children:
                os.kill(child, signal.SIGCONT)
            os.kill(pid, signal.SIGCONT)
        time.sleep(2)
        observations[-1]["recovered"] = post_question("解方程 7(x-302)=35。", index=99003, key=uuid.uuid4().hex)
    old = state["processes"]["grpc"]
    os.killpg(old["pid"], signal.SIGTERM)
    try:
        for _ in range(100):
            if not listening(25051):
                break
            time.sleep(.1)
        attempts = [post_question("解方程 11(x-401)=22。", index=99100+index, key=uuid.uuid4().hex) for index in range(6)]
    finally:
        state["processes"].pop("grpc", None)
        save(state)
        launch("grpc", [sys.executable,"grpc_server.py"], state, environment(state), 25051)
    time.sleep(11)
    observations.append({"dependency":"grpc", "during_fault":attempts[-1],"all_attempts":attempts,
                         "recovered":post_question("解方程 11(x-402)=22。",index=99201,key=uuid.uuid4().hex)})
    write("faults", {"summary": {item["dependency"]: {"fault_status": item["during_fault"]["status"],
                    "fault_response_type": item["during_fault"].get("response", {}).get("response_type"),
                    "fault_latency_ms": item["during_fault"]["latency_ms"], "recovered": item["recovered"]["business_success"]} for item in observations},
                     "observations": observations, "scope": "Real process-pause fault injection into this isolated stack only; restored in finally blocks."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("storage", "simulate", "faults"))
    args = parser.parse_args()
    {"storage": storage, "simulate": simulate, "faults": faults}[args.mode]()
