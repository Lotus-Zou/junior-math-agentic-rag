import test from "node:test";
import assert from "node:assert/strict";

async function withApi(mockFetch, run) {
  const original = globalThis.fetch;
  globalThis.fetch = mockFetch;
  try {
    const api = await import(`./src/api.ts?test=${crypto.randomUUID()}`);
    await run(api);
  } finally {
    globalThis.fetch = original;
  }
}

test("learning requests use the selected session before any chat request", async () => {
  const requests = [];
  await withApi(async (path, init) => {
    if (path === "/auth/session") return Response.json({ user_id: "signed-student" });
    requests.push({ path, init });
    return Response.json(path === "/tasks" ? { task_id: "task-1", status: "queued" } : []);
  }, async (api) => {
    await api.getMistakes({ user_id: "local-id", session_id: "first" });
    await api.getLearningReport({ user_id: "local-id", session_id: "second" });
    await api.sendFeedback("trace", false, { user_id: "local-id", session_id: "first" });
    await api.createLearningReport({ user_id: "local-id", session_id: "second" });
    assert.deepEqual(requests.map(({ init }) => init.headers["X-Session-ID"]), ["first", "second", "first", "second"]);
    assert.ok(requests.every(({ init }) => init.headers["X-User-ID"] === "signed-student"));
    const body = JSON.parse(requests.at(-1).init.body);
    assert.equal(body.kind, "learning_report");
    assert.equal(body.user_id, undefined);
  });
});

test("direct API development mode retains explicit user and session scope", async () => {
  await withApi(async (path, init) => {
    if (path === "/auth/session") return new Response(null, { status: 404 });
    assert.equal(init.headers["X-User-ID"], "dev-user");
    assert.equal(init.headers["X-Session-ID"], "dev-session");
    return Response.json([]);
  }, async (api) => api.getMistakes({ user_id: "dev-user", session_id: "dev-session" }));
});

test("queue overload and outages do not appear as accepted tasks", async () => {
  for (const [status, message] of [[429, "queue_busy"], [503, "service_unavailable"]]) {
    await withApi(async (path) => path === "/auth/session" ? Response.json({ user_id: "student" }) : Response.json({ detail: "busy" }, { status }),
      async (api) => assert.rejects(api.createLearningReport({ user_id: "student", session_id: "session" }), { message }));
  }
});

test("cancelled learning queries propagate abort signals", async () => {
  const controller = new AbortController();
  controller.abort();
  await withApi(async (path, init) => {
    if (path === "/auth/session") return Response.json({ user_id: "student" });
    assert.equal(init.signal.aborted, true);
    init.signal.throwIfAborted();
  }, async (api) => assert.rejects(api.getMistakes({ user_id: "student", session_id: "session" }, controller.signal), { name: "AbortError" }));
});
