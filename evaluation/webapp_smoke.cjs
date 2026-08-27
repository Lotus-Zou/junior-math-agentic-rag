const { chromium } = require("playwright");
const os = require("os");
const path = require("path");
const baseUrl = process.env.SMOKE_BASE_URL || "http://127.0.0.1:8000";

function assertNoInternalCopy(text, state) {
  for (const forbidden of [
    "critic", "模型", "检索", "超时", "model error", "retriev", "timeout",
    "本地生成练习", "本地确定性校验", "locally generated exercise", "local deterministic check",
  ]) {
    if (text.toLowerCase().includes(forbidden.toLowerCase())) {
      throw new Error(`${state} exposed internal copy: ${forbidden}`);
    }
  }
}

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const consoleErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });

  await page.goto(`${baseUrl}/`, { waitUntil: "networkidle" });
  const frontendSource = await page.locator('script[src="/static/app.js"]').getAttribute("src");
  if (frontendSource !== "/static/app.js") throw new Error("Frontend application script missing");
  await page.waitForFunction(() => document.querySelector("#serviceStatus")?.textContent.includes("服务正常"));
  if (!(await page.title()).includes("数问")) throw new Error("Chinese title missing");
  assertNoInternalCopy(await page.locator("body").innerText(), "Chinese first paint");

  await page.locator('[data-language="en"]').click();
  await page.getByRole("heading", { name: "Understand the mistake, not just the answer" }).waitFor();
  assertNoInternalCopy(await page.locator("body").innerText(), "English UI");
  await page.locator('[data-language="zh"]').click();

  const started = Date.now();
  await page.locator("#questionInput").fill("一次函数 y = -2x + 3 的斜率和截距分别是什么？图像大致怎么画？");
  await page.locator("#askForm").evaluate((form) => form.requestSubmit());
  const functionAnswer = page.locator(".message.assistant .assistant-answer").last();
  await functionAnswer.waitFor({ timeout: 8000 });
  await functionAnswer.getByText("斜率 k = -2", { exact: false }).waitFor();
  const elapsedMs = Date.now() - started;
  const functionText = await functionAnswer.innerText();
  if (elapsedMs > 3000) throw new Error(`Function response exceeded 3s: ${elapsedMs}ms`);
  if (functionText.includes("�") || functionText.includes("\\frac") || functionText.includes("\\boxed")) {
    throw new Error("Function response contains mojibake or raw LaTeX commands");
  }

  await page.locator("#newChatButton").click();
  await page.locator('[data-language="en"]').click();
  await page.locator("#questionInput").fill("For y = -2x + 3, what are the slope and intercept, and how should I sketch the graph?");
  await page.locator("#askForm").evaluate((form) => form.requestSubmit());
  const englishAnswer = page.locator(".message.assistant .assistant-answer").last();
  await englishAnswer.getByText("k = -2 and b = 3", { exact: false }).waitFor({ timeout: 3000 });
  if ((await englishAnswer.innerText()).includes("�")) throw new Error("English response contains mojibake");

  await page.locator("#newChatButton").click();
  await page.locator('[data-language="zh"]').click();
  await page.locator("#questionInput").fill("在△ABC和△DEF中，已知AB=DE，AC=DF，还需要什么条件才能证明两个三角形全等？");
  const congruenceStarted = Date.now();
  await page.locator("#askForm").evaluate((form) => form.requestSubmit());
  const congruenceAnswer = page.locator(".message.assistant .assistant-answer").last();
  await page.waitForFunction(
    () => document.querySelector(".message.assistant:last-of-type .assistant-answer")?.textContent.includes("边角边（SAS）"),
    { timeout: 3000 },
  );
  const congruenceResponseMs = Date.now() - congruenceStarted;
  const congruenceText = await congruenceAnswer.innerText();
  if (!congruenceText.includes("边角边（SAS）") || !congruenceText.includes("BC = EF")) {
    throw new Error("Congruence answer omitted the SSS alternative");
  }

  const switchStarted = Date.now();
  await page.locator("#questionInput").fill("换个问题");
  await page.locator("#askForm").evaluate((form) => form.requestSubmit());
  const switchAnswer = page.locator(".message.assistant .assistant-answer").last();
  await switchAnswer.getByText("已切换到新问题", { exact: false }).waitFor({ timeout: 3000 });
  const switchResponseMs = Date.now() - switchStarted;
  if (switchResponseMs > 3000) throw new Error(`Switch response exceeded 3s: ${switchResponseMs}ms`);

  await page.locator("#questionInput").fill("几何");
  const geometryRequest = page.waitForRequest((request) => request.url().endsWith("/ask") && request.method() === "POST");
  const geometryResponse = page.waitForResponse((response) => response.url().endsWith("/ask") && response.request().method() === "POST");
  await page.locator("#askForm").evaluate((form) => form.requestSubmit());
  const firstGeometryRequest = await geometryRequest;
  const firstGeometryResponse = await geometryResponse;
  const firstGeometryBody = await firstGeometryResponse.json();
  if (firstGeometryRequest.postDataJSON().exercise_state !== null) {
    throw new Error("First exercise request unexpectedly carried stale exercise state");
  }
  if (!firstGeometryBody.exercise_state?.exercise_id || !firstGeometryBody.exercise_state?.session_id) {
    throw new Error("Geometry response omitted opaque exercise state");
  }
  const geometryMessage = page.locator(".message.assistant").last();
  const geometryAnswer = geometryMessage.locator(".assistant-answer");
  await geometryAnswer.waitFor({ timeout: 8000 });
  await geometryAnswer.getByText("几何练习", { exact: false }).waitFor();
  const geometryText = await geometryAnswer.innerText();
  if (!geometryText.includes("答案暂不展示")) {
    throw new Error(`Geometry exercise revealed or omitted answer-hiding state: ${JSON.stringify(geometryText)}`);
  }
  if (geometryText.includes("复杂推理服务") || geometryText.includes("超时") || geometryText.includes("Critic")) {
    throw new Error("Geometry exercise exposed internal failure copy");
  }
  await geometryMessage.locator(".meta-chip").filter({ hasText: /^练习$/ }).waitFor();
  assertNoInternalCopy(await page.locator("body").innerText(), "Chinese rendered conversation");

  await page.locator("#questionInput").fill("难一点");
  const harderRequest = page.waitForRequest((request) => request.url().endsWith("/ask") && request.method() === "POST");
  const harderResponse = page.waitForResponse((response) => response.url().endsWith("/ask") && response.request().method() === "POST");
  await page.locator("#askForm").evaluate((form) => form.requestSubmit());
  const harderRequestBody = (await harderRequest).postDataJSON();
  const harderResponseBody = await (await harderResponse).json();
  if (harderRequestBody.exercise_state?.exercise_id !== firstGeometryBody.exercise_state.exercise_id) {
    throw new Error("Frontend did not return the active opaque exercise state");
  }
  if (harderResponseBody.exercise_state?.session_id !== firstGeometryBody.exercise_state.session_id) {
    throw new Error("Harder exercise did not preserve the adaptive session");
  }
  if (harderResponseBody.exercise_state?.fingerprint === firstGeometryBody.exercise_state.fingerprint) {
    throw new Error("Harder exercise repeated the previous problem");
  }
  if (harderResponseBody.exercise_state?.difficulty <= firstGeometryBody.exercise_state.difficulty) {
    throw new Error("Harder command did not increase the exercise difficulty");
  }
  const harderMessage = page.locator(".message.assistant").last();
  await harderMessage.locator(".assistant-answer").getByText("几何练习", { exact: false }).waitFor();
  assertNoInternalCopy(await page.locator("body").innerText(), "Adaptive harder exercise");

  await page.locator('[data-language="en"]').click();
  await harderMessage.locator(".meta-chip").filter({ hasText: /^Practice$/ }).waitFor();
  assertNoInternalCopy(await page.locator("body").innerText(), "English language switch with conversation");
  await page.locator('[data-language="zh"]').click();
  await harderMessage.locator(".meta-chip").filter({ hasText: /^练习$/ }).waitFor();
  assertNoInternalCopy(await page.locator("body").innerText(), "Chinese language switch with conversation");

  if (consoleErrors.length) throw new Error(`Console errors: ${consoleErrors.join(" | ")}`);
  const screenshot = process.env.SMOKE_SCREENSHOT_PATH || path.join(os.tmpdir(), "agentirag-webapp-smoke.png");
  await page.screenshot({ path: screenshot, fullPage: true });
  console.log(JSON.stringify({
    passed: true,
    function_response_ms: elapsedMs,
    congruence_response_ms: congruenceResponseMs,
    switch_response_ms: switchResponseMs,
    screenshot,
  }, null, 2));
  await browser.close();
})().catch((error) => {
  console.error(error.stack || String(error));
  process.exit(1);
});
