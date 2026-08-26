const { chromium } = require("playwright");
const path = require("path");
const baseUrl = process.env.SMOKE_BASE_URL || "http://127.0.0.1:8000";

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
  await page.waitForFunction(() => document.querySelector("#serviceStatus")?.textContent.includes("服务正常"));
  if (!(await page.title()).includes("数问")) throw new Error("Chinese title missing");

  await page.locator('[data-language="en"]').click();
  await page.getByRole("heading", { name: "Understand the mistake, not just the answer" }).waitFor();
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

  await page.locator("#questionInput").fill("出一个几何体我做做");
  await page.locator("#askForm").evaluate((form) => form.requestSubmit());
  const geometryAnswer = page.locator(".message.assistant .assistant-answer").last();
  await geometryAnswer.waitFor({ timeout: 8000 });
  await geometryAnswer.getByText("几何练习", { exact: false }).waitFor();
  const geometryText = await geometryAnswer.innerText();
  if (!geometryText.includes("答案暂不展示")) throw new Error("Geometry exercise revealed or omitted answer-hiding state");
  if (geometryText.includes("复杂推理服务") || geometryText.includes("超时") || geometryText.includes("Critic")) {
    throw new Error("Geometry exercise exposed internal failure copy");
  }

  if (consoleErrors.length) throw new Error(`Console errors: ${consoleErrors.join(" | ")}`);
  const screenshot = path.resolve("evaluation", "skill_review", "iteration-1", "webapp-smoke.png");
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
