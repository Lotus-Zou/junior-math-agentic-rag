const translations = {
  zh: {
    brand: "数问", brandSubtitle: "初中数学错题助手", newCorrection: "新建订正", currentSession: "当前会话",
    untitledSession: "未命名错题", checkingService: "正在检查服务", serviceOnline: "服务正常", serviceOffline: "服务未连接",
    apiDocs: "开发者 API", workspaceTitle: "智能订正", workspaceSubtitle: "先参考教材，再检查每一步",
    welcomeEyebrow: "从错误步骤开始", welcomeTitle: "把错题弄懂，不只算出答案",
    welcomeCopy: "输入完整题目和你的错误作答。系统会参考教材、定位第一处错误，并在回答前检查解题步骤。",
    equationTitle: "方程移项", equationCopy: "为什么移项后要变号？", geometryTitle: "几何证明", geometryCopy: "全等条件应该怎么选？",
    functionTitle: "一次函数", functionCopy: "斜率和截距怎么看？", stepInput: "输入完整题目", stepInputCopy: "保留公式、数字和图形条件",
    stepMistake: "补充错误作答", stepMistakeCopy: "便于定位第一处错误", stepFollow: "根据提示继续追问", stepFollowCopy: "会话会保留当前题目状态",
    addMistake: "添加我的错误作答", mistakeLabel: "我的错误步骤或答案", mistakePlaceholder: "例如：我移项后写成 2x = 11 + 3",
    questionLabel: "数学题目", questionPlaceholder: "粘贴完整题目，或继续追问上一步……", composerNote: "回答会参考教材并检查解题步骤；题设不足时会先向你追问。",
    evidenceLabel: "回答依据", evidenceTitle: "教材来源", helpEyebrow: "三步完成订正", helpTitle: "如何使用数问",
    helpOneTitle: "输入原题", helpOneCopy: "尽量包含完整题干、选项和图形中的已知条件，公式可以直接输入。",
    helpTwoTitle: "写下你的做法", helpTwoCopy: "点击“添加我的错误作答”，写下卡住的位置或算错的步骤。",
    helpThreeTitle: "继续追问", helpThreeCopy: "收到分析后可直接问“为什么变号”或“下一步怎么想”，无需重复题目。",
    startUsing: "开始使用", wrongPrefix: "我的作答：", tutor: "数问", analyzing: "正在分析这道题",
    parseStep: "读懂题目", retrieveStep: "查看教材", verifyStep: "检查步骤", validated: "解题步骤已检查", notValidated: "还需要补充信息",
    practice: "练习", needsMoreInfo: "需要补充信息", scopeLimited: "当前请求暂不支持",
    sources: "查看来源", noSources: "本次回答没有可展示的来源。", sourceUnknown: "教材片段", chapter: "章节", rank: "排序",
    helpful: "回答有帮助", incorrect: "回答有问题", feedbackThanks: "反馈已记录", feedbackFailed: "反馈提交失败",
    emptyQuestion: "请先输入完整题目。", requestFailed: "请补充完整题干、图形条件或需要核对的步骤。", timeout: "请补充完整题干、图形条件或需要核对的步骤。",
    newSessionReady: "已开始新的订正", sourcesCount: "条来源", latency: "耗时", cached: "缓存命中",
    menuLabel: "打开菜单", helpLabel: "使用说明", sendLabel: "发送题目", closeLabel: "关闭",
    examples: {
      equation: "解方程 2x + 3 = 11。我移项后写成 2x = 11 + 3，错在哪里？",
      geometry: "在△ABC和△DEF中，已知AB=DE，AC=DF，还需要什么条件才能证明两个三角形全等？",
      function: "一次函数 y = -2x + 3 的斜率和截距分别是什么？图像大致怎么画？"
    }
  },
  en: {
    brand: "MathTrace", brandSubtitle: "Junior Math Mistake Tutor", newCorrection: "New correction", currentSession: "Current session",
    untitledSession: "Untitled problem", checkingService: "Checking service", serviceOnline: "Service online", serviceOffline: "Service offline",
    apiDocs: "Developer API", workspaceTitle: "Guided correction", workspaceSubtitle: "Use the textbook, then check each step",
    welcomeEyebrow: "Start from the mistake", welcomeTitle: "Understand the mistake, not just the answer",
    welcomeCopy: "Enter the full problem and your incorrect attempt. The tutor uses textbook examples, finds the first error, and checks each step before replying.",
    equationTitle: "Moving terms", equationCopy: "Why does the sign change?", geometryTitle: "Geometry proof", geometryCopy: "Which congruence rule applies?",
    functionTitle: "Linear function", functionCopy: "How do slope and intercept work?", stepInput: "Enter the full problem", stepInputCopy: "Keep formulas, values, and diagram conditions",
    stepMistake: "Add your attempt", stepMistakeCopy: "Helps locate the first error", stepFollow: "Ask a follow-up", stepFollowCopy: "The current problem stays in context",
    addMistake: "Add my incorrect attempt", mistakeLabel: "My incorrect steps or answer", mistakePlaceholder: "Example: I changed it to 2x = 11 + 3",
    questionLabel: "Math problem", questionPlaceholder: "Paste the full problem or ask about the previous step...", composerNote: "Answers refer to the textbook and check each step. Missing conditions prompt one follow-up question.",
    evidenceLabel: "Answer evidence", evidenceTitle: "Textbook sources", helpEyebrow: "Correct in three steps", helpTitle: "How to use MathTrace",
    helpOneTitle: "Enter the problem", helpOneCopy: "Include the full prompt, choices, and known diagram conditions. You can type formulas directly.",
    helpTwoTitle: "Add your work", helpTwoCopy: "Select Add my incorrect attempt and enter where you got stuck or made a mistake.",
    helpThreeTitle: "Ask follow-ups", helpThreeCopy: "After the analysis, ask why a sign changed or what to try next without repeating the problem.",
    startUsing: "Start", wrongPrefix: "My attempt:", tutor: "MathTrace", analyzing: "Analyzing this problem",
    parseStep: "Read problem", retrieveStep: "Review textbook", verifyStep: "Check steps", validated: "Steps checked", notValidated: "More information needed",
    practice: "Practice", needsMoreInfo: "More information needed", scopeLimited: "Request not supported",
    sources: "View sources", noSources: "No displayable sources for this response.", sourceUnknown: "Textbook excerpt", chapter: "Chapter", rank: "Rank",
    helpful: "Helpful answer", incorrect: "Report an issue", feedbackThanks: "Feedback recorded", feedbackFailed: "Could not submit feedback",
    emptyQuestion: "Enter the complete problem first.", requestFailed: "Please add the full problem, diagram conditions, or the step you want checked.", timeout: "Please add the full problem, diagram conditions, or the step you want checked.",
    newSessionReady: "New correction started", sourcesCount: "sources", latency: "Latency", cached: "Cached",
    menuLabel: "Open menu", helpLabel: "How to use", sendLabel: "Send problem", closeLabel: "Close",
    examples: {
      equation: "Solve 2x + 3 = 11. I moved the term and wrote 2x = 11 + 3. Where did I go wrong?",
      geometry: "In triangles ABC and DEF, AB = DE and AC = DF. What other condition proves they are congruent?",
      function: "For y = -2x + 3, what are the slope and intercept, and how should I sketch the graph?"
    }
  }
};

const state = {
  language: localStorage.getItem("mathtrace-language") || "zh",
  history: [],
  summary: "",
  busy: false,
  latestResult: null
};

const elements = {
  form: document.getElementById("askForm"), question: document.getElementById("questionInput"), wrong: document.getElementById("wrongAnswer"),
  mistakeToggle: document.getElementById("mistakeToggle"), mistakeField: document.getElementById("mistakeField"), send: document.getElementById("sendButton"),
  welcome: document.getElementById("welcome"), list: document.getElementById("messageList"), scroll: document.getElementById("conversationScroll"),
  sourcePanel: document.getElementById("sourcePanel"), sourceList: document.getElementById("sourceList"), validation: document.getElementById("validationSummary"),
  contentLayout: document.querySelector(".content-layout"), help: document.getElementById("helpDialog"), sidebar: document.getElementById("sidebar"),
  statusDot: document.getElementById("statusDot"), serviceStatus: document.getElementById("serviceStatus"), sessionTitle: document.getElementById("sessionTitle"),
  toast: document.getElementById("toast")
};

function t(key) { return translations[state.language][key] ?? key; }

function responseLabel(responseType) {
  const responseLabels = {
    verified_answer: t("validated"),
    guided_exercise: t("practice"),
    clarification_required: t("needsMoreInfo"),
    supported_refusal: t("scopeLimited")
  };
  return responseLabels[responseType] ?? t("needsMoreInfo");
}

function applyLanguage(language) {
  state.language = language;
  localStorage.setItem("mathtrace-language", language);
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => { node.placeholder = t(node.dataset.i18nPlaceholder); });
  document.querySelectorAll("[data-language]").forEach((button) => button.classList.toggle("active", button.dataset.language === language));
  const accessibleLabels = [
    ["menuButton", "menuLabel"], ["helpButton", "helpLabel"], ["sendButton", "sendLabel"],
    ["closeHelp", "closeLabel"], ["closeSources", "closeLabel"]
  ];
  accessibleLabels.forEach(([id, key]) => {
    const node = document.getElementById(id);
    node.setAttribute("aria-label", t(key));
    node.title = t(key);
  });
  if (elements.statusDot.classList.contains("online")) elements.serviceStatus.textContent = t("serviceOnline");
  if (elements.statusDot.classList.contains("offline")) elements.serviceStatus.textContent = t("serviceOffline");
  document.querySelectorAll("[data-response-type]").forEach((node) => {
    node.textContent = responseLabel(node.dataset.responseType);
  });
  document.title = language === "zh" ? "数问 · 初中数学错题助手" : "MathTrace · Junior Math Mistake Tutor";
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  window.setTimeout(() => elements.toast.classList.remove("visible"), 2200);
}

function resizeTextarea(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
}

function scrollToLatest() {
  window.requestAnimationFrame(() => { elements.scroll.scrollTop = elements.scroll.scrollHeight; });
}

function appendUserMessage(question, wrongAnswer) {
  const article = document.createElement("article");
  article.className = "message user";
  const content = document.createElement("div");
  content.className = "message-content";
  const questionText = document.createElement("p");
  questionText.textContent = question;
  content.appendChild(questionText);
  if (wrongAnswer) {
    const wrong = document.createElement("p");
    wrong.className = "wrong-answer";
    wrong.textContent = `${t("wrongPrefix")} ${wrongAnswer}`;
    content.appendChild(wrong);
  }
  article.appendChild(content);
  elements.list.appendChild(article);
}

function createLoadingMessage() {
  const article = document.createElement("article");
  article.className = "message loading-message";
  article.innerHTML = `<div class="avatar" aria-hidden="true">Σ</div><div class="message-content"><p class="loading-title"></p><div class="pipeline"><span></span><span></span><span></span></div></div>`;
  article.querySelector(".loading-title").textContent = t("analyzing");
  const stepKeys = ["parseStep", "retrieveStep", "verifyStep"];
  const steps = [...article.querySelectorAll(".pipeline span")];
  steps.forEach((step, index) => { step.textContent = t(stepKeys[index]); });
  steps[0].classList.add("active");
  let active = 0;
  const timer = window.setInterval(() => {
    steps[active].classList.remove("active");
    active = Math.min(active + 1, steps.length - 1);
    steps[active].classList.add("active");
  }, 2800);
  article.dataset.timer = String(timer);
  elements.list.appendChild(article);
  return article;
}

function addMetaChip(container, label, className = "") {
  const chip = document.createElement("span");
  chip.className = `meta-chip ${className}`.trim();
  chip.textContent = label;
  container.appendChild(chip);
  return chip;
}

function readableMath(source) {
  let value = String(source || "").trim();
  for (let index = 0; index < 4; index += 1) {
    value = value.replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "($1)/($2)");
    value = value.replace(/\\sqrt\s*\{([^{}]*)\}/g, "√($1)");
    value = value.replace(/\\boxed\s*\{([^{}]*)\}/g, "$1");
  }
  const symbols = {
    "\\times": "×", "\\cdot": "·", "\\div": "÷", "\\ne": "≠", "\\neq": "≠",
    "\\le": "≤", "\\leq": "≤", "\\ge": "≥", "\\geq": "≥", "\\pm": "±",
    "\\Delta": "Δ", "\\theta": "θ", "\\angle": "∠", "\\perp": "⊥", "\\parallel": "∥"
  };
  Object.entries(symbols).forEach(([command, symbol]) => { value = value.split(command).join(symbol); });
  return value
    .replace(/\\(?:left|right|displaystyle)/g, "")
    .replace(/\\(?:qquad|quad|,|;|!)/g, " ")
    .replace(/\^\{([^{}]+)\}/g, "^$1")
    .replace(/_\{([^{}]+)\}/g, "_$1")
    .replace(/[{}]/g, "")
    .replace(/\\([A-Za-z]+)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

function appendInlineContent(container, source) {
  const tokenPattern = /(\*\*[^*]+\*\*|\\\([\s\S]*?\\\)|\[[1-9]\d*\])/g;
  let cursor = 0;
  for (const match of source.matchAll(tokenPattern)) {
    if (match.index > cursor) container.appendChild(document.createTextNode(source.slice(cursor, match.index)));
    const token = match[0];
    if (token.startsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      container.appendChild(strong);
    } else if (token.startsWith("\\(")) {
      const math = document.createElement("span");
      math.className = "math-inline";
      math.textContent = readableMath(token.slice(2, -2));
      container.appendChild(math);
    } else {
      const citation = document.createElement("sup");
      citation.className = "citation";
      citation.textContent = token;
      container.appendChild(citation);
    }
    cursor = match.index + token.length;
  }
  if (cursor < source.length) container.appendChild(document.createTextNode(source.slice(cursor)));
}

function renderTextBlocks(container, source) {
  source.split(/\n{2,}/).map((block) => block.trim()).filter(Boolean).forEach((block) => {
    const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);
    const heading = lines[0]?.match(/^\*\*(.+)\*\*$/) || lines[0]?.match(/^#{1,4}\s+(.+)$/);
    if (heading) {
      const title = document.createElement("h3");
      appendInlineContent(title, heading[1]);
      container.appendChild(title);
      if (lines.length > 1) renderTextBlocks(container, lines.slice(1).join("\n"));
      return;
    }
    if (lines.length && lines.every((line) => /^[-*]\s+/.test(line))) {
      const list = document.createElement("ul");
      lines.forEach((line) => {
        const item = document.createElement("li");
        appendInlineContent(item, line.replace(/^[-*]\s+/, ""));
        list.appendChild(item);
      });
      container.appendChild(list);
      return;
    }
    if (lines.length && lines.every((line) => /^\d+[.)]\s+/.test(line))) {
      const list = document.createElement("ol");
      lines.forEach((line) => {
        const item = document.createElement("li");
        appendInlineContent(item, line.replace(/^\d+[.)]\s+/, ""));
        list.appendChild(item);
      });
      container.appendChild(list);
      return;
    }
    const paragraph = document.createElement("p");
    appendInlineContent(paragraph, lines.join("\n"));
    container.appendChild(paragraph);
  });
}

function renderRichAnswer(container, answer) {
  container.replaceChildren();
  const source = String(answer || t("requestFailed"))
    .replace(/\r\n?/g, "\n")
    .replace(/^\s*\[\s*$/gm, "\\[")
    .replace(/^\s*\]\s*$/gm, "\\]");
  const displayPattern = /\\\[([\s\S]*?)\\\]/g;
  let cursor = 0;
  for (const match of source.matchAll(displayPattern)) {
    renderTextBlocks(container, source.slice(cursor, match.index));
    const math = document.createElement("div");
    math.className = "math-block";
    math.textContent = readableMath(match[1]);
    container.appendChild(math);
    cursor = match.index + match[0].length;
  }
  renderTextBlocks(container, source.slice(cursor));
}

function appendAssistantMessage(result) {
  const article = document.createElement("article");
  article.className = "message assistant";
  article.innerHTML = `<div class="avatar" aria-hidden="true">Σ</div><div class="message-content"><div class="assistant-answer"></div><div class="answer-meta"></div><div class="message-actions"></div></div>`;
  renderRichAnswer(article.querySelector(".assistant-answer"), result.answer);
  const meta = article.querySelector(".answer-meta");
  const responseChip = addMetaChip(meta, responseLabel(result.response_type), result.validation_passed ? "valid" : "invalid");
  responseChip.dataset.responseType = result.response_type;
  (result.knowledge_points || []).slice(0, 4).forEach((point) => addMetaChip(meta, point));
  if (result.sources?.length) addMetaChip(meta, `${result.sources.length} ${t("sourcesCount")}`);
  if (result.metrics?.latency_ms) addMetaChip(meta, `${t("latency")} ${(result.metrics.latency_ms / 1000).toFixed(1)}s`);
  if (result.cached) addMetaChip(meta, t("cached"));

  const actions = article.querySelector(".message-actions");
  const sources = document.createElement("button");
  sources.type = "button";
  sources.textContent = "≡";
  sources.title = t("sources");
  sources.setAttribute("aria-label", t("sources"));
  sources.addEventListener("click", () => showSources(result));
  actions.appendChild(sources);
  [["✓", true, "helpful"], ["!", false, "incorrect"]].forEach(([symbol, correct, label]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = symbol;
    button.title = t(label);
    button.setAttribute("aria-label", t(label));
    button.addEventListener("click", () => submitFeedback(result.trace_id, correct, button));
    actions.appendChild(button);
  });
  elements.list.appendChild(article);
}

function showSources(result) {
  elements.sourceList.replaceChildren();
  elements.validation.className = `validation-summary${result.validation_passed ? "" : " invalid"}`;
  elements.validation.textContent = responseLabel(result.response_type);
  elements.validation.dataset.responseType = result.response_type;
  if (!result.sources?.length) {
    const empty = document.createElement("p");
    empty.className = "empty-sources";
    empty.textContent = t("noSources");
    elements.sourceList.appendChild(empty);
  } else {
    result.sources.forEach((source, index) => {
      const item = document.createElement("article");
      item.className = "source-item";
      const title = document.createElement("strong");
      title.textContent = source.source || `${t("sourceUnknown")} ${index + 1}`;
      const detail = document.createElement("p");
      detail.textContent = `${t("chapter")}: ${source.chapter || "-"} · ${t("rank")}: ${source.rank || index + 1}`;
      item.append(title, detail);
      elements.sourceList.appendChild(item);
    });
  }
  elements.contentLayout.classList.add("sources-open");
}

async function submitFeedback(traceId, correct, button) {
  if (!traceId) return;
  try {
    const response = await fetch("/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trace_id: traceId, correct, comment: "" }) });
    if (!response.ok) throw new Error("feedback failed");
    button.classList.add("selected");
    showToast(t("feedbackThanks"));
  } catch { showToast(t("feedbackFailed")); }
}

async function submitQuestion(event) {
  event.preventDefault();
  if (state.busy) return;
  const question = elements.question.value.trim();
  const wrongAnswer = elements.wrong.value.trim();
  if (!question) { showToast(t("emptyQuestion")); elements.question.focus(); return; }

  state.busy = true;
  elements.send.disabled = true;
  elements.welcome.hidden = true;
  elements.sessionTitle.textContent = question.length > 24 ? `${question.slice(0, 24)}…` : question;
  appendUserMessage(question, wrongAnswer);
  const loading = createLoadingMessage();
  elements.question.value = "";
  resizeTextarea(elements.question);
  scrollToLatest();

  const query = wrongAnswer ? `${question}\n\n${state.language === "zh" ? "学生错误作答" : "Student's incorrect attempt"}: ${wrongAnswer}` : question;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, language: state.language, conversation_history: state.history, conversation_summary: state.summary }),
      signal: controller.signal
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "request failed");
    if ("conversation_history" in result) state.history = result.conversation_history;
    if ("conversation_summary" in result) state.summary = result.conversation_summary;
    state.latestResult = result;
    appendAssistantMessage(result);
  } catch (error) {
    appendAssistantMessage({ answer: error.name === "AbortError" ? t("timeout") : t("requestFailed"), response_type: "clarification_required", validation_passed: false, sources: [] });
  } finally {
    window.clearTimeout(timeoutId);
    window.clearInterval(Number(loading.dataset.timer));
    loading.remove();
    state.busy = false;
    elements.send.disabled = false;
    elements.question.focus();
    scrollToLatest();
  }
}

function resetConversation() {
  state.history = [];
  state.summary = "";
  state.latestResult = null;
  elements.list.replaceChildren();
  elements.welcome.hidden = false;
  elements.question.value = "";
  elements.wrong.value = "";
  elements.mistakeField.hidden = true;
  elements.mistakeToggle.setAttribute("aria-expanded", "false");
  elements.sessionTitle.textContent = t("untitledSession");
  elements.contentLayout.classList.remove("sources-open");
  elements.sidebar.classList.remove("open");
  showToast(t("newSessionReady"));
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error();
    elements.statusDot.className = "status-dot online";
    elements.serviceStatus.textContent = t("serviceOnline");
  } catch {
    elements.statusDot.className = "status-dot offline";
    elements.serviceStatus.textContent = t("serviceOffline");
  }
}

document.querySelectorAll("[data-language]").forEach((button) => button.addEventListener("click", () => applyLanguage(button.dataset.language)));
document.querySelectorAll("[data-example]").forEach((button) => button.addEventListener("click", () => {
  elements.question.value = translations[state.language].examples[button.dataset.example];
  resizeTextarea(elements.question);
  elements.question.focus();
}));
elements.form.addEventListener("submit", submitQuestion);
elements.question.addEventListener("input", () => resizeTextarea(elements.question));
elements.wrong.addEventListener("input", () => resizeTextarea(elements.wrong));
elements.mistakeToggle.addEventListener("click", () => {
  const open = elements.mistakeToggle.getAttribute("aria-expanded") === "true";
  elements.mistakeToggle.setAttribute("aria-expanded", String(!open));
  elements.mistakeField.hidden = open;
  if (!open) elements.wrong.focus();
});
document.getElementById("newChatButton").addEventListener("click", resetConversation);
document.getElementById("helpButton").addEventListener("click", () => elements.help.showModal());
document.getElementById("closeHelp").addEventListener("click", () => elements.help.close());
document.getElementById("startButton").addEventListener("click", () => { elements.help.close(); elements.question.focus(); });
document.getElementById("closeSources").addEventListener("click", () => elements.contentLayout.classList.remove("sources-open"));
document.getElementById("menuButton").addEventListener("click", () => elements.sidebar.classList.toggle("open"));
elements.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) elements.form.requestSubmit();
});

applyLanguage(state.language);
checkHealth();
