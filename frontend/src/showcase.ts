import type { AskResponse, ConversationTurn, ExerciseState, Language, LearningReport, LearningTask, MistakeQuestion, SessionIdentity } from "./types";

export const examples = [
  {
    id: "equation", title: "一元一次方程", grade: 7,
    question: "解方程：3(x - 2) = 2x + 5。", wrong: "3x - 2 = 2x + 5，所以 x = 7。",
    points: ["一元一次方程", "去括号"], error: "去括号漏乘",
    answer: "### 先找错因\n去括号时，3 要乘括号内的每一项，因此 $3(x-2)=3x-6$，不是 $3x-2$。\n\n### 分步求解\n1. 去括号：$3x-6=2x+5$。\n2. 两边同时减去 $2x$：$x-6=5$。\n3. 两边同时加上 6：$x=11$。\n\n**代入检查：** 左边 $3(11-2)=27$，右边 $2\\times11+5=27$，相等。\n\n**答案：$x=11$。**",
    followup: "为什么去括号要乘每一项？",
    explanation: "分配律是 $a(b+c)=ab+ac$。这里把减法看成加上负数：$3(x-2)=3[x+(-2)]=3x+3\\times(-2)=3x-6$。可以代入 $x=4$ 比较：原式等于 6，$3x-6$ 也等于 6，而 $3x-2$ 等于 10。",
  },
  {
    id: "function", title: "一次函数", grade: 8,
    question: "一次函数经过点 (1, 3) 和 (3, 7)，求函数解析式。", wrong: "斜率是 (7 - 3) / (3 + 1) = 1。",
    points: ["一次函数", "待定系数法"], error: "斜率分母误用加法",
    answer: "### 先找错因\n斜率是纵坐标的变化量除以横坐标的变化量，分母应是 $3-1$，不是 $3+1$。\n\n### 分步求解\n1. 设 $y=kx+b$。\n2. $k=\\frac{7-3}{3-1}=2$。\n3. 代入 $(1,3)$：$3=2+b$，得 $b=1$。\n\n**答案：$y=2x+1$。** 代入 $x=3$ 得 $y=7$，与第二个点一致。",
    followup: "斜率为什么用坐标的差？",
    explanation: "两点满足 $y_1=kx_1+b$ 和 $y_2=kx_2+b$。两式相减，截距 $b$ 消去，得到 $y_2-y_1=k(x_2-x_1)$。本题两点横坐标不同，可以除以 $x_2-x_1$，所以 $k=(y_2-y_1)/(x_2-x_1)$。",
  },
  {
    id: "pythagoras", title: "勾股定理", grade: 8,
    question: "直角三角形的两条直角边分别长 6 和 8，求斜边长。", wrong: "斜边长为 6 + 8 = 14。",
    points: ["勾股定理", "算术平方根"], error: "把平方和误作边长和",
    answer: "### 先找错因\n勾股定理联系的是三边的**平方**，不能直接把两条直角边相加。\n\n### 分步求解\n设斜边长为 $c$，则\n$$c^2=6^2+8^2=36+64=100.$$\n边长为正，故 $c=\\sqrt{100}=10$。\n\n**答案：10。** 同时 $8<10<6+8$，符合斜边最长和三角形两边之和大于第三边。",
    followup: "为什么不能取负的平方根？",
    explanation: "方程 $c^2=100$ 的实数解是 $c=10$ 或 $c=-10$，但这里 $c$ 表示边长，必须大于零。所以只保留 $c=10$。这体现了代数解还需要满足题目中的实际约束。",
  },
  {
    id: "quadratic", title: "一元二次方程", grade: 9,
    question: "解方程：x² - 5x + 6 = 0。", wrong: "(x - 2)(x - 3) = 0，所以 x = -2 或 -3。",
    points: ["一元二次方程", "因式分解"], error: "因式为零时移项符号错误",
    answer: "### 先找错因\n因式分解正确，但 $x-2=0$ 应解得 $x=2$，不是 $-2$。\n\n### 分步求解\n1. 寻找积为 6、和为 $-5$ 的两个数：$-2$ 和 $-3$。\n2. 因式分解：$(x-2)(x-3)=0$。\n3. 因此 $x-2=0$ 或 $x-3=0$。\n\n**答案：$x=2$ 或 $x=3$。** 代入原式分别得到 $4-10+6=0$ 和 $9-15+6=0$。",
    followup: "为什么乘积为零就至少一个因式为零？",
    explanation: "在实数范围，若 $ab=0$：当 $a=0$ 时结论成立；当 $a\\ne0$ 时，两边可同时除以 $a$，得到 $b=0$。因此乘积为零等价于至少一个因式为零。注意不能一开始就除以 $x-2$，否则会丢掉解 $x=2$。",
  },
  {
    id: "probability", title: "基础概率", grade: 9,
    question: "同时掷两枚均匀硬币，恰好一枚正面朝上的概率是多少？", wrong: "结果有两正、一正、两反三种，所以概率是 1/3。",
    points: ["概率", "等可能样本空间"], error: "把不等可能事件当作等可能结果",
    answer: "### 先找错因\n“一正一反”包含两个基本结果，不能把正面数量的三种取值当作等可能结果。\n\n### 分步求解\n给两枚硬币编号，等可能结果为：正正、正反、反正、反反。\n其中恰好一枚正面的结果是正反、反正，共 2 个。\n\n$$P=\\frac{2}{4}=\\frac{1}{2}.$$\n\n**答案：$1/2$。**",
    followup: "正反和反正为什么是两种结果？",
    explanation: "两枚硬币可以区分，例如第一枚和第二枚。正反表示第一枚正、第二枚反；反正表示第一枚反、第二枚正。它们各自概率为 $1/4$，合起来才是“一正一反”的 $1/2$。不编号也能研究概率，但不能忽略这种事件包含两个等可能基本结果。",
  },
  {
    id: "inequality", title: "一元一次不等式", grade: 7,
    question: "解不等式：-2x + 3 > 7。", wrong: "-2x > 4，所以 x > -2。",
    points: ["一元一次不等式", "不等式基本性质"], error: "除以负数未改变不等号方向",
    answer: "### 先找错因\n不等式两边同时除以负数时，不等号方向必须改变。\n\n### 分步求解\n1. 两边减 3：$-2x>4$。\n2. 两边除以 $-2$，不等号反向：$x<-2$。\n\n**答案：$x<-2$。** 取 $x=-3$，左边为 9，大于 7；边界 $x=-2$ 使左边等于 7，不满足严格大于。",
    followup: "为什么除以负数要改变不等号？",
    explanation: "如果 $a>b$，则 $a-b>0$。乘以负数 $c$ 后，$c(a-b)<0$，即 $ca<cb$，所以大小关系反向。除以负数等于乘以它的倒数，倒数仍为负数，因此也必须反向。例如 $4>2$，但 $-4<-2$。",
  },
] as const;

type Input = SessionIdentity & { query: string; language: Language; conversation_history: ConversationTurn[]; conversation_summary: string; exercise_state: ExerciseState | null };
type LocalData = { version: 1; mistakes: MistakeQuestion[]; feedback: Record<string, boolean>; tasks: Record<string, LearningTask> };
const emptyData = (): LocalData => ({ version: 1, mistakes: [], feedback: {}, tasks: {} });
const key = (scope: SessionIdentity) => `mathtrace-showcase-v1:${JSON.stringify([scope.user_id, scope.session_id])}`;

function read(scope: SessionIdentity): LocalData {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(key(scope)) || "null");
    if (!parsed || typeof parsed !== "object") return emptyData();
    const data = parsed as LocalData;
    if (data.version !== 1 || !Array.isArray(data.mistakes) || !data.feedback || typeof data.feedback !== "object" || !data.tasks || typeof data.tasks !== "object") return emptyData();
    data.mistakes = data.mistakes.filter((row) => row && Number.isFinite(row.id) && typeof row.question_text === "string" && (row.answer_text === null || typeof row.answer_text === "string") && typeof row.created_at === "string" && (row.error_type === null || typeof row.error_type === "string") && Array.isArray(row.knowledge_points) && row.knowledge_points.every((point) => typeof point === "string")).slice(0, 200);
    return data;
  } catch { return emptyData(); }
}

function write(scope: SessionIdentity, data: LocalData): void {
  try { localStorage.setItem(key(scope), JSON.stringify(data)); }
  catch { throw new Error("showcase_storage_unavailable"); }
}

const normalize = (value: string) => value.normalize("NFKC").replace(/\s+/g, "").replace(/[。？?]/g, "");

export async function askShowcase(input: Input, signal: AbortSignal): Promise<AskResponse> {
  await Promise.resolve();
  signal.throwIfAborted();
  const [question, wrong = ""] = input.query.replace(/^\[(?:初中\d年级|Grade \d)\]\s*/, "").split(/\n\n(?:学生错误作答：|Student's incorrect attempt:)/);
  const example = examples.find((item) => normalize(item.question) === normalize(question));
  const previous = examples.find((item) => input.conversation_summary === `showcase:${item.id}`);
  const followup = previous && normalize(previous.followup) === normalize(question) ? previous : undefined;
  const selected = example || followup;
  let answer = selected
    ? (followup ? followup.explanation : selected.answer)
    : "当前为 GitHub Pages 示例展示，不调用在线模型。此问题不在已编写的示例范围内，不能给出可靠解析。请选择下方的示例题；任意题目求解、真实检索和 OCR 需要运行仓库中的后端服务。";
  if (example && !wrong.trim()) answer = answer.replace(/### 先找错因\n[\s\S]*?\n\n### 分步求解/, "### 分步求解");
  if (example && wrong.trim() && normalize(wrong) !== normalize(example.wrong)) {
    answer = answer.replace(/### 先找错因\n[\s\S]*?\n\n### 分步求解/, "你的作答已保存，但示例模式不自动判断任意作答的错因。以下为该题的预先编写解析。\n\n### 分步求解");
  }
  if (input.language === "en") answer = "*Curated Chinese example; no live model call.*\n\n" + answer;
  const response: AskResponse = {
    response_type: selected ? "general_answer" : "supported_refusal", answer,
    trace_id: `showcase-${crypto.randomUUID()}`, intent: "showcase",
    knowledge_points: selected ? [...selected.points] : [],
    sources: selected ? [{ source: "项目自编数学示例（非在线检索）", chapter: selected.title }] : [],
    validation_passed: false,
    conversation_history: [...input.conversation_history, { role: "student" as const, content: question }, { role: "tutor" as const, content: answer }].slice(-20),
    conversation_summary: selected ? `showcase:${selected.id}` : "",
    exercise_state: null,
    metrics: { model_attempts: 0, model_successes: 0, model_failures: 0, tool_calls: 0, execution_path: "showcase", model_provider: "local", model_name: "示例解析", llm_required: false },
    cached: false,
  };
  signal.throwIfAborted();
  if (example && wrong.trim()) {
    const data = read(input);
    if (!data.mistakes.some((row) => row.question_text === example.question)) {
      const id = Math.max(Date.now(), ...data.mistakes.map((row) => row.id + 1));
      data.mistakes.unshift({ id, question_text: example.question, answer_text: answer,
        knowledge_points: [...example.points], error_type: normalize(wrong) === normalize(example.wrong) ? example.error : "作答待核对", created_at: new Date().toISOString() });
      data.mistakes = data.mistakes.slice(0, 200);
      write(input, data);
    }
  }
  return response;
}

export function getMistakes(scope: SessionIdentity, signal?: AbortSignal): Promise<MistakeQuestion[]> {
  signal?.throwIfAborted();
  return Promise.resolve(read(scope).mistakes);
}

export function getReport(scope: SessionIdentity, signal?: AbortSignal): Promise<LearningReport> {
  signal?.throwIfAborted();
  const rows = read(scope).mistakes;
  const count = (items: string[]): [string, number][] => {
    const totals = new Map<string, number>();
    for (const item of items) totals.set(item, (totals.get(item) || 0) + 1);
    return [...totals].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  };
  const points = count(rows.flatMap((row) => row.knowledge_points));
  return Promise.resolve({ mistake_count: rows.length, analyzed_count: rows.filter((row) => row.error_type !== "作答待核对").length, analysis_limit: 200,
    top_error_types: count(rows.map((row) => row.error_type || "作答待核对")), top_knowledge_points: points,
    recommendations: rows.length ? ["本报告仅统计当前会话中实际提交的示例作答，不代表学习能力测评。", `优先复习「${points[0]?.[0] || "基础知识"}」，独立重做后用代入或定义检查。`, "示例错因来自预编写解析；其他作答需要真实后端进一步核对。"] : ["当前会话尚无错题记录。"] });
}

export async function createReport(scope: SessionIdentity): Promise<LearningTask> {
  const result = await getReport(scope);
  const data = read(scope);
  const task: LearningTask = { task_id: `local-${crypto.randomUUID()}`, status: "done", result };
  data.tasks = Object.fromEntries([...Object.entries(data.tasks).slice(-9), [task.task_id, task]]);
  write(scope, data);
  return task;
}

export function getTask(id: string, scope: SessionIdentity, signal?: AbortSignal): Promise<LearningTask> {
  signal?.throwIfAborted();
  const tasks = read(scope).tasks;
  const task = Object.hasOwn(tasks, id) ? tasks[id] : undefined;
  return task ? Promise.resolve(task) : Promise.reject(new Error("learning_failed"));
}

export function saveFeedback(id: string, correct: boolean, scope: SessionIdentity): Promise<void> {
  const data = read(scope);
  data.feedback = Object.fromEntries([...Object.entries(data.feedback).filter(([entry]) => entry !== id).slice(-199), [id, correct]]);
  write(scope, data);
  return Promise.resolve();
}
