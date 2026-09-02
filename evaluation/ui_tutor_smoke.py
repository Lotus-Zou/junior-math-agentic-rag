"""Real browser regression for the student correction workspace."""

import base64
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "evaluation" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
URL = "http://127.0.0.1:8000/"


def assert_no_horizontal_overflow(page):
    overflow = page.evaluate(
        """() => ({
            body: document.body.scrollWidth - window.innerWidth,
            document: document.documentElement.scrollWidth - window.innerWidth,
            offenders: [...document.querySelectorAll('button, textarea, .message-content')]
              .filter((node) => node.scrollWidth > node.clientWidth + 2)
              .map((node) => node.id || node.className || node.tagName)
              .slice(0, 10)
        })"""
    )
    assert overflow["body"] <= 1, overflow
    assert overflow["document"] <= 1, overflow
    assert not overflow["offenders"], overflow


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    errors = []

    desktop = browser.new_page(viewport={"width": 1440, "height": 960})
    desktop.on("console", lambda message: errors.append(f"console:{message.type}:{message.text}") if message.type == "error" else None)
    desktop.on("pageerror", lambda error: errors.append(f"page:{error}"))
    desktop.add_init_script(
        """
        class MockSpeechRecognition {
          start() {
            setTimeout(() => {
              this.onresult?.({ results: [[{ transcript: "解方程 x 加 1 等于 2" }]] });
              this.onend?.();
            }, 20);
          }
          stop() { this.onend?.(); }
        }
        window.SpeechRecognition = MockSpeechRecognition;
        """
    )
    desktop.route(
        "**/attachments/parse",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "ready",
                    "filename": "wrong-problem.png",
                    "media_type": "image/png",
                    "problem_text": "解方程 2x + 3 = 11",
                    "student_answer": "2x = 11 + 3",
                    "formulas": ["2x + 3 = 11"],
                    "confidence": 0.96,
                    "warnings": [],
                    "page_count": 1,
                    "parser": "vision_agent",
                    "trace_id": "browser-attachment",
                },
                ensure_ascii=False,
            ),
        ),
    )
    desktop.goto(URL, wait_until="networkidle")
    desktop.locator("#questionInput").wait_for(state="visible")
    assert desktop.locator("#welcome").is_visible()
    assert "今天想解决哪道题" in desktop.locator("#welcome").inner_text()
    assert desktop.locator(".prompt-grid").count() == 0
    assert desktop.locator(".mode-switch").count() == 0
    assert "API 已连接" in desktop.locator("#serviceStatus").inner_text()
    desktop.locator("#languageButton").click()
    assert desktop.locator("html").get_attribute("lang") == "en"
    assert "What should we work through" in desktop.locator("#welcome").inner_text()
    desktop.locator("#languageButton").click()
    desktop.locator("#themeButton").click()
    assert desktop.locator("html").get_attribute("data-theme") == "dark"
    desktop.locator("#themeButton").click()

    tiny_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    desktop.locator("#attachmentInput").set_input_files(
        {"name": "wrong-problem.png", "mimeType": "image/png", "buffer": tiny_png}
    )
    desktop.locator("#attachmentPreview").get_by_text("题目已识别，请核对").wait_for()
    assert desktop.locator("#questionInput").input_value() == "解方程 2x + 3 = 11"
    assert desktop.locator("#wrongAnswer").input_value() == "2x = 11 + 3"
    assert desktop.locator("#mistakeField").is_visible()

    desktop.locator("#questionInput").fill("")
    desktop.locator("#wrongAnswer").fill("")
    desktop.locator("#voiceButton").click()
    desktop.wait_for_function(
        "() => document.querySelector('#questionInput')?.value.includes('x 加 1')"
    )
    assert_no_horizontal_overflow(desktop)
    desktop.screenshot(path=str(REPORTS / "ui-workbench-desktop.png"), full_page=True)

    desktop.locator("#questionInput").fill("几何")
    desktop.locator("#askForm").evaluate("(form) => form.requestSubmit()")
    desktop.wait_for_function(
        "() => document.querySelector('#questionInput')?.value === ''",
        timeout=1500,
    )
    desktop.locator(".message.assistant").first.wait_for(state="visible", timeout=185000)
    desktop.locator("#exerciseContext").wait_for(state="visible", timeout=185000)
    first_difficulty = desktop.locator("#difficultyLabel").inner_text()
    assert desktop.locator(".meta-chip.agent").count() >= 1
    assert "模型调用" in desktop.locator(".meta-chip.agent").last.inner_text()
    assert "API" in desktop.locator(".meta-chip.agent").last.inner_text()
    desktop.locator("#evidenceButton").click()
    desktop.locator("[role=dialog]").wait_for(state="visible")
    assert "教材依据" in desktop.locator("[role=dialog]").inner_text()
    desktop.locator("[role=dialog]").get_by_role("button", name="关闭").click()
    assert_no_horizontal_overflow(desktop)
    desktop.screenshot(path=str(REPORTS / "ui-exercise-desktop.png"), full_page=True)

    before = desktop.locator(".message.assistant").count()
    desktop.locator("#harderButton").click()
    desktop.locator(".message.assistant").nth(before).wait_for(state="visible", timeout=185000)
    desktop.wait_for_function(
        "(before) => document.querySelector('#difficultyLabel')?.textContent !== before",
        arg=first_difficulty,
        timeout=185000,
    )
    assert desktop.locator(".meta-chip.agent").count() >= 2
    assert_no_horizontal_overflow(desktop)
    desktop.screenshot(path=str(REPORTS / "ui-harder-desktop.png"), full_page=True)

    mobile = browser.new_page(viewport={"width": 390, "height": 844})
    mobile.on("console", lambda message: errors.append(f"mobile-console:{message.type}:{message.text}") if message.type == "error" else None)
    mobile.on("pageerror", lambda error: errors.append(f"mobile-page:{error}"))
    mobile.goto(URL, wait_until="networkidle")
    mobile.locator("#questionInput").wait_for(state="visible")
    assert mobile.locator("#menuButton").is_visible()
    assert mobile.locator("#sendButton").is_visible()
    assert_no_horizontal_overflow(mobile)
    mobile.screenshot(path=str(REPORTS / "ui-workbench-mobile.png"), full_page=True)

    long_messages = []
    for index in range(36):
        role = "user" if index % 2 == 0 else "assistant"
        long_messages.append(
            {
                "id": f"long-{index}",
                "role": role,
                "content": (
                    "一次函数 y = -2x + 3 的图像怎么画？"
                    if index == 0
                    else "斜率是 -2，纵截距是 3，可以先标出点 (0, 3) 再画直线。"
                    if index == 1
                    else f"第 {index // 2 + 1} 轮：请继续分析这道九年级竞赛题。"
                    if role == "user"
                    else "先整理已知条件，再建立方程并逐步核对推导。"
                ),
                "createdAt": index,
            }
        )
    persisted = {
        "state": {
            "sessions": [
                {
                    "id": "long-chat",
                    "title": "长对话布局回归",
                    "messages": long_messages,
                    "history": [],
                    "summary": "",
                    "exercise": None,
                    "updatedAt": 1,
                }
            ],
            "activeId": "long-chat",
            "language": "zh",
            "theme": "light",
        },
        "version": 0,
    }
    long_chat = browser.new_page(viewport={"width": 390, "height": 844})
    long_chat.add_init_script(
        f"""localStorage.setItem(
          "mathtrace-agent-workspace-v2",
          {json.dumps(json.dumps(persisted, ensure_ascii=False), ensure_ascii=False)}
        );"""
    )
    long_chat.goto(URL, wait_until="networkidle")
    long_chat.locator("#questionInput").wait_for(state="visible")
    assert long_chat.locator(".message").count() == len(long_messages)
    assert long_chat.locator(".math-visual").count() >= 1
    layout = long_chat.evaluate(
        """() => {
          const composer = document.querySelector('.composer-zone').getBoundingClientRect();
          const scroller = document.querySelector('.conversation-scroll');
          return {
            composerTop: composer.top,
            composerBottom: composer.bottom,
            viewportHeight: window.innerHeight,
            bodyOverflow: document.body.scrollHeight - window.innerHeight,
            scrollable: scroller.scrollHeight > scroller.clientHeight,
          };
        }"""
    )
    assert layout["composerTop"] >= 0, layout
    assert layout["composerBottom"] <= layout["viewportHeight"] + 1, layout
    assert layout["bodyOverflow"] <= 1, layout
    assert layout["scrollable"] is True, layout
    assert_no_horizontal_overflow(long_chat)
    long_chat.screenshot(path=str(REPORTS / "ui-long-chat-mobile.png"))

    browser.close()

    assert not errors, errors
    print(
        {
            "passed": True,
            "desktop_screenshots": 3,
            "mobile_screenshots": 2,
            "difficulty_before": first_difficulty,
            "difficulty_after": desktop.locator("#difficultyLabel").inner_text() if False else "changed",
            "console_errors": errors,
        }
    )
