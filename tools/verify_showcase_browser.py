#!/usr/bin/env python3
"""Validate the static showcase on desktop and mobile without a backend."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:28181/junior-math-agentic-rag/")
    parser.add_argument("--output", type=Path, default=Path(".runtime/showcase-browser"))
    parser.add_argument("--timeout-ms", type=int, default=60000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for name, width, height in (("desktop", 1440, 1000), ("mobile", 390, 844)):
                context = browser.new_context(viewport={"width": width, "height": height}, accept_downloads=True)
                page = context.new_page()
                page.set_default_navigation_timeout(args.timeout_ms)
                errors, warnings, failed, backend = [], [], [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("console", lambda message: warnings.append(message.text) if message.type in ("warning", "error") else None)
                page.on("response", lambda response: failed.append({"url": response.url, "status": response.status}) if response.status >= 400 else None)
                page.on("request", lambda request: backend.append(request.url) if any(part in request.url for part in ("/auth/", "/ask/", "/feedback", "/reports/", "/tasks", "/ready", "/runtime", "/attachments/")) else None)
                page.goto(args.url, wait_until="networkidle")
                expect(page.locator("[data-example]")).to_have_count(6)
                expect(page.locator("#serviceStatus")).to_contain_text("示例解析")
                page.screenshot(path=str(args.output / f"{name}-home.png"), full_page=True)
                page.locator("[data-example=equation]").click()
                expect(page.locator(".message.assistant .answer-meta")).to_contain_text("示例解析")
                expect(page.locator(".assistant-answer").first).to_contain_text("去括号")
                expect(page.locator(".proof-rail")).to_have_count(0)
                page.get_by_role("button", name="为什么去括号要乘每一项？", exact=True).click()
                expect(page.locator(".message.assistant")).to_have_count(2)
                page.get_by_role("button", name="Helpful").last.click()
                page.locator("#evidenceButton").click()
                expect(page.get_by_role("dialog")).to_contain_text("在线模型调用 0 次")
                expect(page.get_by_role("dialog")).not_to_contain_text("Trace")
                page.get_by_role("dialog").get_by_role("button", name="关闭", exact=True).click()
                page.locator("#learningButton").click()
                expect(page.locator(".learning-mistake")).to_have_count(1)
                page.locator("#learning-report-tab").click()
                page.get_by_role("button", name="生成报告", exact=True).click()
                expect(page.locator(".learning-task-status")).to_have_text("报告已完成")
                expect(page.locator(".learning-totals dd").first).to_have_text("1")
                with page.expect_download() as pending:
                    page.get_by_role("button", name="下载报告", exact=True).click()
                report = json.loads(Path(pending.value.path()).read_text())
                assert report["mistake_count"] == 1
                assert report["top_error_types"] == [["去括号漏乘", 1]]
                page.screenshot(path=str(args.output / f"{name}-report.png"), full_page=True)
                page.get_by_role("dialog").get_by_role("button", name="关闭", exact=True).click()
                page.reload(wait_until="networkidle")
                expect(page.locator(".message.assistant")).to_have_count(2)
                page.locator("#learningButton").click()
                expect(page.locator(".learning-mistake")).to_have_count(1)
                page.get_by_role("dialog").get_by_role("button", name="关闭", exact=True).click()
                if width < 860:
                    page.locator("#menuButton").click()
                page.locator("#newChatButton").click()
                expect(page.locator("[data-example]")).to_have_count(6)
                page.locator("#learningButton").click()
                expect(page.locator(".learning-content")).to_contain_text("当前会话还没有错题记录")
                page.get_by_role("dialog").get_by_role("button", name="关闭", exact=True).click()
                page.locator("#questionInput").fill("解方程 3(x - 2) = 2x + 6")
                page.locator("#sendButton").click()
                expect(page.locator(".assistant-answer")).to_contain_text("不在已编写的示例范围")
                page.locator("#uploadButton").click()
                expect(page.locator("#attachmentPreview")).to_contain_text("需要运行后端服务")
                page.locator("#attachmentPreview button").click()
                page.locator("#themeButton").click()
                assert page.locator("html").get_attribute("data-theme") == "dark"
                page.locator("#themeButton").click()
                page.locator("#languageButton").click()
                expect(page.locator("#serviceStatus")).to_contain_text("Curated answers")
                page.locator("#languageButton").click()
                overflow = page.evaluate("""() => ({document: document.documentElement.scrollWidth > innerWidth+1, elements: [...document.querySelectorAll('button,textarea,header')].filter(e => { const r=e.getBoundingClientRect(), s=getComputedStyle(e); return s.visibility!=='hidden' && r.width>0 && r.height>0 && r.right>2 && r.left<innerWidth-2 && (r.right>innerWidth+2 || r.left < -2); }).map(e=>({tag:e.tagName,id:e.id,cls:e.className}))})""")
                assert not overflow["document"] and not overflow["elements"], overflow
                assert page.locator(".katex-error").count() == 0
                assert not errors and not warnings and not failed and not backend, (errors, warnings, failed, backend)
                checks.append({"viewport": name, "passed": True, "console_errors": errors, "console_warnings": warnings, "failed_assets": failed, "backend_requests": backend, "overflow": overflow, "report_mistakes": report["mistake_count"]})
                context.close()
        finally:
            browser.close()
    result = {"url": args.url, "passed": all(check["passed"] for check in checks), "checks": checks}
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
