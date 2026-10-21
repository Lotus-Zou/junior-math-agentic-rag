#!/usr/bin/env python3
"""Verify the learning workspace against the real gateway, MySQL and worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def verify(page, base: str, evidence: Path, viewport: str) -> dict:
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(base, wait_until="networkidle")
    header_overflow = page.evaluate("""() => {
      const bar = document.querySelector('.workspace-bar');
      const controls = [...bar.querySelectorAll('button')].filter(e => {
        const r = e.getBoundingClientRect();
        return r.width && r.height && (r.left < -1 || r.right > innerWidth + 1);
      }).map(e => e.id || e.textContent);
      return { document: document.documentElement.scrollWidth > innerWidth + 1, controls };
    }""")
    assert not header_overflow["document"] and not header_overflow["controls"], header_overflow
    # A genuine wrong-answer request creates its own test data through the API.
    page.locator("#questionInput").fill("解方程 2(x-3)=8。学生错误作答：x=1。请分析错因并订正。")
    page.locator("#sendButton").click()
    page.get_by_role("button", name="Helpful", exact=True).first.wait_for(timeout=90000)
    page.locator("#learningButton").click()
    panel = page.get_by_role("dialog")
    expect(panel.locator(".learning-mistake").first).to_be_visible(timeout=20000)
    panel.locator("summary").first.click()
    expect(panel.locator("details[open]")).to_be_visible()
    mistakes_screenshot = evidence / f"learning-{viewport}-mistakes.png"
    page.screenshot(path=str(mistakes_screenshot), full_page=True)
    panel.get_by_role("tab", name="学习报告", exact=True).click()
    panel.get_by_role("button", name="生成报告", exact=True).click()
    expect(panel.get_by_role("status").filter(has_text="报告已完成")).to_be_visible(timeout=30000)
    expect(panel.locator(".learning-totals dd").first).to_have_text("1")
    with page.expect_download() as download:
        panel.get_by_role("button", name="下载报告", exact=True).click()
    download_path = evidence / f"learning-{viewport}-download.json"
    download.value.save_as(download_path)
    exported = json.loads(download_path.read_text())
    assert exported["mistake_count"] == exported["analyzed_count"] == 1
    expect(panel.get_by_role("alert")).to_have_count(0)
    overflow = page.evaluate("""() => {
      const panel = document.querySelector('.learning-drawer');
      return { document: document.documentElement.scrollWidth > innerWidth + 1,
        panel: panel.scrollWidth > panel.clientWidth + 1,
        controls: [...panel.querySelectorAll('button,[role=tab],h2')].filter(e => {
          const r = e.getBoundingClientRect();
          return r.width && r.height && (r.left < -1 || r.right > innerWidth + 1);
        }).map(e => e.outerHTML) };
    }""")
    assert not overflow["document"] and not overflow["panel"] and not overflow["controls"], overflow
    report_screenshot = evidence / f"learning-{viewport}-report.png"
    page.screenshot(path=str(report_screenshot), full_page=True)
    panel.get_by_role("button", name="关闭", exact=True).click()
    page.locator("#learningButton").click()
    expect(panel.get_by_role("status").filter(has_text="报告已完成")).to_be_visible()
    panel.get_by_role("button", name="关闭", exact=True).click()
    # A different conversation belonging to the same user must be empty.
    if viewport.startswith("mobile"):
        page.locator("#menuButton").click()
    page.locator("#newChatButton").click()
    page.locator("#learningButton").click()
    expect(panel.get_by_text("当前会话还没有错题记录", exact=True)).to_be_visible(timeout=20000)
    panel.get_by_role("tab", name="学习报告", exact=True).click()
    expect(panel.locator(".learning-totals dd").first).to_have_text("0")
    assert not errors, errors
    return {"viewport": viewport, "passed": True, "page_errors": errors, "overflow": overflow,
            "header_overflow": header_overflow,
            "checks": ["real wrong-answer archival", "explanation", "Redis worker report completion",
                       "download", "close/reopen task retention", "conversation isolation"],
            "screenshots": [str(mistakes_screenshot.relative_to(ROOT)), str(report_screenshot.relative_to(ROOT))]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:28080/")
    parser.add_argument("--output", default="evaluation/reports/completion-20261020-browser.json")
    args = parser.parse_args()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for name, width, height in (("desktop", 1440, 900), ("mobile", 390, 844), ("mobile-small", 320, 740)):
                context = browser.new_context(viewport={"width": width, "height": height}, accept_downloads=True)
                try:
                    results.append(verify(context.new_page(), args.base, output.parent, name))
                finally:
                    context.close()
        finally:
            browser.close()
    output.write_text(json.dumps({"passed": all(item["passed"] for item in results), "viewports": results}, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
