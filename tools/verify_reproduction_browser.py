#!/usr/bin/env python3
"""Exercise the shipped UI against the real gateway on desktop and mobile."""

import json
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.reproduce_experiments import BASE, PREFIX, REPORTS, write


def main():
    checks = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for name, width, height in (("desktop", 1440,900), ("mobile",390,844)):
                context = browser.new_context(viewport={"width": width,"height":height}, device_scale_factor=1)
                page = context.new_page()
                errors = []
                requests = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("response", lambda response: requests.append({"url": response.url, "status": response.status}))
                page.goto(BASE+"/", wait_until="networkidle")
                page.locator("#questionInput").fill("解方程 2(x-3)=8。")
                page.locator("#sendButton").click()
                page.locator('button[aria-label="Helpful"]').first.wait_for(timeout=60000)
                page.locator('button[aria-label="Helpful"]').first.click()
                page.wait_for_timeout(500)
                overflow = page.evaluate("""() => ({document: document.documentElement.scrollWidth > innerWidth+1,
                    elements: [...document.querySelectorAll('button,textarea,header')].filter(e => {
                      const r=e.getBoundingClientRect(), s=getComputedStyle(e);
                      return s.visibility!=='hidden' && r.width>0 && r.height>0 && r.right>2 && r.left<innerWidth-2 && (r.right>innerWidth+2 || r.left < -2);
                    }).map(e=>({tag:e.tagName,id:e.id,cls:e.className}))})""")
                screenshot = REPORTS / f"{PREFIX}-browser-{name}.png"
                page.screenshot(path=str(screenshot), full_page=True)
                api = [item for item in requests if any(path in item["url"] for path in ("/ask/stream", "/auth/session", "/feedback"))]
                checks.append({"viewport": name,"width":width,"height":height,"page_errors":errors,"overflow":overflow,
                               "api":api,"screenshot":str(screenshot.relative_to(ROOT)),
                               "passed":not errors and not overflow["document"] and not overflow["elements"] and any("/feedback" in item["url"] and item["status"]==200 for item in api)})
                context.close()
        finally:
            browser.close()
    write("browser", {"summary":{"passed":all(item["passed"] for item in checks)},"viewports":checks})


if __name__ == "__main__":
    main()
