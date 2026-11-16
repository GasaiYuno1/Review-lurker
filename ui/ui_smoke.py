#!/usr/bin/env python3
"""Дымовой прогон Streamlit-интерфейса в headless-браузере: запускает app.py, нажимает «Анализировать»,
задаёт вопрос, проходит по вкладкам, делает скриншоты в build/ui-*.png. Нужен playwright и chromium.
python3 tools/ui_smoke.py [--port 8511] [--chromium /opt/pw-browsers/chromium]
"""
import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # корень репозитория


def wait_port(port: int, timeout: float = 60) -> None:
    t = time.time()
    while time.time() - t < timeout:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise SystemExit("streamlit не поднялся")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8511)
    ap.add_argument("--chromium", default=os.environ.get("CHROMIUM_PATH") or None)
    ap.add_argument("--out", default=str(ROOT / "build"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "streamlit.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "ui" / "app.py"), "--server.headless", "true",
                             "--server.port", str(a.port), "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"],
                            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    try:
        wait_port(a.port)
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            kw = {"executable_path": a.chromium} if a.chromium else {}
            b = p.chromium.launch(**kw)
            page = b.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(f"http://127.0.0.1:{a.port}", wait_until="networkidle")
            page.get_by_text("Review-lurker").first.wait_for(timeout=60_000)
            page.get_by_role("button", name="Загрузить и подготовить").click()
            page.get_by_text("Вопрос к данным").wait_for(timeout=300_000)
            page.wait_for_timeout(2000)
            page.screenshot(path=str(out / "ui-1-ready.png"), full_page=True)
            box = page.locator("input[aria-label^='Что посмотреть']")
            box.fill("как часто нас оскорбляли и угрожали физической расправой за последние 2 месяца")
            box.press("Enter")
            try:
                page.get_by_text("Тематики внутри выборки").wait_for(timeout=300_000)
            except Exception:
                (out / "ui-error.txt").write_text(page.inner_text("body"), encoding="utf-8")
                page.screenshot(path=str(out / "ui-error.png"), full_page=True)
                raise
            page.wait_for_timeout(2500)
            page.screenshot(path=str(out / "ui-2-question.png"), full_page=True)
            box.fill("сравни, что говорили про мобильное приложение в январе и в июле")
            box.press("Enter")
            page.get_by_text("Сравнение периодов").wait_for(timeout=300_000)
            page.wait_for_timeout(2500)
            page.screenshot(path=str(out / "ui-3-compare.png"), full_page=True)
            page.get_by_role("tab", name="Обзор всех тем").click()
            page.wait_for_timeout(2500)
            page.screenshot(path=str(out / "ui-4-overview.png"), full_page=True)
            b.close()
        print("скриншоты:", ", ".join(str(p) for p in sorted(out.glob("ui-*.png"))))
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
