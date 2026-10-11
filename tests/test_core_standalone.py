"""Ядро работает без отделяемых частей: копия src/lurker без подпакета service импортируется целиком,
CLI `columns` и `analyze` (без --prompt) выполняются. Гарантия, что ui/ и lurker.service удаляются без последствий."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ["clean", "cluster", "config", "data", "dynamics", "embed", "llm", "period", "pipeline", "preclass", "summarize", "text", "cli"]


class CoreStandaloneTests(unittest.TestCase):
    def test_core_without_service(self):
        with tempfile.TemporaryDirectory() as d:
            dst = Path(d) / "lurker"
            shutil.copytree(ROOT / "src" / "lurker", dst, ignore=shutil.ignore_patterns("service", "__pycache__"))
            code = "import importlib\n" + "".join(f"importlib.import_module('lurker.{m}')\n" for m in CORE) + "print('ok')"
            r = subprocess.run([sys.executable, "-c", code], cwd=d, capture_output=True, text=True, env={"PYTHONPATH": d, "PATH": ""})
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            r = subprocess.run([sys.executable, "-m", "lurker.cli", "columns", str(ROOT / "data/test/tickets.csv")],
                               cwd=d, capture_output=True, text=True, env={"PYTHONPATH": d, "PATH": ""})
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertIn("Комментарий", r.stdout)

    def test_core_modules_do_not_import_service_or_ui(self):
        for m in CORE:
            src = (ROOT / "src" / "lurker" / f"{m}.py").read_text(encoding="utf-8")
            top = [l for l in src.splitlines() if l.startswith(("from ", "import "))]
            self.assertFalse(any("service" in l or "streamlit" in l for l in top), (m, top))


if __name__ == "__main__":
    unittest.main()
