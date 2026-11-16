"""Общее для исследовательских скриптов: пути, загрузка с кэшем."""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data" / "research"          # данные (не в git: скачиваются скриптами)
RAW = DATA / "raw"
BUILD = ROOT / "build" / "research"         # кэш эмбеддингов и промежуточные файлы
RESULTS = ROOT / "research" / "results"     # небольшие итоговые таблицы (в git)
for p in (DATA, RAW, BUILD, RESULTS):
    p.mkdir(parents=True, exist_ok=True)


def download(url: str, dst: Path) -> Path:
    if not dst.exists() or dst.stat().st_size == 0:
        print(f"скачиваю {url} → {dst}")
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(dst)
    return dst


def unzip_one(zpath: Path, member_suffix: str) -> Path:
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.endswith(member_suffix))
        out = zpath.parent / Path(name).name
        if not out.exists():
            z.extract(name, zpath.parent)
            extracted = zpath.parent / name
            if extracted != out:
                extracted.rename(out)
    return out
