from __future__ import annotations

import subprocess
from pathlib import Path


def run(script_name: str) -> None:
    module_name = script_name.replace("/", ".").replace("\\", ".")
    if module_name.endswith(".py"):
        module_name = module_name[:-3]
    command = [str(Path(".venv") / "Scripts" / "python.exe"), "-m", module_name]
    subprocess.run(command, check=True)


def main() -> None:
    run("scripts/run_text_memory_demo.py")
    run("scripts/train_numeric_demo.py")
    run("scripts/run_v2_demo.py")
    run("scripts/generate_paper_v2.py")


if __name__ == "__main__":
    main()
