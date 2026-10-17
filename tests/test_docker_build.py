from pathlib import Path
import shlex
import tomllib


def test_package_metadata_is_copied_before_docker_install() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    required = {"pyproject.toml", "uv.lock", project["readme"], project["license"]["file"]}
    setup = (root / "Dockerfile").read_text().split("RUN pip install uv", 1)[0]
    copied = {
        source
        for line in setup.splitlines()
        if line.startswith("COPY ")
        for source in shlex.split(line)[1:-1]
    }

    assert required <= copied, f"Missing package metadata before install: {required - copied}"
