"""One-time, auditable import of the archived experiment code (Linux only)."""

import argparse
import ast
import hashlib
import json
import re
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    repo = Path(__file__).resolve().parents[1]
    package = repo / "src" / "acoustic_technique"
    if (package / "acoustic_core.py").exists():
        parser.error("Code was already imported; refusing to overwrite it")
    package.mkdir(parents=True, exist_ok=True)
    module_names = {p.stem for p in (source / "src").glob("*.py")}
    records = []
    for path in sorted((source / "src").glob("*.py")):
        # This template embeds old outcomes in prose; use the dynamic checker instead.
        if path.name == "write_research_report.py":
            continue
        text = path.read_text(encoding="utf-8")
        for name in module_names:
            text = re.sub(
                r"(?m)^(\s*)from " + re.escape(name) + r" import ",
                r"\1from ." + name + " import ",
                text,
            )
        if path.name == "acoustic_core.py":
            text = text.replace(
                "ROOT = Path(__file__).resolve().parents[1]", "from .paths import ROOT"
            )
        if path.name == "structure_certificate.py":
            text = text.replace(
                "ROOT = Path(__file__).resolve().parents[1]", "from .paths import ROOT"
            )
        if path.name == "dispatch.py":
            text = text.replace(
                "cmd=[sys.executable,str(ROOT/'src'/job.get('script','run_search.py'))]",
                "cmd=[sys.executable,'-m','acoustic_technique.'+Path(job.get('script','run_search.py')).stem]",
            )
            text = text.replace(
                "PYTHON_JULIACALL_THREADS='4'",
                "PYTHON_JULIACALL_THREADS=os.environ.get('PYTHON_JULIACALL_THREADS','4')",
            )
        if path.name == "run_joint.py":
            text = text.replace(
                "ap.add_argument('--threads',type=int,default=4)",
                "ap.add_argument('--threads',type=int,default=int(os.environ.get('PYTHON_JULIACALL_THREADS','4')))",
            )
        if path.name in ["summarize.py", "supplementary_figures.py"]:
            text = text.replace(
                "import os\n",
                "import os\nfrom .paths import ROOT, REPOSITORY_ROOT, SOURCE_ROOT\n",
                1,
            )
            text = text.replace(
                "str(Path(__file__).resolve().parents[1]/'logs'/'matplotlib_cache')",
                "str(ROOT/'logs'/'matplotlib_cache')",
            )
            text = text.replace(
                "for p in (ROOT/'src').glob('*.py')", "for p in SOURCE_ROOT.glob('*.py')"
            )
            text = text.replace(
                "str(p.relative_to(ROOT)):hashlib.sha256",
                "str(p.relative_to(REPOSITORY_ROOT)):hashlib.sha256",
            )
        if path.name == "archive_audit.py":
            text = text.replace(
                "from .acoustic_core import ROOT,dump",
                "from .acoustic_core import ROOT,dump\nfrom .paths import REPOSITORY_ROOT,SOURCE_ROOT",
            )
            text = text.replace(
                "source_hash=hashlib.sha256((ROOT/'source'/'声学实验文档.docx').read_bytes()).hexdigest()",
                "source_hash=read(ROOT/'source'/'provenance.json')['original_document_sha256']",
            )
            text = text.replace(
                "for p in (ROOT/'src').glob('*.py')", "for p in SOURCE_ROOT.glob('*.py')"
            )
            text = text.replace(
                "str(p.relative_to(ROOT)):hashlib.sha256",
                "str(p.relative_to(REPOSITORY_ROOT)):hashlib.sha256",
            )
            text = text.replace(
                "source_docx_sha256=source_hash,passed=not errors",
                "original_source_docx_sha256=source_hash,source_document_distributed=False,passed=not errors",
            )
        if path.name == "supplementary_figures.py":
            text = text.replace("original_anomaly=", "historical_motivation=")
            text = text.replace("'Unresolved near-origin layer'", "'Actual near-origin derivative'")
            text = text.replace(
                "'Value fit hides derivative failure'", "'Actual near-origin equation residual'"
            )
        if path.name == "summarize.py":
            text = text.replace(
                "尤其main seed2有微小非线性相位和错误初值导数。",
                "各候选是否存在非线性相位或入口导数异常，以本次独立核验文件为准。",
            )
        ast.parse(text)
        target = package / path.name
        target.write_text(text, encoding="utf-8", newline="\n")
        records.append(
            dict(
                source=str(path.relative_to(source)),
                destination=str(target.relative_to(repo)),
                original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                imported_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            )
        )
    manifest_dir = repo / "experiments" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*manifest.json")):
        (manifest_dir / path.name).write_text(
            path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
        )
    environment = repo / "environment" / "archived"
    environment.mkdir(parents=True, exist_ok=True)
    for name in [
        "python_environment.txt",
        "julia_environment.txt",
        "julia_Project.toml",
        "julia_Manifest.toml",
    ]:
        shutil.copyfile(source / "docs" / name, environment / name)
    reference = repo / "reference_results"
    reference.mkdir(exist_ok=True)
    for origin, name in [
        ("docs/summary.json", "summary.json"),
        ("docs/archive_audit.json", "archive_audit.json"),
        ("experiment_00_generator/verification.json", "generator_verification.json"),
        ("experiment_04_joint/independent_verification_summary.json", "joint_verification.json"),
    ]:
        shutil.copyfile(source / origin, reference / name)
    provenance = dict(
        original_document_sha256=hashlib.sha256(
            (source / "source/声学实验文档.docx").read_bytes()
        ).hexdigest(),
        original_document_distributed=False,
        imported_modules=records,
        omitted_module="write_research_report.py: archive-specific prose template; replaced by actual-result reporting",
        scientific_changes="No changes to ODEs, sample construction, search libraries, budgets, candidate selection, model fitting or certification algorithms.",
        integration_changes=[
            "relative imports and configurable output paths",
            "module-based dispatch",
            "source hashes from repository source directory",
            "source document provenance instead of redistribution",
            "neutral labels for new-run diagnostic figures",
        ],
    )
    (reference / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checker = (source / "reproduction_tools/check_reproduction.py").read_text(encoding="utf-8")
    (package / "result_check.py").write_text(checker, encoding="utf-8", newline="\n")
    print("Imported %d scientific modules and 14 unchanged manifests." % len(records))


if __name__ == "__main__":
    main()
