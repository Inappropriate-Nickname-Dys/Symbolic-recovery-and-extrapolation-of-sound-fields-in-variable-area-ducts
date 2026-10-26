"""Verify a small real search/training pipeline; this is not the full study."""

import json
import numpy as np
from .paths import ROOT


def main():
    search = ROOT / "experiment_01_recovery/uniform_1_N160_uniform_seed0_smoke"
    baseline = ROOT / "experiment_01_recovery/baselines/uniform_1_N160_uniform_seed0"
    r = json.loads((search / "result.json").read_text())
    c = json.loads((search / "structure_certificate.json").read_text())
    d = np.load(search / "data.npz")
    assert d["Xtrain"].shape == (160, 1)
    assert np.max(d["xtrain"]) <= 0.4 and np.max(d["xval"]) <= 0.4
    assert c["family_structure_matched"] and c["coefficients_match_theory_within_tolerance"]
    assert r["metrics"]["ood"]["nrmse"] < 1e-6
    assert (baseline / "mlp_polished.pt").is_file()
    assert (baseline / "physical_diagnostics.json").is_file()
    out = dict(
        passed=True,
        profile="smoke",
        searches=1,
        baseline_sets=1,
        selected_expression=r["selected"]["refined"],
        ood_nrmse=r["metrics"]["ood"]["nrmse"],
        target_family_recovered=True,
        note="Checks real PySR, MLP, refinement and certificate execution. Not the 157-search study.",
    )
    (ROOT / "docs/smoke_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
