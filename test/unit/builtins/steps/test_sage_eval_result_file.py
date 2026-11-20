# SPDX-License-Identifier: Apache-2.0

"""The sage-eval steps' results-file search must find SALAD-Bench's own result (#482).

After the eval script, each step emits the first results file under the experiment's
output tree as its sage_eval_results artifact. SALAD-Bench writes
safety-eval/salad-bench/<model>/base_set-result.json. With only results*.json,
metrics*.json and metrics*.jsonl to match, a SALAD eval alone in its tree found
nothing and exited 1 after a complete run (builds f71274d5, 1131fd71); in a shared
tree it emitted a sibling eval's file instead. The search is also anchored at
/<experiment>/, so one rung's name cannot match a longer one (ckpt_25, ckpt_250).
These tests run each step's real search block in bash against such trees.

On AWS, sage_eval_config.result_subdir names the eval's own results directory under
the experiment. When it is set, the searches start there, so a sibling eval's
result cannot be emitted in place of this eval's own.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

from .sage_eval_steps import one_script

REPO_ROOT = Path(__file__).resolve().parents[4]
ENVS = REPO_ROOT / "configurations/assets/environments/skypilot"

# The steps under test, and the variable each roots its search at.
STEPS = {
    "aws/steps/sage-eval": "OUTPUT_BASE_PATH",
    "lsf/ibm-bluevela/steps/sage-eval": "OUTPUT_BASE_PATH",
}

EXPERIMENT = "exp-ckpt_1000"
SALAD = "safety-eval/salad-bench/export-1000/base_set-result.json"
SALAD_DIR = "safety-eval/salad-bench"
IFEVAL = "ifeval/export-1000/metrics-all.jsonl"
AWS = "aws/steps/sage-eval"


def _search_block(step: str) -> str:
    """The step's run-script lines from the search root (or, without one, the first
    results search) to the emit's `fi`."""
    run = one_script(ENVS / step / "step.yaml", "GB_ARTIFACT_ID:sage_eval_results")
    lines = run.splitlines()
    start = next(
        i
        for i, l in enumerate(lines)
        if 'if [ -n "$RESULT_SUBDIR" ]' in l or "RESULT_FILE=$(find" in l
    )
    emit = next(i for i, l in enumerate(lines) if 'if [ -n "$RESULT_FILE" ]' in l)
    indent = len(lines[emit]) - len(lines[emit].lstrip())
    end = next(
        i
        for i in range(emit + 1, len(lines))
        if lines[i].strip() == "fi" and len(lines[i]) - len(lines[i].lstrip()) == indent
    )
    return "\n".join(lines[start : end + 1])


# mtimes: the eval's log, and files written before or after it (the -newer fallback).
LOG_MTIME, OLD, NEW = 1000, 0, 2000


def _emitted(
    step: str, root: Path, tmp_path: Path, result_subdir: str = ""
) -> str | None:
    log = tmp_path / "sage-eval.log"
    log.write_text("")
    os.utime(log, (LOG_MTIME, LOG_MTIME))
    # The fallback search compares against a file touched before the eval: the
    # eval's log on LSF, a start marker on AWS. Point either at a private copy.
    block = _search_block(step)
    env = {
        **os.environ,
        STEPS[step]: str(root),
        "EXPERIMENT": EXPERIMENT,
        "RESULT_SUBDIR": result_subdir,
    }
    if "/tmp/sage-eval.log" in block:
        block = block.replace("/tmp/sage-eval.log", str(log))
    else:
        assert '-newer "$EVAL_START_MARKER"' in block
        env["EVAL_START_MARKER"] = str(log)
    # -e as in the step's run script. check=False: an empty tree is meant to exit 1;
    # the emitted line is what counts.
    out = subprocess.run(
        ["bash", "-ec", block], env=env, capture_output=True, text=True, check=False
    ).stdout
    m = re.search(r"GB_ARTIFACT_ID:sage_eval_results GB_ARTIFACT_PATH:(.+)", out)
    return m.group(1) if m else None


def _touch(
    root: Path, rel: str, experiment: str = EXPERIMENT, mtime: int = OLD
) -> Path:
    p = root / experiment / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}")
    # OLD by default: older than the eval's log, so only the primary search sees it.
    os.utime(p, (mtime, mtime))
    return p


@pytest.mark.parametrize("step", STEPS)
def test_a_salad_eval_alone_in_its_tree_emits_its_own_result(step, tmp_path):
    """The eval-rerun case: RERUN_SAGE_DIR holds only the SALAD eval's output."""
    root = tmp_path / "out"
    _touch(root, SALAD)
    _touch(root, "safety-eval/salad-bench/export-1000/base_set-gen.json")
    assert _emitted(step, root, tmp_path) == str(root / EXPERIMENT / SALAD)


@pytest.mark.parametrize("step", STEPS)
def test_a_longer_rung_name_is_not_this_rungs_result(step, tmp_path):
    """A failed exp-ckpt_1000 must not pass on exp-ckpt_10000's result."""
    root = tmp_path / "out"
    _touch(root, "safety-eval/attaq/results.json", experiment=EXPERIMENT + "0")
    (root / EXPERIMENT / "logs-dir").mkdir(parents=True)
    assert _emitted(step, root, tmp_path) is None


@pytest.mark.parametrize("step", STEPS)
def test_an_empty_tree_emits_nothing(step, tmp_path):
    root = tmp_path / "out"
    (root / EXPERIMENT / "logs-dir").mkdir(parents=True)
    assert _emitted(step, root, tmp_path) is None


@pytest.mark.parametrize("step", STEPS)
def test_the_fallback_emits_a_new_csv_under_the_experiment(step, tmp_path):
    """Only the fallback matches '*.csv', and only files newer than the eval's log."""
    root = tmp_path / "out"
    _touch(root, "code-eval/old.csv")
    new = _touch(root, "code-eval/summary.csv", mtime=NEW)
    assert _emitted(step, root, tmp_path) == str(new)


@pytest.mark.parametrize("step", STEPS)
def test_the_fallback_ignores_a_sibling_rungs_new_result(step, tmp_path):
    """A crashed rung's log is older than results a parallel rung writes afterwards."""
    root = tmp_path / "out"
    _touch(
        root, "safety-eval/attaq/results.json", experiment="exp-ckpt_2000", mtime=NEW
    )
    _touch(root, "summary.csv", experiment="exp-ckpt_2000", mtime=NEW)
    (root / EXPERIMENT / "logs-dir").mkdir(parents=True)
    assert _emitted(step, root, tmp_path) is None


@pytest.mark.parametrize("step", STEPS)
def test_an_empty_experiment_fails_before_the_eval(step):
    """An empty experiment makes the anchored search match nothing; refuse it early."""
    run = one_script(ENVS / step / "step.yaml", "GB_ARTIFACT_ID:sage_eval_results")
    check = next(l for l in run.splitlines() if '[ -n "$EXPERIMENT" ]' in l)
    eval_line = next(
        i for i, l in enumerate(run.splitlines()) if "/sage/cluster/gb/scripts/" in l
    )
    assert run.splitlines().index(check) < eval_line
    env = {**os.environ, "EXPERIMENT": ""}
    r = subprocess.run(
        ["bash", "-ec", f"{check}\necho REACHED"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 1 and "REACHED" not in r.stdout
    assert "experiment is empty" in r.stdout


@pytest.mark.parametrize("step", STEPS)
def test_traversal_order_decides_between_salad_and_a_sibling_eval(step, tmp_path):
    """Known limit: the step emits *a* result under the experiment, not its own eval's.

    With SALAD next to IFEval, whichever find reaches first is emitted. A crashed eval
    can therefore pass on a sibling's file until each eval gets its own output path.
    """
    root = tmp_path / "out"
    salad = _touch(root, SALAD)
    ifeval = _touch(root, "ifeval/export-1000/metrics-all.jsonl")
    assert _emitted(step, root, tmp_path) in {str(salad), str(ifeval)}


def test_result_subdir_emits_this_evals_result_next_to_a_sibling(tmp_path):
    """With result_subdir set, SALAD next to IFEval emits SALAD's own result."""
    root = tmp_path / "out"
    _touch(root, IFEVAL)
    salad = _touch(root, SALAD)
    assert _emitted(AWS, root, tmp_path, SALAD_DIR) == str(salad)


def test_result_subdir_does_not_pass_on_a_siblings_result(tmp_path):
    """A SALAD eval that wrote nothing fails, though IFEval's result is on disk."""
    root = tmp_path / "out"
    _touch(root, IFEVAL)
    (root / EXPERIMENT / SALAD_DIR).mkdir(parents=True)
    assert _emitted(AWS, root, tmp_path, SALAD_DIR) is None


def test_result_subdir_limits_the_fallback_to_this_eval(tmp_path):
    """The fallback emits a new CSV under this eval's directory, not a sibling's."""
    root = tmp_path / "out"
    _touch(root, "ifeval/summary.csv", mtime=NEW)
    assert _emitted(AWS, root, tmp_path, SALAD_DIR) is None
    own = _touch(root, f"{SALAD_DIR}/summary.csv", mtime=NEW)
    assert _emitted(AWS, root, tmp_path, SALAD_DIR) == str(own)


def _subdir_check() -> str:
    """The AWS step's result_subdir check: the ``case`` block that validates it."""
    run = one_script(ENVS / AWS / "step.yaml", "GB_ARTIFACT_ID:sage_eval_results")
    lines = run.splitlines()
    start = next(i for i, l in enumerate(lines) if 'case "/${RESULT_SUBDIR}/"' in l)
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "esac")
    eval_line = next(i for i, l in enumerate(lines) if "/sage/cluster/gb/scripts/" in l)
    assert end < eval_line, "result_subdir must be checked before the eval"
    return "\n".join(lines[start : end + 1])


@pytest.mark.parametrize(
    "value, accepted",
    [
        ("", True),
        (SALAD_DIR, True),
        ("ifeval/", True),
        ("a..b", True),
        ("/abs/path", False),
        ("..", False),
        ("../other-exp", False),
        ("a/../../other-exp", False),
        ("a/..", False),
    ],
)
def test_result_subdir_must_stay_under_the_experiment(value, accepted):
    """An absolute path or a '..' segment would leave <output_dir>/<experiment>."""
    env = {**os.environ, "RESULT_SUBDIR": value}
    r = subprocess.run(
        ["bash", "-ec", f"{_subdir_check()}\necho REACHED"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ("REACHED" in r.stdout) is accepted, r.stdout
    if not accepted:
        assert r.returncode == 1 and "result_subdir" in r.stdout
