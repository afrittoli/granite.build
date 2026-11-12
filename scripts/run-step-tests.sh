#!/usr/bin/env bash
# Run the step-framework tests that need no cluster, no image and no credentials.
#
# Each step keeps its tests in steps/<path>/<env>/test/ and runs them with
# `make test`, which also renders the space and builds the step image. This script
# runs only the tests, the way `make test` invokes pytest: one pytest process per
# step, from the repo root, with the step's src/ on PYTHONPATH (the tests import the
# step's own modules from there). The marker expression leaves out the per-cluster
# build tests: those under test/<cluster>/ are marked `extended`, and the BlueVela
# ones `ibm`.
#
# One process per step, because several steps reuse the same test file names
# (test_step_template.py), and one pytest session over all of steps/ stops on the
# module name clash.
#
# The vendored granite-evals tests (steps/eval/*/tests, steps/eval/src/core/tests)
# are not run: they import the granite_evals package and each family's extras,
# which the repo virtualenv does not install. Their directories are named tests/,
# so the test/ search below does not find them.
#
# Usage: scripts/run-step-tests.sh [STEP_TEST_DIR ...] [-- PYTEST_ARGS ...]
#   With no directory, runs every steps/**/test/ directory.
#   Run it inside the repo virtualenv (`make step-tests` activates it).
#
# Environment:
#   STEP_TEST_MARKERS  pytest marker expression (default: "not ibm and not extended")
#   GB_ENVIRONMENT     default STANDALONE, as in `make test`
#   GBTEST_MODE        default mock: no secrets are loaded and Hugging Face is mocked

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

markers="${STEP_TEST_MARKERS:-not ibm and not extended}"
export GB_ENVIRONMENT="${GB_ENVIRONMENT:-STANDALONE}"
export GBTEST_MODE="${GBTEST_MODE:-mock}"

dirs=()
pytest_args=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then
        shift
        pytest_args=("$@")
        break
    fi
    dirs+=("${1%/}")
    shift
done

if [ ${#dirs[@]} -eq 0 ]; then
    # Top-level test/ directories only: per-cluster subdirectories are collected by
    # pytest from their parent.
    while IFS= read -r d; do
        dirs+=("$d")
    done < <(find steps -type d -name test -not -path '*/test/*' -not -path '*/node_modules/*' | sort)
fi

# Run each step's tests on several workers when pytest-xdist is installed (it is
# in the repo virtualenv's dev extra).
xdist=()
if python -c 'import xdist' 2>/dev/null; then
    xdist=(-n auto)
fi

failed=()
# Expanded with the ${a[@]+...} form throughout: bash 3.2 (macOS /bin/bash) treats
# "${a[@]}" of an empty array as unset under `set -u`.
for d in "${dirs[@]}"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: $d is not a directory" >&2
        exit 2
    fi
    step_dir="$(dirname "$d")"
    echo "::group::$step_dir"
    rc=0
    PYTHONPATH="$step_dir/src${PYTHONPATH:+:$PYTHONPATH}" \
        python -m pytest "$d" -m "$markers" -p no:cacheprovider \
        -o log_cli=false -q -rfEs \
        ${xdist[@]+"${xdist[@]}"} ${pytest_args[@]+"${pytest_args[@]}"} || rc=$?
    echo "::endgroup::"
    case "$rc" in
        0) ;;
        # 5: nothing collected, e.g. a step whose tests are all cluster build tests.
        5) echo "$step_dir: no tests selected" ;;
        *) failed+=("$step_dir (pytest exit $rc)") ;;
    esac
done

echo
if [ ${#failed[@]} -gt 0 ]; then
    echo "Step tests failed in ${#failed[@]} of ${#dirs[@]} steps:" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
echo "Step tests passed in ${#dirs[@]} steps."
