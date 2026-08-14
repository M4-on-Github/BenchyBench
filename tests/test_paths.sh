#!/bin/bash
# test_paths.sh — local test suite for benchybench_paths.sh
#
#     bash test_paths.sh [path/to/benchybench_paths.sh]
#
# Builds synthetic directory trees in a temp dir and asserts the resolver picks
# the right root in each. No cluster, no GPU, no SSH.
#
# Scope: proves the resolution LOGIC is correct for a given tree, including the
# SLURM spool case where the script's own location is useless. Cannot prove
# apptainer bind behaviour on pleiades.

# Library under test: an explicit argument, else a copy sitting beside this
# script, else the deployed copy in DeGF (all deployed copies are asserted
# identical at the end of this file, so any of them serves as the reference).
_here="$(cd "$(dirname "$0")" && pwd)"
if [[ -n "${1:-}" ]]; then
    LIB="$1"
elif [[ -f "$_here/benchybench_paths.sh" ]]; then
    LIB="$_here/benchybench_paths.sh"
else
    LIB="$_here/../DeGF/CASTOR/benchybench_paths.sh"
fi

if [[ ! -f "$LIB" ]]; then
    echo "FATAL: library not found: $LIB" >&2
    echo "       Pass one explicitly:  bash $0 path/to/benchybench_paths.sh" >&2
    exit 1
fi
LIB="$(cd "$(dirname "$LIB")" && pwd)/$(basename "$LIB")"
echo "library under test: $LIB"

PASS=0
FAIL=0
TMPROOT="$(mktemp -d)"
trap 'rm -rf "$TMPROOT"' EXIT

ok()  { PASS=$((PASS+1)); printf "  \033[32mPASS\033[0m  %s\n" "$1"; }
bad() { FAIL=$((FAIL+1)); printf "  \033[31mFAIL\033[0m  %s\n" "$1"
        printf "        expected: %s\n        actual:   %s\n" "$2" "$3"; }

assert_eq() {
    [[ "$2" == "$3" ]] && ok "$1" || bad "$1" "$2" "$3"
}
assert_fails() {
    [[ "$2" -ne 0 ]] && ok "$1" || bad "$1" "non-zero exit" "exit 0"
}

# Build <base>/BenchyBench/<repo>/CASTOR/benchybench_paths.sh
# images_at: nested | standalone | both | none
make_tree() {
    local name="$1" images_at="$2" repo_name="${3:-DeGF}"
    local bench="$TMPROOT/$name/BenchyBench"
    local repo="$bench/$repo_name"

    mkdir -p "$repo/CASTOR"
    cp "$LIB" "$repo/CASTOR/benchybench_paths.sh"

    case "$images_at" in
        nested|both) mkdir -p "$bench/$IMG"; : > "$bench/$IMG/00001.jpg" ;;
    esac
    case "$images_at" in
        standalone|both) mkdir -p "$repo/$IMG"; : > "$repo/$IMG/00001.jpg" ;;
    esac
    echo "$repo"
}

IMG="shipwreck_wiki_images/sorted_images/aground"

# Run bb_resolve_root in a clean subshell with a controlled environment.
#   $1 lib to source   $2 cwd   rest: VAR=VAL assignments
resolve_in() {
    local lib="$1" cwd="$2"; shift 2
    ( cd "$cwd" 2>/dev/null || cd /
      unset BENCHYBENCH_ROOT SLURM_SUBMIT_DIR
      while [[ $# -gt 0 ]]; do export "$1"; shift; done
      source "$lib"
      bb_resolve_root 2>/dev/null )
}

echo
echo "benchybench_paths.sh — resolution tests"
echo "======================================="
echo

# ── Layout detection ─────────────────────────────────────────────────────────
repo="$(make_tree nested nested)"
assert_eq "nested layout resolves to BenchyBench root" \
    "$(dirname "$repo")" "$(resolve_in "$repo/CASTOR/benchybench_paths.sh" /)"

repo="$(make_tree standalone standalone)"
assert_eq "standalone layout resolves to repo root" \
    "$repo" "$(resolve_in "$repo/CASTOR/benchybench_paths.sh" /)"

repo="$(make_tree both both)"
assert_eq "both present prefers nested over stale in-repo copy" \
    "$(dirname "$repo")" "$(resolve_in "$repo/CASTOR/benchybench_paths.sh" /)"

# ── Failure is loud, never a guess ───────────────────────────────────────────
repo="$(make_tree missing none)"
out="$(resolve_in "$repo/CASTOR/benchybench_paths.sh" /)"; rc=$?
assert_fails "missing images exits non-zero" "$rc"
assert_eq    "missing images prints nothing to stdout" "" "$out"

# ── Explicit override ────────────────────────────────────────────────────────
repo="$(make_tree override nested)"
alt="$TMPROOT/alt"; mkdir -p "$alt/$IMG"
assert_eq "BENCHYBENCH_ROOT overrides auto-detection" \
    "$alt" "$(resolve_in "$repo/CASTOR/benchybench_paths.sh" / "BENCHYBENCH_ROOT=$alt")"

repo="$(make_tree badenv nested)"
out="$(resolve_in "$repo/CASTOR/benchybench_paths.sh" / "BENCHYBENCH_ROOT=$TMPROOT/nope")"; rc=$?
assert_fails "invalid BENCHYBENCH_ROOT exits non-zero" "$rc"
assert_eq    "invalid BENCHYBENCH_ROOT does NOT silently fall back" "" "$out"

# ── REGRESSION: cwd must never influence resolution ──────────────────────────
# The config.json '../shipwreck_wiki_images' bug resolved against cwd.
repo="$(make_tree cwdtest nested)"
lib="$repo/CASTOR/benchybench_paths.sh"; expect="$(dirname "$repo")"
for dir in / "$TMPROOT" "$repo" "$repo/CASTOR"; do
    assert_eq "cwd=$(basename "$dir") does not affect result" \
        "$expect" "$(resolve_in "$lib" "$dir")"
done

# ── SLURM_SUBMIT_DIR is a VALIDATED hint ─────────────────────────────────────
repo="$(make_tree slurmvalid nested)"
lib="$repo/CASTOR/benchybench_paths.sh"; bench="$(dirname "$repo")"

assert_eq "SLURM_SUBMIT_DIR at the root is used" \
    "$bench" "$(resolve_in "$lib" / "SLURM_SUBMIT_DIR=$bench")"

assert_eq "SLURM_SUBMIT_DIR inside a repo resolves up to the root" \
    "$bench" "$(resolve_in "$lib" / "SLURM_SUBMIT_DIR=$repo")"

# The original bug: a bogus submit dir was trusted blindly.
assert_eq "invalid SLURM_SUBMIT_DIR is ignored, not trusted" \
    "$bench" "$(resolve_in "$lib" / "SLURM_SUBMIT_DIR=$TMPROOT/wrong/place")"

assert_eq "BENCHYBENCH_ROOT takes precedence over SLURM_SUBMIT_DIR" \
    "$alt" "$(resolve_in "$lib" / "BENCHYBENCH_ROOT=$alt" "SLURM_SUBMIT_DIR=$bench")"

# ── CRITICAL: the SLURM spool case ───────────────────────────────────────────
# SLURM copies the batch script to /var/spool/... before executing, so the
# script's own location tells us nothing. Resolution must still succeed via
# SLURM_SUBMIT_DIR. This is the case that would fail on the cluster while
# passing every location-based test.
repo="$(make_tree spool nested)"
bench="$(dirname "$repo")"
spool="$TMPROOT/var_spool_slurmd/job00042"
mkdir -p "$spool"
cp "$LIB" "$spool/benchybench_paths.sh"     # detached from any repo structure

assert_eq "resolves from a spool copy via SLURM_SUBMIT_DIR" \
    "$bench" "$(resolve_in "$spool/benchybench_paths.sh" / "SLURM_SUBMIT_DIR=$repo")"

out="$(resolve_in "$spool/benchybench_paths.sh" /)"; rc=$?
assert_fails "spool copy with no hints fails loudly" "$rc"

# ── Derived paths ────────────────────────────────────────────────────────────
repo="$(make_tree imgdir nested)"
lib="$repo/CASTOR/benchybench_paths.sh"
got="$( cd / ; unset BENCHYBENCH_ROOT SLURM_SUBMIT_DIR; source "$lib"; bb_images_dir )"
assert_eq "bb_images_dir returns the image directory" \
    "$(dirname "$repo")/shipwreck_wiki_images/sorted_images" "$got"

repo="$(make_tree gtmissing nested)"
lib="$repo/CASTOR/benchybench_paths.sh"
out="$( cd / ; unset BENCHYBENCH_ROOT SLURM_SUBMIT_DIR; source "$lib"; bb_gt_csv 2>/dev/null )"; rc=$?
assert_fails "missing GT csv exits non-zero" "$rc"
assert_eq    "missing GT csv prints nothing to stdout" "" "$out"

repo="$(make_tree gtpresent nested)"
lib="$repo/CASTOR/benchybench_paths.sh"; bench="$(dirname "$repo")"
mkdir -p "$bench/Eval_CASTOR/human_ground_truth_label"
: > "$bench/Eval_CASTOR/human_ground_truth_label/human_gt.csv"
got="$( cd / ; unset BENCHYBENCH_ROOT SLURM_SUBMIT_DIR; source "$lib"; bb_gt_csv )"
assert_eq "bb_gt_csv returns the ground-truth path" \
    "$bench/Eval_CASTOR/human_ground_truth_label/human_gt.csv" "$got"

# ── Repo-name independence (same library ships in three repos) ───────────────
for name in ONLY QWEN-Maritime; do
    repo="$(make_tree "named_$name" nested "$name")"
    assert_eq "resolves identically from repo named $name" \
        "$(dirname "$repo")" "$(resolve_in "$repo/CASTOR/benchybench_paths.sh" /)"
done

# ── END-TO-END: a real batch script run the way SLURM runs it ───────────────
# Writes a script carrying the bootstrap prologue, copies it to a spool dir
# (as slurmd does), and executes it there with $0 pointing at the spool copy.
# This is the closest local approximation of the cluster invocation.
make_batch_script() {
    cat > "$1" <<'BATCH'
#!/bin/bash
set -e
_bb_lib=""
for _c in "$(dirname "$0")/benchybench_paths.sh" \
          "${SLURM_SUBMIT_DIR:-}/CASTOR/benchybench_paths.sh" \
          "${SLURM_SUBMIT_DIR:-}/benchybench_paths.sh" \
          "${BENCHYBENCH_ROOT:-}/DeGF/CASTOR/benchybench_paths.sh"; do
    [[ -f "$_c" ]] && { _bb_lib="$_c"; break; }
done
[[ -n "$_bb_lib" ]] || { echo "ERROR: benchybench_paths.sh not found" >&2; exit 1; }
source "$_bb_lib"
BENCHYBENCH_ROOT="$(bb_resolve_root)" || exit 1
echo "$BENCHYBENCH_ROOT"
BATCH
    chmod +x "$1"
}

repo="$(make_tree e2e nested)"
bench="$(dirname "$repo")"
make_batch_script "$repo/CASTOR/submit_job.sh"

# (a) Submitted from the repo — the documented workflow.
spool="$TMPROOT/spool_a/job00001"; mkdir -p "$spool"
cp "$repo/CASTOR/submit_job.sh" "$spool/slurm_script"
got="$( cd / ; unset BENCHYBENCH_ROOT; SLURM_SUBMIT_DIR="$repo" \
        bash "$spool/slurm_script" 2>/dev/null )"
assert_eq "E2E: spooled batch script, submitted from repo" "$bench" "$got"

# (b) Submitted from the BenchyBench root — where the old code broke.
spool="$TMPROOT/spool_b/job00002"; mkdir -p "$spool"
cp "$repo/CASTOR/submit_job.sh" "$spool/slurm_script"
got="$( cd / ; unset BENCHYBENCH_ROOT; SLURM_SUBMIT_DIR="$bench" \
        BENCHYBENCH_ROOT="$bench" bash "$spool/slurm_script" 2>/dev/null )"
assert_eq "E2E: spooled batch script, submitted from BenchyBench root" "$bench" "$got"

# (c) No hints at all — must fail loudly rather than run against a guess.
spool="$TMPROOT/spool_c/job00003"; mkdir -p "$spool"
cp "$repo/CASTOR/submit_job.sh" "$spool/slurm_script"
( cd / ; unset BENCHYBENCH_ROOT SLURM_SUBMIT_DIR; bash "$spool/slurm_script" ) >/dev/null 2>&1
assert_fails "E2E: spooled batch script with no hints fails loudly" "$?"

# ── Deployed copies must not drift ──────────────────────────────────────────
# The library is duplicated into each repo so a standalone clone still works.
# Duplication is only safe if divergence is caught, so enforce it here rather
# than relying on remembering to update all three.
BB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
deployed=()
for r in DeGF ONLY QWEN-Maritime; do
    [[ -f "$BB_ROOT/$r/CASTOR/benchybench_paths.sh" ]] && \
        deployed+=("$BB_ROOT/$r/CASTOR/benchybench_paths.sh")
done

if [[ ${#deployed[@]} -eq 0 ]]; then
    printf "  \033[33mSKIP\033[0m  no deployed copies found (run from BenchyBench root)\n"
else
    ref_sum="$(md5sum < "${deployed[0]}" | cut -d' ' -f1)"
    all_match=true
    for f in "${deployed[@]}"; do
        [[ "$(md5sum < "$f" | cut -d' ' -f1)" == "$ref_sum" ]] || all_match=false
    done
    $all_match \
        && ok "all ${#deployed[@]} deployed copies are byte-identical" \
        || bad "deployed copies have drifted" "identical checksums" "mismatch"
fi

echo
echo "======================================="
printf "  passed: %d   failed: %d\n" "$PASS" "$FAIL"
echo
[[ "$FAIL" -eq 0 ]]
