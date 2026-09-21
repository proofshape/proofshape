#!/usr/bin/env bash
set -euo pipefail

# Download the F-06 golden capture set from the shared Google Drive folders.
# Photos remain outside git. This script verifies, for each sample, both the
# expected JPEG count and the expected content hash recorded in
# golden_capture_manifest.json (not just the manifest file's own checksum),
# so a Drive file that's replaced or corrupted without changing the frame
# count is caught instead of silently producing a different golden input on
# two developers' machines. It writes local per-file SHA-256 hashes after
# download so repeated downloads can be diffed by hand too.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${GOLDEN_CAPTURE_OUT_DIR:-${1:-$ROOT_DIR/fixtures/data/golden_capture}}"
MANIFEST="${GOLDEN_CAPTURE_MANIFEST:-$ROOT_DIR/fixtures/golden_capture_manifest.json}"

# Portable SHA-256: prefer sha256sum (GNU coreutils, Linux/CI), fall back to
# the BSD/macOS-native `shasum -a 256`. The project's documented local dev
# path is native Apple Silicon (D-010/D-012), which does not ship coreutils
# by default, so relying on sha256sum alone breaks there.
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

sha256_stdin() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | awk '{print $1}'
  else
    shasum -a 256 | awk '{print $1}'
  fi
}

# Combined content hash for one sample directory: sha256 of the sorted
# "<per-file sha256>  <basename>" lines. Changes if any file's bytes change,
# or if a file is added/removed/renamed — independent of the JPEG count
# check, which alone would miss a same-count file swap.
sample_content_hash() {
  local dir="$1"
  find "$dir" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) -print0 \
    | sort -z \
    | while IFS= read -r -d '' f; do
        printf '%s  %s\n' "$(sha256_file "$f")" "$(basename "$f")"
      done \
    | sha256_stdin
}

expected_count() {
  python3 -c "
import json, sys
data = json.load(open(sys.argv[2]))
for c in data['captures']:
    if c['sample_id'] == sys.argv[1]:
        print(c['frames'])
        sys.exit(0)
sys.exit(1)
" "$1" "$MANIFEST"
}

expected_folder_id() {
  python3 -c "
import json, sys
data = json.load(open(sys.argv[2]))
for c in data['captures']:
    if c['sample_id'] == sys.argv[1]:
        print(c['drive_folder_id'])
        sys.exit(0)
sys.exit(1)
" "$1" "$MANIFEST"
}

expected_content_sha256() {
  python3 -c "
import json, sys
data = json.load(open(sys.argv[2]))
for c in data['captures']:
    if c['sample_id'] == sys.argv[1]:
        print(c.get('content_sha256', ''))
        sys.exit(0)
sys.exit(1)
" "$1" "$MANIFEST"
}

download_sample() {
  local sample="$1"
  local folder_id="$2"
  local expected="$3"
  local dest="$OUT_DIR/$sample"

  rm -rf "$dest"
  mkdir -p "$dest"
  gdown --folder "https://drive.google.com/drive/folders/$folder_id" -O "$dest"

  local count
  count="$(find "$dest" -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l | tr -d ' ')"
  if [[ "$count" != "$expected" ]]; then
    echo "$sample: expected $expected JPEGs, found $count"
    exit 1
  fi

  local expected_hash actual_hash
  expected_hash="$(expected_content_sha256 "$sample")"
  actual_hash="$(sample_content_hash "$dest")"
  if [[ -n "$expected_hash" && "$actual_hash" != "$expected_hash" ]]; then
    echo "$sample: content hash mismatch (expected $expected_hash, got $actual_hash)."
    echo "$sample: this means the downloaded bytes differ from what the manifest recorded,"
    echo "$sample: e.g. a Drive file was replaced or corrupted without changing the frame count."
    exit 1
  fi

  echo "$sample: $count JPEGs, content hash verified"
}

main() {
  if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown is required. Install it with: python3 -m pip install gdown"
    exit 1
  fi

  mkdir -p "$OUT_DIR"

  for sample in s-01 s-02 s-03 s-04 s-05; do
    download_sample "$sample" "$(expected_folder_id "$sample")" "$(expected_count "$sample")"
  done

  (
    cd "$OUT_DIR"
    : > SHA256SUMS.local
    find . -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) -print0 \
      | sort -z \
      | while IFS= read -r -d '' f; do
          printf '%s  %s\n' "$(sha256_file "$f")" "$f"
        done >> SHA256SUMS.local
  )

  echo "Golden capture download complete: $OUT_DIR"
  echo "Local content hashes: $OUT_DIR/SHA256SUMS.local"
}

# Only run when executed directly, not when sourced (tests source this file
# to exercise download_sample() and the hashing helpers in isolation).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
