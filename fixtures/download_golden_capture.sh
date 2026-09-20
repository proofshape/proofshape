#!/usr/bin/env bash
set -euo pipefail

# Download the F-06 golden capture set from the shared Google Drive folders.
# Photos remain outside git. This script verifies the expected JPEG count for
# each sample after download and writes local SHA-256 hashes for the downloaded
# files so a developer can compare repeated downloads.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/fixtures/data/golden_capture}"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown is required. Install it with: python3 -m pip install gdown"
  exit 1
fi

mkdir -p "$OUT_DIR"

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
  echo "$sample: $count JPEGs"
}

download_sample "s-01" "1NB2E7HydD8KsD3tCRKA9j14fkdjCAJMH" 26
download_sample "s-02" "1FOPhzL2Bo-_Qfqvk-lVFiXg73WXi_8sU" 30
download_sample "s-03" "10HQtYToX3ptszNI_HE24RZbtQDoJYxyj" 27
download_sample "s-04" "1a-KgtnW_FueYx0_zMZNlWfNaD-adf3xQ" 30
download_sample "s-05" "1EkStcpwZ5eDg-3KUtdaSmuw7iTcAIs5y" 29

(
  cd "$OUT_DIR"
  find . -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS.local
)

echo "Golden capture download complete: $OUT_DIR"
echo "Local content hashes: $OUT_DIR/SHA256SUMS.local"
