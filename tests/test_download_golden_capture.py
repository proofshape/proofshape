"""Tests for fixtures/download_golden_capture.sh.

The script is bash, not Python, so these tests exercise it as a subprocess:
a fake `gdown` executable is put first on PATH so no network access or real
Drive folder is needed, and a throwaway manifest (pointed at via the
GOLDEN_CAPTURE_MANIFEST env var the script reads) drives expected counts and
content hashes. The script guards its five-sample "main" run behind
`[[ "${BASH_SOURCE[0]}" == "${0}" ]]`, so sourcing it from a small bash
wrapper makes `download_sample` and the hashing helpers callable in
isolation without running the real download list.

Covers the two failure paths raised in review on PR #22:
- a frame-count mismatch is caught and fails loudly
- a content-hash mismatch (same count, different bytes — e.g. a Drive file
  replaced or corrupted) is caught and fails loudly, not just the count
- the missing-gdown early exit
- the happy path, including that SHA256SUMS.local is written with the
  portable hasher (sha256sum or shasum -a 256, whichever is on PATH)
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "fixtures" / "download_golden_capture.sh"


def _write_fake_gdown(bin_dir: Path, files_by_sample: dict[str, list[str]]) -> None:
    """A fake `gdown` that writes fixed content into -O <dest> based on which
    Drive folder id ("https://drive.google.com/drive/folders/<id>") it was
    asked for, so each test sample gets deterministic, controllable bytes."""
    fake = bin_dir / "gdown"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "folder=''",
        "dest=''",
        "",
        "while [[ $# -gt 0 ]]; do",
        '  case "$1" in',
        "    --folder) shift ;;",
        '    -O) dest="$2"; shift 2 ;;',
        '    http*) folder="$1"; shift ;;',
        "    *) shift ;;",
        "  esac",
        "done",
        "",
        'case "$folder" in',
    ]
    for folder_id, contents in files_by_sample.items():
        lines.append(f'  *"/{folder_id}") ')
        for name, content in contents:
            lines.append(f'    printf %s "{content}" > "$dest/{name}"')
        lines.append("    ;;")
    lines += ["esac", "exit 0"]
    fake.write_text("\n".join(lines) + "\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)


def _content_sha256(files: list[tuple[str, str]]) -> str:
    """Mirror the script's sample_content_hash(): sha256 of the "<sha256 of file>  <basename>"
    lines, in filename order (the script's `find | sort` sorts paths, then hashes each file --
    it never sorts by hash), hashed again as one blob.

    F-08 (#196): sorting the combined "hash  name" strings instead, as this helper used to,
    sorts primarily by hash -- a different order from the script's whenever a hash's lexicographic
    order doesn't happen to match its file's name order. The two only agreed before by chance of
    the specific fixture names/content used here.
    """
    entries = [
        f"{hashlib.sha256(content.encode()).hexdigest()}  {name}"
        for name, content in sorted(files, key=lambda pair: pair[0])
    ]
    return hashlib.sha256(("\n".join(entries) + "\n").encode()).hexdigest()


def _manifest(
    tmp_path: Path, sample_id: str, folder_id: str, frames: int, content_sha256: str
) -> Path:
    manifest = {
        "captures": [
            {
                "sample_id": sample_id,
                "drive_folder_id": folder_id,
                "frames": frames,
                "content_sha256": content_sha256,
            }
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _multi_manifest(tmp_path: Path, captures: list[dict]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"captures": captures}))
    return path


def _run_download_sample(
    tmp_path: Path,
    bin_dir: Path,
    manifest: Path,
    sample: str,
    folder_id: str,
    expected: int,
) -> subprocess.CompletedProcess[str]:
    dest_root = tmp_path / "out"
    dest_root.mkdir(exist_ok=True)
    wrapper = tmp_path / "run.sh"
    wrapper.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
source "{SCRIPT}"
download_sample "{sample}" "{folder_id}" "{expected}"
"""
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["GOLDEN_CAPTURE_MANIFEST"] = str(manifest)
    env["GOLDEN_CAPTURE_OUT_DIR"] = str(dest_root)
    return subprocess.run(
        ["bash", str(wrapper)],
        cwd=dest_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_sample_ids(tmp_path: Path, manifest: Path) -> subprocess.CompletedProcess[str]:
    wrapper = tmp_path / "run_sample_ids.sh"
    wrapper.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
source "{SCRIPT}"
sample_ids
"""
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ)
    env["GOLDEN_CAPTURE_MANIFEST"] = str(manifest)
    env["GOLDEN_CAPTURE_OUT_DIR"] = str(tmp_path / "out")
    return subprocess.run(
        ["bash", str(wrapper)], env=env, capture_output=True, text=True, check=False
    )


@pytest.fixture()
def bin_dir(tmp_path: Path) -> Path:
    d = tmp_path / "bin"
    d.mkdir()
    return d


def test_committed_manifest_matches_its_committed_sha256() -> None:
    # F-08: fixtures/golden_capture_manifest.sha256 is the committed checksum of the versioned
    # inventory manifest (fixtures/README.md) -- it only catches an accidental edit to the
    # manifest if something actually checks the two still agree.
    manifest_path = REPO_ROOT / "fixtures" / "golden_capture_manifest.json"
    sha_path = REPO_ROOT / "fixtures" / "golden_capture_manifest.sha256"
    expected_line = sha_path.read_text(encoding="utf-8").strip()
    expected_hash, _, expected_name = expected_line.partition("  ")
    assert expected_name == "golden_capture_manifest.json"
    actual_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert actual_hash == expected_hash


def test_missing_gdown_fails_with_install_hint(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PATH"] = "/usr/bin:/bin"  # deliberately exclude any gdown on PATH
    result = subprocess.run(
        ["bash", str(SCRIPT), str(tmp_path / "out")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "gdown is required" in result.stdout


def test_download_sample_succeeds_on_matching_count_and_hash(
    tmp_path: Path, bin_dir: Path
) -> None:
    files = [("a.jpg", "AAA"), ("b.jpg", "BBB")]
    _write_fake_gdown(bin_dir, {"folder123": files})
    manifest = _manifest(tmp_path, "s-test", "folder123", 2, _content_sha256(files))

    result = _run_download_sample(tmp_path, bin_dir, manifest, "s-test", "folder123", 2)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "content hash verified" in result.stdout


def test_download_sample_fails_on_frame_count_mismatch(
    tmp_path: Path, bin_dir: Path
) -> None:
    files = [("a.jpg", "AAA")]  # only 1 file, but expected below says 2
    _write_fake_gdown(bin_dir, {"folder123": files})
    manifest = _manifest(tmp_path, "s-test", "folder123", 2, _content_sha256(files))

    result = _run_download_sample(tmp_path, bin_dir, manifest, "s-test", "folder123", 2)

    assert result.returncode == 1
    assert "expected 2 JPEGs, found 1" in result.stdout


def test_download_sample_fails_on_content_hash_mismatch(
    tmp_path: Path, bin_dir: Path
) -> None:
    """Same file count, different bytes — the P1 review gap: the old script
    only checked the count, so a swapped/corrupted Drive file with the same
    number of files would have passed silently."""
    original = [("a.jpg", "AAA"), ("b.jpg", "BBB")]
    swapped = [("a.jpg", "AAA"), ("b.jpg", "CORRUPTED")]
    _write_fake_gdown(bin_dir, {"folder123": swapped})
    manifest = _manifest(tmp_path, "s-test", "folder123", 2, _content_sha256(original))

    result = _run_download_sample(tmp_path, bin_dir, manifest, "s-test", "folder123", 2)

    assert result.returncode == 1
    assert "content hash mismatch" in result.stdout


def test_sample_ids_reads_every_sample_from_the_manifest(tmp_path: Path) -> None:
    # #196: main() looped over a hardcoded s-01..s-05 list instead of the manifest.
    manifest = _multi_manifest(
        tmp_path,
        [
            {"sample_id": "s-01", "drive_folder_id": "f1", "frames": 1},
            {"sample_id": "s-02", "drive_folder_id": "f2", "frames": 1},
            {"sample_id": "s-09", "drive_folder_id": "f9", "frames": 1},
        ],
    )

    result = _run_sample_ids(tmp_path, manifest)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == ["s-01", "s-02", "s-09"]


def test_main_downloads_exactly_the_manifests_samples_not_a_hardcoded_list(
    tmp_path: Path, bin_dir: Path
) -> None:
    # #196: main()'s own loop (not just sample_ids() in isolation) must actually drive off the
    # manifest -- reverting its `for sample in $(sample_ids)` back to a literal `s-01 s-02 s-03
    # s-04 s-05` list passes every other test in this file, since none of them exercise main()
    # end to end. A manifest using sample ids outside that hardcoded set is the only way to
    # prove it: this would fail looking up a hardcoded id's manifest entry if the old behaviour
    # ever came back.
    files_a = [("x.jpg", "AAA")]
    files_b = [("y.jpg", "BBB")]
    _write_fake_gdown(bin_dir, {"folderA": files_a, "folderB": files_b})
    manifest = _multi_manifest(
        tmp_path,
        [
            {
                "sample_id": "zz-alpha",
                "drive_folder_id": "folderA",
                "frames": 1,
                "content_sha256": _content_sha256(files_a),
            },
            {
                "sample_id": "zz-beta",
                "drive_folder_id": "folderB",
                "frames": 1,
                "content_sha256": _content_sha256(files_b),
            },
        ],
    )
    out_dir = tmp_path / "out"
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["GOLDEN_CAPTURE_MANIFEST"] = str(manifest)

    result = subprocess.run(
        ["bash", str(SCRIPT), str(out_dir)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "zz-alpha: 1 JPEGs, content hash verified" in result.stdout
    assert "zz-beta: 1 JPEGs, content hash verified" in result.stdout
    assert (out_dir / "zz-alpha" / "x.jpg").is_file()
    assert (out_dir / "zz-beta" / "y.jpg").is_file()


def test_download_sample_reports_when_the_manifest_has_no_hash_recorded(
    tmp_path: Path, bin_dir: Path
) -> None:
    # #196: a sample with no content_sha256 recorded printed "content hash verified" even
    # though the comparison was silently skipped -- the hash check never ran.
    files = [("a.jpg", "AAA"), ("b.jpg", "BBB")]
    _write_fake_gdown(bin_dir, {"folder123": files})
    manifest = _manifest(tmp_path, "s-test", "folder123", 2, "")

    result = _run_download_sample(tmp_path, bin_dir, manifest, "s-test", "folder123", 2)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "content hash verified" not in result.stdout
    assert "no content hash recorded" in result.stdout
    assert (tmp_path / "out" / "s-test" / "a.jpg").is_file()  # still swapped in


def test_download_sample_leaves_the_old_copy_untouched_on_count_mismatch(
    tmp_path: Path, bin_dir: Path
) -> None:
    # #196: download_sample deleted $dest before downloading, so a failed/partial download
    # destroyed the last verified copy instead of leaving it in place.
    dest = tmp_path / "out" / "s-test"
    dest.mkdir(parents=True)
    (dest / "old.jpg").write_text("OLD")

    files = [("a.jpg", "AAA")]  # only 1 file, but expected below says 2
    _write_fake_gdown(bin_dir, {"folder123": files})
    manifest = _manifest(tmp_path, "s-test", "folder123", 2, _content_sha256(files))

    result = _run_download_sample(tmp_path, bin_dir, manifest, "s-test", "folder123", 2)

    assert result.returncode == 1
    assert (dest / "old.jpg").read_text() == "OLD"  # untouched
    assert not (dest / "a.jpg").exists()  # the bad download never got swapped in


def test_download_sample_leaves_the_old_copy_untouched_when_gdown_fails(
    tmp_path: Path, bin_dir: Path
) -> None:
    dest = tmp_path / "out" / "s-test"
    dest.mkdir(parents=True)
    (dest / "old.jpg").write_text("OLD")

    fake = bin_dir / "gdown"
    fake.write_text("#!/usr/bin/env bash\necho 'drive error' >&2\nexit 1\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    manifest = _manifest(tmp_path, "s-test", "folder123", 1, "")

    result = _run_download_sample(tmp_path, bin_dir, manifest, "s-test", "folder123", 1)

    assert result.returncode != 0
    assert (dest / "old.jpg").read_text() == "OLD"  # untouched
