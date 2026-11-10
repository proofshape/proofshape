"""Generate C-14's canned D-062 deviation GLB fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from recon.mesh_cleanup import write_glb


def generate(path: Path) -> None:
    vertices = np.array(
        [[0, 0, 0], [30, 0, 0], [0, 30, 0], [30, 30, 0], [15, 15, 20]],
        dtype=np.float32,
    )
    faces = np.array(
        [[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4], [0, 2, 1], [1, 2, 3]],
        dtype=np.int32,
    )
    colors = np.array(
        [
            [255, 255, 255],
            [128, 128, 128],
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
        ],
        dtype=np.uint8,
    )
    write_glb(
        path,
        vertices,
        faces,
        colors=colors,
        extra_vertex_attributes={
            "_DEVIATION_MM": np.array([-2.0, 0.0, 1.0, 0.0, 3.0], dtype=np.float32),
            "_DEVIATION_MEASURED": np.array(
                [1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32
            ),
        },
    )


def main() -> None:
    path = Path("capture/stub/fixtures/model-deviation.glb")
    generate(path)
    print(path)


if __name__ == "__main__":
    main()
