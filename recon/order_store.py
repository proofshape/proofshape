"""R-34: file-backed buyer order store.

Each order has its own directory beneath PROOFSHAPE_ORDERS_DIR. The JSON record is the source of
truth; an uploaded CAD is stored byte-for-byte beside it.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from recon.completion_contract import write_json_atomic

CROCKFORD_BASE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
ORDER_CODE_PREFIX = "PO-"
ORDER_CODE_LENGTH = 8
ORDER_CODE_RE = re.compile(
    rf"^{re.escape(ORDER_CODE_PREFIX)}[{CROCKFORD_BASE32}]{{{ORDER_CODE_LENGTH}}}$"
)


def is_valid_order_code(order_code: str) -> bool:
    """Only server-shaped order codes may ever become filesystem path components."""
    return ORDER_CODE_RE.fullmatch(order_code) is not None


class OrderStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def order_dir(self, order_code: str) -> Path:
        if not is_valid_order_code(order_code):
            raise ValueError("invalid order code")
        return self.root / order_code

    def record_path(self, order_code: str) -> Path:
        return self.order_dir(order_code) / "order.json"

    def exists(self, order_code: str) -> bool:
        if not is_valid_order_code(order_code):
            return False
        return self.record_path(order_code).is_file()

    def load(self, order_code: str) -> dict[str, Any] | None:
        if not is_valid_order_code(order_code):
            return None
        path = self.record_path(order_code)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def cad_path(self, order_code: str) -> Path | None:
        record = self.load(order_code)
        if record is None:
            return None
        cad = record.get("cad")
        if not cad:
            return None
        return self.order_dir(order_code) / cad["stored_name"]

    def create(
        self,
        *,
        created_at: str,
        spec: dict[str, Any],
        placement_note: str | None,
        buyer_reference: str | None,
        cad_name: str | None = None,
        cad_source_path: Path | None = None,
        cad_units: str | None = None,
        cad_bbox_mm: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        if (cad_name is None) != (cad_source_path is None):
            raise ValueError("cad_name and cad_source_path must be supplied together")

        order_code, order_dir = self._reserve_order_dir()
        try:
            cad_record = None
            if cad_name is not None and cad_source_path is not None:
                suffix = Path(cad_name).suffix.lower()
                stored_name = f"cad{suffix}"
                cad_path = order_dir / stored_name
                Path(cad_source_path).replace(cad_path)
                cad_record = {
                    "original_name": Path(cad_name).name,
                    "stored_name": stored_name,
                    "units": cad_units,
                    "bounding_box_mm": cad_bbox_mm,
                }

            record = {
                "order_code": order_code,
                "created_at": created_at,
                "has_cad": cad_record is not None,
                "cad": cad_record,
                "spec": spec,
                "placement_note": placement_note,
                "buyer_reference": buyer_reference,
            }
            write_json_atomic(self.record_path(order_code), record)
            return record
        except BaseException:
            shutil.rmtree(order_dir, ignore_errors=True)
            raise

    def _reserve_order_dir(self) -> tuple[str, Path]:
        """Reserve a unique code with an atomic mkdir, safe across multiple worker processes."""
        import secrets

        for _ in range(100):
            body = "".join(
                secrets.choice(CROCKFORD_BASE32) for _ in range(ORDER_CODE_LENGTH)
            )
            code = f"{ORDER_CODE_PREFIX}{body}"
            order_dir = self.order_dir(code)
            try:
                order_dir.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                continue
            return code, order_dir
        raise RuntimeError("could not allocate a unique order code")
