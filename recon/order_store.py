"""R-34: file-backed buyer order store.

Each order has its own directory beneath PROOFSHAPE_ORDERS_DIR. The JSON record is the source of
truth; an uploaded CAD is stored byte-for-byte beside it.
"""

from __future__ import annotations

import json
import secrets
import threading
from pathlib import Path
from typing import Any

CROCKFORD_BASE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
ORDER_CODE_PREFIX = "PO-"
ORDER_CODE_LENGTH = 8


class OrderStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def order_dir(self, order_code: str) -> Path:
        return self.root / order_code

    def record_path(self, order_code: str) -> Path:
        return self.order_dir(order_code) / "order.json"

    def exists(self, order_code: str) -> bool:
        return self.record_path(order_code).exists()

    def load(self, order_code: str) -> dict[str, Any] | None:
        path = self.record_path(order_code)
        if not path.exists():
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
        cad_bytes: bytes | None = None,
        cad_units: str | None = None,
        cad_bbox_mm: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        if (cad_name is None) != (cad_bytes is None):
            raise ValueError("cad_name and cad_bytes must be supplied together")

        with self._lock:
            order_code = self._new_code()
            order_dir = self.order_dir(order_code)
            order_dir.mkdir(parents=False, exist_ok=False)

            cad_record = None
            if cad_name is not None and cad_bytes is not None:
                suffix = Path(cad_name).suffix.lower()
                stored_name = f"cad{suffix}"
                cad_path = order_dir / stored_name
                temp_cad = cad_path.with_suffix(cad_path.suffix + ".tmp")
                temp_cad.write_bytes(cad_bytes)
                temp_cad.replace(cad_path)
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
            self._write_record(record)
            return record

    def _new_code(self) -> str:
        for _ in range(100):
            body = "".join(secrets.choice(CROCKFORD_BASE32) for _ in range(ORDER_CODE_LENGTH))
            code = f"{ORDER_CODE_PREFIX}{body}"
            if not self.order_dir(code).exists():
                return code
        raise RuntimeError("could not allocate a unique order code")

    def _write_record(self, record: dict[str, Any]) -> None:
        path = self.record_path(record["order_code"])
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temp_path.replace(path)
