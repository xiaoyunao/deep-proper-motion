import json

import duckdb
import pytest

from deep_pm.footprint_reference_packs_v24 import pack
from deep_pm.io import digest, save_json


def test_pack_checks_rows_content_and_identical_detection_bucket(tmp_path):
    db = duckdb.connect()
    shards = []
    for label, ids in (("a", ["1", "2"]), ("b", ["2"]), ("empty", [])):
        d = tmp_path / label
        d.mkdir()
        source = d / "reference.parquet"
        db.execute("CREATE OR REPLACE TABLE mock(detection_id VARCHAR)")
        for identity in ids:
            db.execute("INSERT INTO mock VALUES (?)", [identity])
        db.execute(f"COPY mock TO '{source}' (FORMAT PARQUET)")
        save_json(d / "v22_reference_input_core_summary.json",
                  dict(measurements=len(ids), parquet_sha256=digest(source)))
        shards.append((label, d))
    marker = pack(tmp_path, "packed", shards, db)
    receipt = json.loads(marker.read_text())
    assert receipt["rows"] == 3
    assert pack(tmp_path, "packed", shards, db, verify=True) == marker
    copies = db.execute(f"""SELECT detection_id, count(*), count(DISTINCT detection_bucket)
        FROM read_parquet('{marker.parent}/parts/*/*.parquet', hive_partitioning=true)
        GROUP BY detection_id ORDER BY detection_id""").fetchall()
    assert copies == [("1", 1, 1), ("2", 2, 1)]
    corrupted = marker.parent / next(iter(receipt["files"]))
    corrupted.write_bytes(b"broken")
    with pytest.raises(ValueError, match="changed"):
        pack(tmp_path, "packed", shards, db, verify=True)
    db.close()
