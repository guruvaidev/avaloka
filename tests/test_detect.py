from app.api.cloud_connections import detect_folder_table_type


def test_iceberg():
    r = detect_folder_table_type([
        "metadata/v3.metadata.json", "metadata/v2.metadata.json",
        "data/00000-0-abc.parquet",
    ])
    assert r["table_type"] == "iceberg"
    assert r["metadata_key"] == "metadata/v3.metadata.json"


def test_hive_partitioned():
    r = detect_folder_table_type([
        "dt=2026-01-20/part-0.parquet", "dt=2026-01-21/part-0.parquet",
    ])
    assert r["table_type"] == "hive_parquet"
    assert r["hive_partitioning"] is True


def test_flat_parquet_dir():
    r = detect_folder_table_type(["part-0.parquet", "part-1.parquet"])
    assert r["table_type"] == "parquet_dir"
    assert r["glob"] == "*.parquet"


def test_iceberg_wins_over_hive():
    # Iceberg data files can also sit under col=value/ dirs — iceberg must win.
    r = detect_folder_table_type([
        "metadata/v1.metadata.json",
        "data/dt=2026-01-20/00000-0-abc.parquet",
    ])
    assert r["table_type"] == "iceberg"


def test_unknown():
    assert detect_folder_table_type(["readme.txt", "notes.csv"])["table_type"] == "unknown"


def test_empty():
    assert detect_folder_table_type([])["table_type"] == "empty"