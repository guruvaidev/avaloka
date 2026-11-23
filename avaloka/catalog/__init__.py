"""The dataset catalog, as a package the CLI can ship (design part C).

``store`` is the catalog itself: tables, registration, versions, reuse,
delete. It imports nothing from ``app``. ``local`` is the CLI's catalog: a
SQLite file under ``~/.avaloka`` and the owner's identity on this machine.
``cli`` is ``avaloka catalog list|show|use|forget``. The application wraps the
store in ``app/catalog/store.py`` to add the lineage graph.
"""
