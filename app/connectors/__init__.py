"""The app side of the connector contract (design part B, 15b).

``registry`` turns a connector's ``ReadReceipt`` into the owner's catalog
version and the lineage edge. Handle opening (sealed credentials to
``Credentials``) and the egress pre-flight belong here too and arrive with
the egress policy (P2).
"""

from app.connectors.registry import (KINDS_BY_CONNECTOR, dataset_kind, locator_for, read_and_record,
                                     record_read, try_record_read)

__all__ = ["KINDS_BY_CONNECTOR", "dataset_kind", "locator_for", "read_and_record", "record_read",
           "try_record_read"]
