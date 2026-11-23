"""Data-source connectors behind one contract (``avaloka.connectors.base``).

``CONNECTORS`` is the registry the conformance suite is parametrised from
(design §F.B, T15.1): a connector that is not listed here is not a connector,
and one that is listed cannot avoid the suite.
"""

from avaloka.connectors.base import (PLANNER_METADATA_MAX_BYTES, READ_MAX_BYTES, READ_MAX_ROWS,
                                     SAMPLE_FALLBACKS, SAMPLE_METHODS, Capability, Column, Connector, ConnectorBase,
                                     ConnectorError, Credentials, Endpoint, Predicate, Query,
                                     ReadReceipt, ReadResult, SourceVersion, Table, WindowSnapshot,
                                     bounded_batches, cap_metadata, predicates_from_plan,
                                     projection_from_plan, query_digest, requires)


def _registry():
    from avaloka.connectors.files import FileConnector
    from avaloka.connectors.objectstore import ObjectStoreConnector
    from avaloka.connectors.postgres import PostgresConnector, SQLAlchemyConnector
    return (FileConnector, ObjectStoreConnector, SQLAlchemyConnector, PostgresConnector)


#: Every connector class in this tree. Append yours; the suite will find it.
CONNECTORS = _registry()

__all__ = [
    "CONNECTORS",
    "PLANNER_METADATA_MAX_BYTES",
    "READ_MAX_BYTES",
    "READ_MAX_ROWS",
    "SAMPLE_FALLBACKS",
    "SAMPLE_METHODS",
    "Capability",
    "Column",
    "Connector",
    "ConnectorBase",
    "ConnectorError",
    "Credentials",
    "Endpoint",
    "Predicate",
    "Query",
    "ReadReceipt",
    "ReadResult",
    "SourceVersion",
    "Table",
    "WindowSnapshot",
    "bounded_batches",
    "cap_metadata",
    "predicates_from_plan",
    "projection_from_plan",
    "query_digest",
    "requires",
]
