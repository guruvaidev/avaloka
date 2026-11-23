"""
Multi-Database Customer Onboarding API for Avaloka MCP Integration

Handles customer database registration and API key management for secure
multi-tenant access to PostgreSQL, MySQL/MariaDB, SQLite, SQL Server, Oracle
(via SQLAlchemy) and MongoDB (via pymongo).
"""

import os
import hashlib
import hmac
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio
from urllib.parse import quote_plus

import sqlalchemy as sa
from sqlalchemy import create_engine, text
from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, Field
import httpx
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from urllib.parse import urlparse, unquote

from app.mcp_server.credentials import (
    NEEDS_REREGISTERING, WRONG_KEY_MESSAGE, CredentialKeyMissing, api_key_matches, credential_handle,
    load_registry, load_unreadable, mark_unreadable_inactive, migrate_registry, save_registry,
    service_headers, verify_user_assertion,
)

# ---- Optional MongoDB driver (guarded) ----
try:
    from pymongo import MongoClient
    _PYMONGO_AVAILABLE = True
except Exception:  # pragma: no cover
    MongoClient = None
    _PYMONGO_AVAILABLE = False

logger = logging.getLogger(__name__)

SUPPORTED_DB_TYPES = ['postgresql', 'mysql', 'sqlite', 'mssql', 'oracle', 'mariadb', 'mongodb']
MONGO_SERVER_SELECTION_TIMEOUT_MS = int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "5000"))


class DatabaseConnectionRequest(BaseModel):
    customer_id: str
    customer_name: str
    database_type: str
    host: Optional[str] = None
    port: Optional[int] = None
    database_name: str
    username: Optional[str] = None
    password: Optional[str] = None
    connection_params: Optional[Dict[str, Any]] = {}
    # Full URI alternative (MongoDB / Atlas mongodb+srv://, or any SQLAlchemy DSN).
    connection_string: Optional[str] = None
    contact_email: str
    environment: str = Field(default="production", alias="type")
    query_timeout: int = 30
    max_rows: int = 1000

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"

    @validator('customer_id')
    def validate_customer_id(cls, v):
        if not v or len(v) < 3:
            raise ValueError('customer_id must be at least 3 characters')
        if not v.replace('_', '').isalnum():
            raise ValueError('customer_id can only contain letters, numbers, and underscores')
        return v.lower()

    @validator('database_type')
    def validate_database_type(cls, v):
        if v.lower() not in SUPPORTED_DB_TYPES:
            raise ValueError(f'database_type must be one of: {SUPPORTED_DB_TYPES}')
        return v.lower()

    @validator('username', 'password', 'host', 'database_name', 'customer_name')
    def _strip_whitespace(cls, v):
        # Strip stray surrounding whitespace (e.g. a copy-pasted leading tab).
        return v.strip() if isinstance(v, str) else v


class CustomerResponse(BaseModel):
    customer_id: str
    api_key: str
    mcp_endpoint: str
    status: str
    created_at: str
    database_type: str
    instructions: Dict[str, Any]


class CustomerStatus(BaseModel):
    customer_id: str
    database_type: str
    status: str
    last_health_check: Optional[str]
    connection_healthy: bool
    error_message: Optional[str]


class DatabaseConnectionManager:
    """Manages database connections and testing for multiple database types"""

    @staticmethod
    def build_connection_string(conn: DatabaseConnectionRequest) -> str:
        """Build connection string for different database types"""

        if conn.database_type == 'sqlite':
            return f"sqlite:///{conn.database_name}"

        elif conn.database_type == 'postgresql':
            password = quote_plus(conn.password) if conn.password else ""
            port = conn.port or 5432
            return f"postgresql://{conn.username}:{password}@{conn.host}:{port}/{conn.database_name}"

        elif conn.database_type in ['mysql', 'mariadb']:
            password = quote_plus(conn.password) if conn.password else ""
            port = conn.port or 3306
            return f"mysql+pymysql://{conn.username}:{password}@{conn.host}:{port}/{conn.database_name}"

        elif conn.database_type == 'mssql':
            password = quote_plus(conn.password) if conn.password else ""
            port = conn.port or 1433
            return f"mssql+pyodbc://{conn.username}:{password}@{conn.host}:{port}/{conn.database_name}?driver=ODBC+Driver+17+for+SQL+Server"

        elif conn.database_type == 'oracle':
            password = quote_plus(conn.password) if conn.password else ""
            port = conn.port or 1521
            return f"oracle+cx_oracle://{conn.username}:{password}@{conn.host}:{port}/{conn.database_name}"

        elif conn.database_type == 'mongodb':
            # A full URI (mongodb:// or mongodb+srv://) wins when provided.
            if conn.connection_string:
                return conn.connection_string
            user = quote_plus(conn.username) if conn.username else ""
            password = quote_plus(conn.password) if conn.password else ""
            auth = f"{user}:{password}@" if user else ""
            port = conn.port or 27017
            params = conn.connection_params or {}
            auth_source = params.get("authSource") or params.get("auth_source")
            q = f"?authSource={auth_source}" if auth_source else ""
            return f"mongodb://{auth}{conn.host}:{port}/{conn.database_name}{q}"

        else:
            raise ValueError(f"Unsupported database type: {conn.database_type}")

    @staticmethod
    def test_database_connection(conn: DatabaseConnectionRequest) -> Dict[str, Any]:
        """Test customer database connection and return metadata"""

        # ---- MongoDB ----
        if conn.database_type == 'mongodb':
            if not _PYMONGO_AVAILABLE:
                return {
                    "success": False,
                    "error": "pymongo is not installed on the onboarding service (pip install pymongo dnspython).",
                    "error_type": "RuntimeError",
                }
            connection_string = DatabaseConnectionManager.build_connection_string(conn)
            client = None
            try:
                client = MongoClient(
                    connection_string,
                    serverSelectionTimeoutMS=max(conn.query_timeout * 1000, MONGO_SERVER_SELECTION_TIMEOUT_MS),
                )
                client.admin.command("ping")
                version = ""
                try:
                    version = client.server_info().get("version", "")
                except Exception:
                    pass
                parsed = urlparse(connection_string)
                db_name = (parsed.path or "").lstrip("/").split("?")[0] or conn.database_name
                db = client[db_name]
                try:
                    table_count = len(db.list_collection_names())
                except Exception:
                    table_count = 0
                return {
                    "success": True,
                    "database": db_name,
                    "user": conn.username or "mongodb",
                    "version": version or "Unknown",
                    "table_count": table_count,
                    "database_type": "mongodb",
                }
            except Exception as e:
                return {"success": False, "error": str(e), "error_type": type(e).__name__}
            finally:
                if client is not None:
                    client.close()

        # ---- SQL ----
        try:
            connection_string = DatabaseConnectionManager.build_connection_string(conn)

            engine_kwargs = {'pool_pre_ping': True}

            if conn.database_type == 'sqlite':
                engine_kwargs['connect_args'] = {
                    'timeout': conn.query_timeout,
                    'check_same_thread': False
                }
            else:
                engine_kwargs['connect_args'] = {
                    'connect_timeout': conn.query_timeout,
                    **conn.connection_params
                }

            engine = create_engine(connection_string, **engine_kwargs)

            with engine.connect() as connection:
                if conn.database_type == 'postgresql':
                    result = connection.execute(text("SELECT version(), current_database(), current_user"))
                    version, database, user = result.fetchone()

                elif conn.database_type in ['mysql', 'mariadb']:
                    result = connection.execute(text("SELECT VERSION(), DATABASE(), USER()"))
                    version, database, user = result.fetchone()

                elif conn.database_type == 'sqlite':
                    result = connection.execute(text("SELECT sqlite_version()"))
                    version = result.fetchone()[0]
                    database = conn.database_name
                    user = "sqlite"

                elif conn.database_type == 'mssql':
                    result = connection.execute(text("SELECT @@VERSION, DB_NAME(), SUSER_SNAME()"))
                    version, database, user = result.fetchone()

                elif conn.database_type == 'oracle':
                    result = connection.execute(text("SELECT * FROM v$version WHERE banner LIKE 'Oracle%' AND ROWNUM = 1"))
                    version = result.fetchone()[0] if result.fetchone() else "Oracle"
                    result = connection.execute(text("SELECT SYS_CONTEXT('USERENV', 'DB_NAME'), USER FROM dual"))
                    database, user = result.fetchone()

                try:
                    if conn.database_type == 'postgresql':
                        result = connection.execute(text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"))
                    elif conn.database_type in ['mysql', 'mariadb']:
                        result = connection.execute(text(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{conn.database_name}'"))
                    elif conn.database_type == 'sqlite':
                        result = connection.execute(text("SELECT COUNT(*) FROM sqlite_master WHERE type='table'"))
                    elif conn.database_type == 'mssql':
                        result = connection.execute(text("SELECT COUNT(*) FROM information_schema.tables WHERE table_type = 'BASE TABLE'"))
                    elif conn.database_type == 'oracle':
                        result = connection.execute(text("SELECT COUNT(*) FROM user_tables"))

                    table_count = result.fetchone()[0]
                except Exception:
                    table_count = 0

            engine.dispose()

            return {
                "success": True,
                "database": database,
                "user": user,
                "version": version.split()[0] if version else "Unknown",
                "table_count": table_count,
                "database_type": conn.database_type
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }


class CustomerManager:
    def __init__(self, mcp_server_url: str):
        self.mcp_server_url = mcp_server_url
        self.customers_file = os.getenv("CUSTOMERS_FILE", "customers.json")
        self.load_customers()
        self.db_manager = DatabaseConnectionManager()

    def _read_customers_file(self) -> dict:
        """Read the customers registry FRESH from disk (see original notes).

        Secrets are sealed on disk and opened here, in process; what this
        returns is for this process only and never for a response body.
        """
        try:
            return load_registry(self.customers_file)
        except Exception as e:
            print(f"Error reading customers file: {e}")
            return getattr(self, "customers", {})

    def unreadable_customers(self) -> dict:
        """Records on disk this deployment's key cannot open, shown without any
        secret and with status NEEDS_REREGISTERING (credentials.load_unreadable)."""
        try:
            return load_unreadable(self.customers_file)
        except Exception as e:
            print(f"Error reading customers file: {type(e).__name__}")
            return {}

    def load_customers(self):
        try:
            # A registry written before secrets were sealed is re-sealed on
            # the first load by a process that holds the key.
            migrate_registry(self.customers_file)
        except Exception as e:
            print(f"Error migrating customers file: {e}")
        self.customers = self._read_customers_file()

    def save_customers(self):
        try:
            save_registry(self.customers_file, self.customers)
        except CredentialKeyMissing as e:
            # Never fall back to a plaintext write: the file stays as it was.
            print(f"Refusing to save customers in plaintext: {e}")
            raise
        except Exception as e:
            print(f"Error saving customers: {e}")

    def generate_api_key(self, customer_id: str) -> str:
        timestamp = str(int(datetime.now().timestamp()))
        raw_key = f"{customer_id}:{timestamp}:{os.urandom(16).hex()}"
        return f"ak_{hashlib.sha256(raw_key.encode()).hexdigest()[:32]}"

    async def register_customer(self, registration: DatabaseConnectionRequest, owner_user_id: str) -> CustomerResponse:
        """Register a new customer with their database, owned by ``owner_user_id``."""

        # Fresh: an id another onboarding process registered must 409 here,
        # not be overwritten -- that would hand its owner's database to the caller.
        # The one exception: the same owner registering again over a record
        # they deactivated (the API rolls a registration back this way when
        # saving the connection fails), so a retry is not "already registered".
        self.customers = self._read_customers_file()
        # A record this key cannot open still belongs to its owner: it is not a
        # free id (credentials.load_unreadable).
        existing = self.customers.get(registration.customer_id) or \
            self.unreadable_customers().get(registration.customer_id)
        if (existing is not None and existing.get("status") == "inactive"
                and not (existing.get("owner_user_id") or "").strip()):
            # Deleted while bound to nobody (#564): retired for good. Saved
            # connections and DTA aliases can still name it, so whoever
            # registered it again would receive the next user's transfer.
            raise HTTPException(
                status_code=409,
                detail=f"Customer {registration.customer_id} was deleted; choose a new id",
            )
        if existing is not None and not (
            existing.get("status") == "inactive"
            and existing.get("owner_user_id")
            and existing.get("owner_user_id") == owner_user_id
        ):
            if existing.get("status") == NEEDS_REREGISTERING and existing.get("owner_user_id") == owner_user_id:
                # Its owner, re-sending it: say why it is refused, and what works.
                raise HTTPException(
                    status_code=409,
                    detail=f"Connection {registration.customer_id}: {WRONG_KEY_MESSAGE}.",
                )
            raise HTTPException(
                status_code=409,
                detail=f"Customer {registration.customer_id} already registered"
            )

        print(f"Testing {registration.database_type} connection for customer: {registration.customer_id}")
        connection_test = self.db_manager.test_database_connection(registration)

        if not connection_test["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Database connection failed: {connection_test['error']}"
            )

        api_key = self.generate_api_key(registration.customer_id)
        connection_string = self.db_manager.build_connection_string(registration)

        customer_data = {
            "customer_id": registration.customer_id,
            "owner_user_id": owner_user_id,
            "customer_name": registration.customer_name,
            "database_type": registration.database_type,
            "connection_string": connection_string,
            "contact_email": registration.contact_email,
            "environment": registration.environment,
            "api_key": api_key,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "query_timeout": registration.query_timeout,
            "max_rows": registration.max_rows,
            "last_health_check": datetime.now().isoformat(),
            "database_info": connection_test
        }

        try:
            async with httpx.AsyncClient() as client:
                # The secrets travel sealed under DB_ENCRYPTION_KEY (the body is
                # also signed, below); the MCP server opens them with its copy of
                # the key, so the request carries no password (1.6 port, 88120343).
                mcp_payload = {
                    "customer_id": registration.customer_id,
                    "connection_string_enc": credential_handle(connection_string),
                    "database_type": registration.database_type,
                    "api_key_enc": credential_handle(api_key),
                    "created_at": datetime.now().isoformat(),
                    "status": "active",
                    "max_rows": registration.max_rows,
                    "query_timeout": registration.query_timeout,
                    "owner_user_id": owner_user_id,
                }

                # Signed over exactly these bytes: the MCP server's admin
                # routes take only the onboarding service's credential.
                body = json.dumps(mcp_payload).encode("utf-8")
                mcp_response = await client.post(
                    f"{self.mcp_server_url}/admin/customers",
                    content=body,
                    headers={"Content-Type": "application/json",
                             **service_headers("POST", "/admin/customers", body)},
                    timeout=30.0
                )

                if mcp_response.status_code != 200:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to register with MCP server: {mcp_response.text}"
                    )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"MCP server registration failed: {str(e)}"
            )

        self.customers[registration.customer_id] = customer_data
        try:
            self.save_customers()
        except CredentialKeyMissing as e:
            del self.customers[registration.customer_id]
            raise HTTPException(status_code=500, detail=str(e))

        print(f"Successfully registered {registration.database_type} customer: {registration.customer_id}")

        return CustomerResponse(
            customer_id=registration.customer_id,
            api_key=api_key,
            mcp_endpoint=f"{self.mcp_server_url}/sse",
            status="active",
            created_at=customer_data["created_at"],
            database_type=registration.database_type,
            instructions={
                "usage": "Use the API key in Authorization header: Bearer <api_key>",
                "endpoint": f"{self.mcp_server_url}/sse",
                "database_type": registration.database_type,
                "tools_available": ["query", "list_tables", "describe_table"],
                "example_queries": self._get_example_queries(registration.database_type),
                "supported_features": self._get_database_features(registration.database_type),
                "support_contact": "support@avaloka.com"
            }
        )

    async def evict_from_mcp(self, customer_id: str) -> bool:
        """Tell the MCP server to drop a deactivated customer's api_key and engine now.

        Signed like /admin/customers. ``False`` if the call could not be made
        or was refused; the record is inactive either way, and the MCP server
        checks the registry on every /call_tool, so the key is still refused.
        """
        path = f"/admin/customers/{customer_id}"
        try:
            headers = service_headers("DELETE", path, b"")
            async with httpx.AsyncClient() as client:
                resp = await client.delete(f"{self.mcp_server_url}{path}", headers=headers, timeout=10.0)
        except Exception as e:
            logger.error(f"Evicting {customer_id} from the MCP server failed: {type(e).__name__}: {e}")
            return False
        if resp.status_code != 200:
            logger.error(f"Evicting {customer_id} from the MCP server answered HTTP {resp.status_code}")
            return False
        return True

    def _get_example_queries(self, database_type: str) -> Dict[str, str]:
        examples = {
            "postgresql": {
                "basic_select": "SELECT * FROM your_table_name LIMIT 10",
                "with_filter": "SELECT column1, column2 FROM your_table WHERE column1 > 100",
                "aggregation": "SELECT COUNT(*), AVG(numeric_column) FROM your_table GROUP BY category_column"
            },
            "mysql": {
                "basic_select": "SELECT * FROM your_table_name LIMIT 10",
                "with_filter": "SELECT column1, column2 FROM your_table WHERE column1 > 100",
                "show_tables": "SHOW TABLES"
            },
            "mariadb": {
                "basic_select": "SELECT * FROM your_table_name LIMIT 10",
                "with_filter": "SELECT column1, column2 FROM your_table WHERE column1 > 100",
                "show_tables": "SHOW TABLES"
            },
            "sqlite": {
                "basic_select": "SELECT * FROM your_table_name LIMIT 10",
                "list_tables": "SELECT name FROM sqlite_master WHERE type='table'",
                "table_info": "PRAGMA table_info(your_table_name)"
            },
            "mssql": {
                "basic_select": "SELECT TOP 10 * FROM your_table_name",
                "with_filter": "SELECT column1, column2 FROM your_table WHERE column1 > 100",
                "list_tables": "SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE'"
            },
            "oracle": {
                "basic_select": "SELECT * FROM your_table_name WHERE ROWNUM <= 10",
                "list_tables": "SELECT table_name FROM user_tables",
                "describe_table": "SELECT column_name, data_type FROM user_tab_columns WHERE table_name = 'YOUR_TABLE'"
            },
            "mongodb": {
                "list_collections": "SHOW COLLECTIONS",
                "sample_collection": "SELECT * FROM your_collection LIMIT 10",
                "json_query": '{"collection": "your_collection", "filter": {}, "limit": 10}',
                "aggregation": '{"collection": "your_collection", "pipeline": [{"$group": {"_id": "$category", "n": {"$sum": 1}}}]}'
            }
        }
        return examples.get(database_type, examples["postgresql"])

    def _get_database_features(self, database_type: str) -> List[str]:
        features = {
            "postgresql": ["Advanced Analytics", "JSON Support", "Full-text Search", "Window Functions"],
            "mysql": ["High Performance", "Replication", "Full-text Indexing"],
            "mariadb": ["High Performance", "Replication", "JSON Support", "Columnstore"],
            "sqlite": ["Lightweight", "Embedded", "ACID Compliant", "Cross-platform"],
            "mssql": ["Enterprise Features", "Advanced Analytics", "Business Intelligence", "High Availability"],
            "oracle": ["Enterprise Grade", "Advanced Analytics", "PL/SQL", "High Availability"],
            "mongodb": ["Document Store", "Flexible Schema", "Aggregation Pipeline", "Horizontal Scaling"]
        }
        return features.get(database_type, ["Standard SQL Support"])

    def get_customer_status(self, customer_id: str) -> CustomerStatus:
        if customer_id not in self.customers:
            raise HTTPException(status_code=404, detail="Customer not found")

        customer = self.customers[customer_id]
        connection_test = _test_stored_connection(customer)

        customer["last_health_check"] = datetime.now().isoformat()
        customer["connection_healthy"] = connection_test["success"]
        if not connection_test["success"]:
            customer["last_error"] = connection_test["error"]

        self.save_customers()

        return CustomerStatus(
            customer_id=customer_id,
            database_type=customer["database_type"],
            status=customer["status"],
            last_health_check=customer["last_health_check"],
            connection_healthy=connection_test["success"],
            error_message=connection_test.get("error")
        )

    def list_customers(self, caller: "Caller") -> List[Dict[str, Any]]:
        """The caller's own customers only. Never a secret: no connection_string, no api_key."""
        customers = self._read_customers_file()
        return [
            {
                "customer_id": cid,
                "customer_name": data.get("customer_name"),
                "database_type": data.get("database_type"),
                "environment": data.get("environment"),
                "status": data.get("status"),
                "created_at": data.get("created_at"),
                "last_health_check": data.get("last_health_check"),
                "database_info": {
                    "database": data.get("database_info", {}).get("database"),
                    "table_count": data.get("database_info", {}).get("table_count"),
                    "version": data.get("database_info", {}).get("version")
                }
            }
            for cid, data in customers.items()
            if caller.owns(cid, data)
        ] + [
            # Saved, but not openable under this deployment's key: listed, with
            # what is wrong and what to do, rather than silently left out.
            {
                "customer_id": cid,
                "customer_name": data.get("customer_name"),
                "database_type": data.get("database_type"),
                "environment": data.get("environment"),
                "status": NEEDS_REREGISTERING,
                "message": f"This connection {WRONG_KEY_MESSAGE}.",
                "created_at": data.get("created_at"),
            }
            for cid, data in self.unreadable_customers().items()
            if data.get("status") != "inactive" and cid not in customers and caller.owns(cid, data)
        ]


def _test_stored_connection(customer: Dict[str, Any]) -> Dict[str, Any]:
    """Health-check a stored connection_string, dispatching Mongo vs SQL."""
    conn_str = customer.get("connection_string", "")
    db_type = customer.get("database_type", "")

    if db_type == "mongodb":
        if not _PYMONGO_AVAILABLE:
            return {"success": False, "error": "pymongo not installed"}
        client = None
        try:
            client = MongoClient(conn_str, serverSelectionTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS)
            client.admin.command("ping")
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            if client is not None:
                client.close()

    try:
        engine = create_engine(conn_str, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


# FastAPI Application
app = FastAPI(
    title="Avaloka Multi-Database Customer Onboarding API",
    description="API for registering multiple database types with Avaloka MCP",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The MCP server's own default port (multi_tenant_mcp_server binds MCP_SERVER_PORT,
# default 8080). It said :8082, which nothing bound.
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080")
customer_manager = CustomerManager(MCP_SERVER_URL)

router = APIRouter()


# ---- Who is calling ----
#
# Every /customers route needs one of three identities:
#
# * a customer's own api_key (#526), as Bearer -- scoped to that one customer;
# * a user assertion (credentials.user_assertion), as Bearer -- the user the
#   API server authenticated, scoped to the customers that user registered;
# * the API service (#564): the shared MCP_SERVICE_TOKEN in
#   X-Avaloka-Service-Token plus the user it acts for in X-Avaloka-User --
#   scoped like that user's assertion, with one exception below.
#
# No identity, or one that does not verify, is 401. An identity that is fine
# but does not own the customer named is 404, the same answer as a customer
# that does not exist, so ids are not confirmed to anyone they do not belong to.
#
# The one exception (#618), for customers registered before owners were
# recorded: a record bound to nobody is reachable by the service-token
# identity, for whichever user it names, on the routes a transfer needs
# (status, test-connection, credentials) -- so a legacy database can still be
# a transfer endpoint. Never by a user assertion, never by another customer's
# api_key, never for delete, and never listed. POST /customers/{id}/owner
# binds such a record to the user who proves its api_key, which closes it.
SERVICE_TOKEN_HEADER = "X-Avaloka-Service-Token"
SERVICE_USER_HEADER = "X-Avaloka-User"


def _is_service_request(request: Request) -> bool:
    """True only when MCP_SERVICE_TOKEN is set AND the request presents it.

    Fails closed: with the variable unset or empty no request is a service
    request. The MCP server's /admin routes do not accept this token; they
    take only the signed service MAC (credentials.verify_service_mac).
    """
    expected = os.getenv("MCP_SERVICE_TOKEN", "")
    presented = request.headers.get(SERVICE_TOKEN_HEADER, "")
    if not expected or not presented:
        return False
    return hmac.compare_digest(presented.encode(), expected.encode())


class Caller:
    def __init__(self, user_id: Optional[str] = None, customer_id: Optional[str] = None,
                 is_service: bool = False):
        self.user_id = user_id
        self.customer_id = customer_id
        self.is_service = is_service

    def owns(self, customer_id: str, record: Dict[str, Any]) -> bool:
        if self.customer_id is not None:
            return customer_id == self.customer_id
        # A record registered before owners were recorded has none and is
        # reachable only by its api_key: unowned is nobody's, not everybody's
        # (the service-token exception is _require_transfer_customer, not this).
        owner = record.get("owner_user_id")
        return bool(self.user_id) and bool(owner) and owner == self.user_id


def _is_unbound(record: Dict[str, Any]) -> bool:
    return not (record.get("owner_user_id") or "").strip()


def _bearer_token(authorization: Optional[str]) -> str:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[len("bearer "):].strip()
    return ""


def _require_caller(request: Request, authorization: Optional[str] = Header(None)) -> Caller:
    """Resolve the caller to the service, a user or a customer, reading the registry fresh.

    Fresh, so a key revoked by a re-registration stops working at once.
    """
    if _is_service_request(request):
        user_id = request.headers.get(SERVICE_USER_HEADER, "").strip()
        if not user_id:
            raise HTTPException(status_code=401, detail="Authentication required")
        return Caller(user_id=user_id, is_service=True)
    token = _bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing Bearer credential")
    user_id = verify_user_assertion(token)
    if user_id:
        return Caller(user_id=user_id)
    for cid, record in customer_manager._read_customers_file().items():
        # A deactivated customer's key is revoked here too, not only on /call_tool.
        if record.get("status", "active") == "active" and api_key_matches(record, token):
            return Caller(customer_id=cid)
    raise HTTPException(status_code=401, detail="Invalid credential")


def _require_user(caller: Caller = Depends(_require_caller)) -> Caller:
    """Routes that act for a person -- registration -- need the user, not a database's key."""
    if not caller.user_id:
        raise HTTPException(status_code=401, detail="A user credential is required")
    return caller


def _refuse_unreadable(customer_id: str, caller: "Caller", service_may_use_unbound: bool = False) -> None:
    """409, with the truth, for a record the caller may reach that this key cannot open."""
    record = customer_manager.unreadable_customers().get(customer_id)
    if record is None or record.get("status") == "inactive":
        return
    if caller.owns(customer_id, record) or (service_may_use_unbound and caller.is_service
                                             and _is_unbound(record)):
        raise HTTPException(status_code=409, detail=f"Connection {customer_id}: {WRONG_KEY_MESSAGE}.")


def _require_customer(customer_id: str, caller: Caller = Depends(_require_caller)) -> Dict[str, Any]:
    """The named customer's record, if the caller owns it; 404 otherwise (409 if it cannot be opened)."""
    customer = customer_manager._read_customers_file().get(customer_id)
    if customer is None:
        _refuse_unreadable(customer_id, caller)
    if customer is None or not caller.owns(customer_id, customer):
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


def _require_transfer_customer(customer_id: str, caller: Caller = Depends(_require_caller)) -> Dict[str, Any]:
    """An ACTIVE customer the caller owns, or (#618) an active unbound one for the service token; 404 otherwise.

    For the routes a transfer uses. A deactivated customer is not a transfer
    endpoint for anyone, its owner included (#564): its handle stays sealed.
    """
    customer = customer_manager._read_customers_file().get(customer_id)
    if customer is None:
        _refuse_unreadable(customer_id, caller, service_may_use_unbound=True)
    if customer is None or customer.get("status", "active") != "active":
        raise HTTPException(status_code=404, detail="Customer not found")
    if caller.owns(customer_id, customer):
        return customer
    if caller.is_service and _is_unbound(customer):
        return customer
    raise HTTPException(status_code=404, detail="Customer not found")


@router.post("/customers/register", response_model=CustomerResponse)
async def register_customer(
    registration: DatabaseConnectionRequest,
    background_tasks: BackgroundTasks,
    caller: Caller = Depends(_require_user),
):
    """
    Register a new customer database with Avaloka, owned by the calling user.

    Supports PostgreSQL, MySQL / MariaDB, SQLite, SQL Server (MSSQL),
    Oracle, and MongoDB.
    """
    return await customer_manager.register_customer(registration, owner_user_id=caller.user_id)


@router.get("/customers/{customer_id}/status", response_model=CustomerStatus)
async def get_customer_status(customer_id: str, customer: Dict[str, Any] = Depends(_require_transfer_customer)):
    customer_manager.customers = customer_manager._read_customers_file()
    return customer_manager.get_customer_status(customer_id)


@router.get("/customers")
async def list_customers(caller: Caller = Depends(_require_caller)):
    customers = customer_manager.list_customers(caller)
    return {
        "customers": customers,
        "total_count": len(customers),
        "supported_databases": SUPPORTED_DB_TYPES
    }


def _require_deletable_customer(customer_id: str, caller: Caller = Depends(_require_caller)) -> Dict[str, Any]:
    """As _require_customer, but its owner may also delete a record this key cannot open."""
    try:
        return _require_customer(customer_id, caller)
    except HTTPException as e:
        if e.status_code != 409:
            raise
        return {"customer_id": customer_id, "status": NEEDS_REREGISTERING}


@router.delete("/customers/{customer_id}")
async def deactivate_customer(customer_id: str, customer: Dict[str, Any] = Depends(_require_deletable_customer)):
    if customer.get("status") == NEEDS_REREGISTERING:
        # Not openable under this key: mark it deleted on disk as it is.
        mark_unreadable_inactive(customer_manager.customers_file, customer_id, datetime.now().isoformat())
        evicted = await customer_manager.evict_from_mcp(customer_id)
        return {"message": f"Customer {customer_id} deactivated", "mcp_evicted": evicted}
    # Mutate the fresh registry, not a possibly stale in-memory copy of it.
    customer_manager.customers = customer_manager._read_customers_file()
    customer_manager.customers[customer_id]["status"] = "inactive"
    customer_manager.customers[customer_id]["deactivated_at"] = datetime.now().isoformat()
    customer_manager.save_customers()
    # The record is the authority; evicting is what makes the MCP server
    # forget the key and dispose of the engine without waiting to notice.
    evicted = await customer_manager.evict_from_mcp(customer_id)

    return {"message": f"Customer {customer_id} deactivated", "mcp_evicted": evicted}


@router.post("/customers/{customer_id}/test-connection")
async def test_customer_connection(customer_id: str, customer: Dict[str, Any] = Depends(_require_transfer_customer)):
    db_type = customer["database_type"]

    # ---- MongoDB ----
    if db_type == "mongodb":
        if not _PYMONGO_AVAILABLE:
            connection_test = {"success": False, "error": "pymongo not installed", "database_type": "mongodb"}
        else:
            client = None
            try:
                client = MongoClient(customer["connection_string"], serverSelectionTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS)
                client.admin.command("ping")
                version = ""
                try:
                    version = client.server_info().get("version", "")
                except Exception:
                    pass
                parsed = urlparse(customer["connection_string"])
                database = (parsed.path or "").lstrip("/").split("?")[0]
                connection_test = {
                    "success": True,
                    "database": database,
                    "version": version.split()[0] if version else "Unknown",
                    "database_type": "mongodb",
                }
            except Exception as e:
                connection_test = {"success": False, "error": str(e), "database_type": "mongodb"}
            finally:
                if client is not None:
                    client.close()
        return {
            "customer_id": customer_id,
            "connection_test": connection_test,
            "timestamp": datetime.now().isoformat(),
        }

    # ---- SQL ----
    try:
        engine = create_engine(customer["connection_string"], pool_pre_ping=True)
        with engine.connect() as conn:
            if db_type == "postgresql":
                result = conn.execute(text("SELECT version(), current_database()"))
                version, database = result.fetchone()
            elif db_type in ["mysql", "mariadb"]:
                result = conn.execute(text("SELECT VERSION(), DATABASE()"))
                version, database = result.fetchone()
            elif db_type == "sqlite":
                result = conn.execute(text("SELECT sqlite_version()"))
                version = result.fetchone()[0]
                database = db_type
            elif db_type == "mssql":
                result = conn.execute(text("SELECT @@VERSION, DB_NAME()"))
                version, database = result.fetchone()
            elif db_type == "oracle":
                result = conn.execute(text("SELECT * FROM v$version WHERE banner LIKE 'Oracle%' AND ROWNUM = 1"))
                version_row = result.fetchone()
                version = version_row[0] if version_row else "Oracle"
                result = conn.execute(text("SELECT SYS_CONTEXT('USERENV', 'DB_NAME') FROM dual"))
                database = result.fetchone()[0]

        engine.dispose()

        connection_test = {
            "success": True,
            "database": database,
            "version": version.split()[0] if version else "Unknown",
            "database_type": db_type
        }

    except Exception as e:
        connection_test = {
            "success": False,
            "error": str(e),
            "database_type": db_type
        }

    return {
        "customer_id": customer_id,
        "connection_test": connection_test,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/supported-databases")
async def get_supported_databases():
    return {
        "databases": {
            "postgresql": {
                "name": "PostgreSQL",
                "default_port": 5432,
                "required_fields": ["host", "port", "database_name", "username", "password"],
                "optional_fields": ["connection_params"],
                "features": ["Advanced Analytics", "JSON Support", "Full-text Search"]
            },
            "mysql": {
                "name": "MySQL",
                "default_port": 3306,
                "required_fields": ["host", "port", "database_name", "username", "password"],
                "optional_fields": ["connection_params"],
                "features": ["High Performance", "Replication", "Full-text Indexing"]
            },
            "mariadb": {
                "name": "MariaDB",
                "default_port": 3306,
                "required_fields": ["host", "port", "database_name", "username", "password"],
                "optional_fields": ["connection_params"],
                "features": ["High Performance", "Replication", "JSON Support"]
            },
            "sqlite": {
                "name": "SQLite",
                "default_port": None,
                "required_fields": ["database_name"],
                "optional_fields": ["connection_params"],
                "features": ["Lightweight", "Embedded", "ACID Compliant"]
            },
            "mssql": {
                "name": "SQL Server",
                "default_port": 1433,
                "required_fields": ["host", "port", "database_name", "username", "password"],
                "optional_fields": ["connection_params"],
                "features": ["Enterprise Features", "Advanced Analytics", "Business Intelligence"]
            },
            "oracle": {
                "name": "Oracle Database",
                "default_port": 1521,
                "required_fields": ["host", "port", "database_name", "username", "password"],
                "optional_fields": ["connection_params"],
                "features": ["Enterprise Grade", "Advanced Analytics", "PL/SQL"]
            },
            "mongodb": {
                "name": "MongoDB",
                "default_port": 27017,
                # host + database_name are enough for an unauthenticated local Mongo;
                # username/password (or a full connection_string) are used when the
                # server requires auth or you are on Atlas (mongodb+srv://).
                "required_fields": ["host", "database_name"],
                "optional_fields": ["port", "username", "password", "connection_string", "connection_params"],
                "features": ["Document Store", "Flexible Schema", "Aggregation Pipeline", "Horizontal Scaling"]
            }
        }
    }


def _require_service_class(caller: Caller = Depends(_require_caller)) -> Caller:
    """A user assertion or the service token; a customer's api_key is 403 here.

    Checked before the record is looked up, so the answer says nothing about
    which ids exist.
    """
    if caller.customer_id is not None or not caller.user_id:
        raise HTTPException(status_code=403, detail="Credentials are released only to the Avaloka API service")
    return caller


@router.get("/customers/{customer_id}/credentials")
async def get_customer_credentials(
    customer_id: str,
    _caller: Caller = Depends(_require_service_class),
    customer: Dict[str, Any] = Depends(_require_transfer_customer),
):
    """Return where a connection points plus a credential *handle* for DTA transfer use.

    Only to a service-class caller: a user assertion for a customer that user
    owns, or the service token (the user's own customers, plus a legacy one
    bound to nobody, #618). A customer's api_key is 403. The password is not in
    the response, to anyone: ``credential_handle`` is the connection string
    sealed under DB_ENCRYPTION_KEY; the API, which holds the same key, opens it
    in process (app.mcp_server.credentials.open_credential_handle).
    """
    conn_str = customer.get("connection_string", "")
    db_type = customer.get("database_type", "")

    try:
        parsed = urlparse(conn_str)
        database = (parsed.path or "").lstrip("/").split("?")[0]

        # Auto-detect first table / collection from the database.
        table = None

        if db_type == "mongodb":
            if _PYMONGO_AVAILABLE:
                client = None
                try:
                    client = MongoClient(conn_str, serverSelectionTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS)
                    names = client[database].list_collection_names()
                    table = names[0] if names else None
                except Exception as te:
                    print(f"[credentials] Could not auto-detect collection for {customer_id}: {te}")
                finally:
                    if client is not None:
                        client.close()
        else:
            try:
                engine = create_engine(conn_str, pool_pre_ping=True)
                with engine.connect() as conn:
                    if db_type == "postgresql":
                        result = conn.execute(text(
                            "SELECT table_name FROM information_schema.tables "
                            "WHERE table_schema='public' ORDER BY table_name LIMIT 1"
                        ))
                    elif db_type in ("mysql", "mariadb"):
                        result = conn.execute(text(
                            f"SELECT table_name FROM information_schema.tables "
                            f"WHERE table_schema='{database}' ORDER BY table_name LIMIT 1"
                        ))
                    else:
                        result = None

                    if result:
                        row = result.fetchone()
                        table = row[0] if row else None
                engine.dispose()
            except Exception as te:
                print(f"[credentials] Could not auto-detect table for {customer_id}: {te}")

        return {
            "customer_id":   customer_id,
            "database_type": db_type,
            "host":          parsed.hostname or "localhost",
            "port":          parsed.port,
            "database":      database,
            "username":      unquote(parsed.username) if parsed.username else None,
            "table":         table,
            "credential_handle": credential_handle(conn_str),
        }
    except CredentialKeyMissing as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not parse credentials: {e}")


@router.post("/customers/{customer_id}/owner")
async def bind_customer_owner(customer_id: str, request: Request):
    """Bind a customer registered before owners existed to the user the service names.

    Service-token only, and the service must also present that customer's own
    api_key (Bearer), so it can bind only a customer the user really holds. A
    customer already bound to someone else is never re-bound; a deactivated
    one is never bound.
    """
    if not _is_service_request(request):
        raise HTTPException(status_code=401, detail="Authentication required")
    user_id = request.headers.get(SERVICE_USER_HEADER, "").strip()
    api_key = _bearer_token(request.headers.get("authorization"))
    if not user_id or not api_key:
        raise HTTPException(status_code=401, detail="Authentication required")

    customer_manager.customers = customer_manager._read_customers_file()
    record = customer_manager.customers.get(customer_id)
    current = (record or {}).get("owner_user_id")
    if (
        record is None
        or record.get("status") == "inactive"
        or not api_key_matches(record, api_key)
        or (current and current != user_id)
    ):
        raise HTTPException(status_code=404, detail="Customer not found")
    if not current:
        record["owner_user_id"] = user_id
        customer_manager.save_customers()
    return {"customer_id": customer_id, "owner_bound": True}


PLANNER_METADATA_MAX_BYTES = int(os.getenv("AVALOKA_PLANNER_METADATA_MAX_BYTES", "16384"))


@router.get("/customers/{customer_id}/planner-metadata")
async def get_customer_planner_metadata(
    customer_id: str,
    customer: Dict[str, Any] = Depends(_require_customer),
    max_bytes: int = PLANNER_METADATA_MAX_BYTES,
):
    """The size-capped catalogue and statistics a planner may see: no row values, no credential.

    Built through the connector contract (avaloka.connectors) with the
    credential resolved here, in process. Document databases are not behind
    the contract yet (L2), so they answer 501 rather than a half-shaped doc.
    """
    db_type = customer.get("database_type", "")
    if db_type == "mongodb":
        raise HTTPException(status_code=501, detail="planner metadata for MongoDB is not behind the connector contract yet")
    from avaloka.connectors.postgres import SQLAlchemyConnector

    try:
        connector = SQLAlchemyConnector(customer.get("connection_string", ""))
        try:
            metadata = connector.planner_metadata(max_bytes=max(1024, min(int(max_bytes), 1 << 20)))
        finally:
            connector.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read planner metadata: {type(e).__name__}")
    return {"customer_id": customer_id, "database_type": db_type, "source_metadata": metadata}


@app.on_event("startup")
async def _require_encryption_key() -> None:
    # The standalone onboarding service seals every connection string under
    # DB_ENCRYPTION_KEY: provisioned from .env when missing; it refuses to
    # start only if it still has none (app/core/encryption_key.py).
    from app.core.encryption_key import require_db_encryption_key
    require_db_encryption_key("The onboarding service")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mcp_server": MCP_SERVER_URL,
        "pymongo_available": _PYMONGO_AVAILABLE,
        "customers_registered": len(customer_manager.customers),
        "database_types": list(set(c.get("database_type", "unknown") for c in customer_manager.customers.values())),
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    print("Starting Avaloka Multi-Database Customer Onboarding API...")
    print(f"MCP Server URL: {MCP_SERVER_URL}")
    print("Supported databases: PostgreSQL, MySQL, MariaDB, SQLite, SQL Server, Oracle, MongoDB")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("ONBOARDING_PORT", "8081")),
        log_level="info"
    )