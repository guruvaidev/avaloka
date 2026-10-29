"""
Multi-Database Customer Onboarding API for Avaloka MCP Integration

Handles customer database registration and API key management for secure
multi-tenant access to PostgreSQL, MySQL/MariaDB, SQLite, SQL Server, Oracle
(via SQLAlchemy) and MongoDB (via pymongo).
"""

import os
import sys
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio
from urllib.parse import quote_plus

import sqlalchemy as sa
from sqlalchemy import create_engine, text
from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, Field
import httpx
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from urllib.parse import urlparse, unquote

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


# ---- Caller identity ----
# Every /customers route needs an identity. There are exactly two kinds:
#   * an end user, holding the customer api_key that /customers/register issued
#     (Authorization: Bearer <api_key>);
#   * the Avaloka API service, presenting the shared MCP_SERVICE_TOKEN and naming
#     the end user it acts for. Only this caller is ever given a password.
# A caller acts within one owner group: the Avaloka user a customer is bound to,
# or, for a customer bound to nobody, that customer alone. Anything outside the
# group is answered 404, never 403, so customer ids do not leak.
SERVICE_TOKEN_HEADER = "X-Avaloka-Service-Token"
SERVICE_USER_HEADER = "X-Avaloka-User"


@dataclass(frozen=True)
class Caller:
    owner: str
    user_id: Optional[str] = None
    is_service: bool = False


def _is_service_request(request: Request) -> bool:
    """True only when MCP_SERVICE_TOKEN is set AND the request presents it.

    Fails closed: with the variable unset or empty no request is a service
    request, so no password can be released.
    """
    expected = os.getenv("MCP_SERVICE_TOKEN", "")
    presented = request.headers.get(SERVICE_TOKEN_HEADER, "")
    if not expected or not presented:
        return False
    return hmac.compare_digest(presented.encode(), expected.encode())


def _service_headers() -> Dict[str, str]:
    token = os.getenv("MCP_SERVICE_TOKEN", "")
    return {SERVICE_TOKEN_HEADER: token} if token else {}


def _bearer(request: Request) -> str:
    scheme, _, value = request.headers.get("authorization", "").partition(" ")
    return value.strip() if scheme.lower() == "bearer" else ""


def _owner_of(customer_id: str, record: Dict[str, Any]) -> str:
    owner_user_id = (record.get("owner_user_id") or "").strip()
    return f"user:{owner_user_id}" if owner_user_id else f"customer:{customer_id}"


def _caller_for_api_key(api_key: str, customers: Dict[str, Any]) -> Optional[Caller]:
    for cid, record in customers.items():
        stored = record.get("api_key") or ""
        if stored and record.get("status") != "inactive" and hmac.compare_digest(
            stored.encode(), api_key.encode()
        ):
            return Caller(owner=_owner_of(cid, record), user_id=record.get("owner_user_id") or None)
    return None


def resolve_caller(request: Request) -> Optional[Caller]:
    if _is_service_request(request):
        user_id = request.headers.get(SERVICE_USER_HEADER, "").strip()
        if not user_id:
            return None
        return Caller(owner=f"user:{user_id}", user_id=user_id, is_service=True)
    api_key = _bearer(request)
    if api_key:
        return _caller_for_api_key(api_key, customer_manager._read_customers_file())
    return None


def require_caller(request: Request) -> Caller:
    caller = resolve_caller(request)
    if caller is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return caller


def _is_retired(record: Dict[str, Any]) -> bool:
    """A deleted customer that was bound to nobody: its id is never reused.

    Saved connections and DTA aliases can still name it, so whoever registered
    it again with their own DSN would receive the next user's transfer.
    """
    return record.get("status") == "inactive" and not (record.get("owner_user_id") or "").strip()


def _may_replace(record: Dict[str, Any], owner_user_id: Optional[str]) -> bool:
    """A deleted customer's id may be registered again only by the user it was bound to."""
    if record.get("status") != "inactive":
        return False
    bound = (record.get("owner_user_id") or "").strip()
    return bool(bound) and bound == (owner_user_id or "")


async def _evict_from_mcp(customer_id: str) -> None:
    """Drop a deleted customer's key and engine from the MCP server.

    In-process when this router is mounted on the MCP server (the chart's
    layout); otherwise through its service-only DELETE /admin/customers/{id}.
    Best-effort: the MCP server also checks customers.json on every call, so
    a missed eviction cannot keep the key alive.
    """
    mts = sys.modules.get("app.mcp_server.multi_tenant_mcp_server")
    if mts is not None and hasattr(mts, "_evict_customer"):
        mts._evict_customer(customer_id)
        return
    headers = _service_headers()
    if not headers:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.delete(f"{customer_manager.mcp_server_url}/admin/customers/{customer_id}", headers=headers)
    except Exception as e:
        logger.warning(f"Could not evict {customer_id} from the MCP server: {e}")


def _owned_customer(customers: Dict[str, Any], customer_id: str, caller: Caller) -> Dict[str, Any]:
    record = customers.get(customer_id)
    if (
        record is None
        or record.get("status") == "inactive"
        or _owner_of(customer_id, record) != caller.owner
    ):
        raise HTTPException(status_code=404, detail="Customer not found")
    return record


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
        """Read the customers registry FRESH from disk (see original notes)."""
        try:
            if os.path.exists(self.customers_file):
                with open(self.customers_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error reading customers file: {e}")
            return getattr(self, "customers", {})

    def load_customers(self):
        self.customers = self._read_customers_file()

    def save_customers(self):
        try:
            with open(self.customers_file, 'w') as f:
                json.dump(self.customers, f, indent=2)
        except Exception as e:
            print(f"Error saving customers: {e}")

    def generate_api_key(self, customer_id: str) -> str:
        timestamp = str(int(datetime.now().timestamp()))
        raw_key = f"{customer_id}:{timestamp}:{os.urandom(16).hex()}"
        return f"ak_{hashlib.sha256(raw_key.encode()).hexdigest()[:32]}"

    async def register_customer(
        self, registration: DatabaseConnectionRequest, owner_user_id: Optional[str] = None
    ) -> CustomerResponse:
        """Register a new customer with their database.

        Only ever creates: an id already on disk is refused, so a registration can
        never overwrite another customer's record or api_key. A deleted id may be
        registered again (see _may_replace); the old key stays dead.
        """

        self.load_customers()
        existing = self.customers.get(registration.customer_id)
        if existing is not None and _is_retired(existing):
            raise HTTPException(
                status_code=409,
                detail=f"Customer {registration.customer_id} was deleted; choose a new id",
            )
        if existing is not None and not _may_replace(existing, owner_user_id):
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
        if owner_user_id:
            customer_data["owner_user_id"] = owner_user_id

        try:
            async with httpx.AsyncClient() as client:
                mcp_payload = {
                    "customer_id": registration.customer_id,
                    "connection_string": connection_string,
                    "database_type": registration.database_type,
                    "api_key": api_key,
                    "created_at": datetime.now().isoformat(),
                    "status": "active",
                    "max_rows": registration.max_rows,
                    "query_timeout": registration.query_timeout
                }

                mcp_response = await client.post(
                    f"{self.mcp_server_url}/admin/customers",
                    json=mcp_payload,
                    headers=_service_headers(),
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
        self.save_customers()

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

    def list_customers(self, caller: Caller) -> List[Dict[str, Any]]:
        """The caller's own customers only. Never carries connection_string,
        password or api_key."""
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
            if _owner_of(cid, data) == caller.owner
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

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8082")
customer_manager = CustomerManager(MCP_SERVER_URL)

router = APIRouter()


@router.post("/customers/register", response_model=CustomerResponse)
async def register_customer(
    registration: DatabaseConnectionRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """
    Register a new customer database with Avaloka.

    Supports PostgreSQL, MySQL / MariaDB, SQLite, SQL Server (MSSQL),
    Oracle, and MongoDB.

    Open, but it only creates: an existing id is refused with 409 and nothing of
    it is read or changed. A deleted id that was bound to a user may be
    registered again only by that user (the API service naming them); a deleted
    id that was bound to nobody is retired for good. The new customer is bound to the caller's Avaloka user
    when there is one (the API service naming its user, or a key already bound to
    a user); otherwise it is bound to nobody and only its own api_key reaches it.
    """
    caller = resolve_caller(request)
    owner_user_id = caller.user_id if caller else None
    return await customer_manager.register_customer(registration, owner_user_id=owner_user_id)


@router.get("/customers/{customer_id}/status", response_model=CustomerStatus)
async def get_customer_status(customer_id: str, caller: Caller = Depends(require_caller)):
    _owned_customer(customer_manager.customers, customer_id, caller)
    return customer_manager.get_customer_status(customer_id)


@router.get("/customers")
async def list_customers(caller: Caller = Depends(require_caller)):
    customers = customer_manager.list_customers(caller)
    return {
        "customers": customers,
        "total_count": len(customers),
        "supported_databases": SUPPORTED_DB_TYPES
    }


@router.delete("/customers/{customer_id}")
async def deactivate_customer(customer_id: str, caller: Caller = Depends(require_caller)):
    _owned_customer(customer_manager.customers, customer_id, caller)

    customer_manager.customers[customer_id]["status"] = "inactive"
    customer_manager.customers[customer_id]["deactivated_at"] = datetime.now().isoformat()
    customer_manager.save_customers()
    await _evict_from_mcp(customer_id)

    return {"message": f"Customer {customer_id} deactivated"}


@router.post("/customers/{customer_id}/test-connection")
async def test_customer_connection(customer_id: str, caller: Caller = Depends(require_caller)):
    customer = _owned_customer(customer_manager.customers, customer_id, caller)
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


@router.get("/customers/{customer_id}/credentials")
async def get_customer_credentials(customer_id: str, caller: Caller = Depends(require_caller)):
    """Return full connection credentials parsed from connection_string for DTA transfer use.

    The password goes only to the Avaloka API service (MCP_SERVICE_TOKEN), and
    only for a customer bound to the user it names. An end user's own api_key
    is not enough.
    """
    _customers = customer_manager._read_customers_file()
    customer = _owned_customer(_customers, customer_id, caller)
    if not caller.is_service:
        raise HTTPException(status_code=403, detail="Credentials are released only to the Avaloka API service")
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
            "username":      parsed.username,
            "password":      unquote(parsed.password or ""),
            "table":         table,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not parse credentials: {e}")


@router.post("/customers/{customer_id}/owner")
async def bind_customer_owner(customer_id: str, request: Request):
    """Bind a customer to the Avaloka user the API service names.

    Service-only, and the service must also present that customer's own api_key,
    so it can bind only a customer the user really holds. A customer already
    bound to someone else is never re-bound.
    """
    if not _is_service_request(request):
        raise HTTPException(status_code=401, detail="Authentication required")
    user_id = request.headers.get(SERVICE_USER_HEADER, "").strip()
    api_key = _bearer(request)
    if not user_id or not api_key:
        raise HTTPException(status_code=401, detail="Authentication required")

    customer_manager.load_customers()
    record = customer_manager.customers.get(customer_id)
    stored = (record or {}).get("api_key") or ""
    current = (record or {}).get("owner_user_id")
    if (
        not stored
        or record.get("status") == "inactive"
        or not hmac.compare_digest(stored.encode(), api_key.encode())
        or (current and current != user_id)
    ):
        raise HTTPException(status_code=404, detail="Customer not found")
    if not current:
        record["owner_user_id"] = user_id
        customer_manager.save_customers()
    return {"customer_id": customer_id, "owner_bound": True}


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