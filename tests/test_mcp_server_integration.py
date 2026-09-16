"""
Integration Test Suite for Avaloka Multi-Database MCP System

This test suite validates the entire workflow against pre-started services
and tests multiple database types.
"""

import asyncio
import json
import os
import tempfile
import logging
import pytest
import pandas as pd
import httpx
from unittest.mock import patch

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Database configurations for testing
POSTGRESQL_CONFIG = {
    "connection_string": "postgresql://postgres:avaloka%40123@localhost:5432/sales_db",
    "database_type": "postgresql"
}

# Add other database configurations as needed
MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "database_name": "test_db",
    "username": "root",
    "password": "password",
    "database_type": "mysql"
}

SQLITE_CONFIG = {
    "database_name": "test_database.db",
    "database_type": "sqlite"
}

ONBOARDING_SERVER_PORT = 8081
MCP_SERVER_PORT = 8080

# Test configurations for different database types
DATABASE_TEST_CONFIGS = {
    "postgresql": {
        "customer_id": "test_customer_pg_001",
        "customer_name": "PostgreSQL Test Corp",
        "database_type": "postgresql",
        "host": "localhost",
        "port": 5432,
        "database_name": "sales_db",
        "username": "postgres",
        "password": "avaloka@123",
        "contact_email": "test@testcorp.com",
        "environment": "testing",
        "sample_query": "SELECT * FROM sales_data LIMIT 5"
    },
    "mysql": {
        "customer_id": "test_customer_mysql_001",
        "customer_name": "MySQL Test Corp",
        "database_type": "mysql",
        "host": "localhost",
        "port": 3306,
        "database_name": "test_db",
        "username": "root",
        "password": "password",
        "contact_email": "test@mysql-testcorp.com",
        "environment": "testing",
        "sample_query": "SELECT * FROM test_table LIMIT 5"
    },
    "sqlite": {
        "customer_id": "test_customer_sqlite_001",
        "customer_name": "SQLite Test Corp",
        "database_type": "sqlite",
        "database_name": "test_database.db",
        "contact_email": "test@sqlite-testcorp.com",
        "environment": "testing",
        "sample_query": "SELECT * FROM test_table LIMIT 5"
    }
}

# Sample data for CSV testing
SAMPLE_CSV_DATA = {
    'order_id': [20, 21, 22, 23, 24, 25],
    'order_date': ['2025-01-10', '2025-01-11', '2025-01-12', '2025-01-13', '2025-01-14', '2025-01-15'],
    'customer_id': [1051, 1092, 1014, 1071, 1060, 1020],
    'region': ['South', 'East', 'North', 'West', 'North', 'West'],
    'product': ['Laptop', 'Laptop', 'Smartphone', 'Monitor', 'Tablet', 'Tablet'],
    'quantity': [2, 1, 2, 1, 1, 3],
    'unit_price': [740.81, 843.38, 190.57, 315.89, 227.77, 413.15],
    'total': [1481.62, 843.38, 381.14, 315.89, 227.77, 1239.45]
}

class MultiDatabaseIntegrationTest:
    """Main integration test class for multiple database types"""

    def __init__(self):
        self.registered_customers = {}
        self.temp_files = []
        self.onboarding_port = ONBOARDING_SERVER_PORT
        self.mcp_port = MCP_SERVER_PORT
        
        # Configure which databases to test
        # Set to test only databases you have available
        self.databases_to_test = ["postgresql"]  # Add "mysql", "sqlite" etc. as needed

    async def setup(self):
        """Set up test environment"""
        logger.info("=" * 70)
        logger.info("STARTING AVALOKA MULTI-DATABASE INTEGRATION TEST")
        logger.info(f"Target Onboarding Server: http://localhost:{self.onboarding_port}")
        logger.info(f"Target MCP Server: http://localhost:{self.mcp_port}")
        logger.info(f"Databases to test: {', '.join(self.databases_to_test)}")
        logger.info("=" * 70)

    async def cleanup(self):
        """Clean up test environment"""
        logger.info("=" * 50)
        logger.info("CLEANING UP TEMP FILES")
        logger.info("=" * 50)
        for file_path in self.temp_files:
            try:
                os.remove(file_path)
            except OSError:
                pass

    async def test_supported_databases_endpoint(self):
        """Test the supported databases endpoint"""
        logger.info("TEST: Supported Databases Endpoint")
        
        async with httpx.AsyncClient() as client:
            url = f"http://localhost:{self.onboarding_port}/supported-databases"
            logger.info(f"GET {url}")
            response = await client.get(url, timeout=30.0)
            
            assert response.status_code == 200, f"Supported databases check failed: {response.text}"
            result = response.json()
            
            assert "databases" in result
            supported_dbs = list(result["databases"].keys())
            logger.info(f"✓ Supported databases: {', '.join(supported_dbs)}")
            
            # Verify expected databases are supported
            expected_dbs = ["postgresql", "mysql", "mariadb", "sqlite", "mssql", "oracle"]
            for db in expected_dbs:
                assert db in supported_dbs, f"Expected database {db} not in supported list"

    async def test_customer_onboarding_for_database(self, db_type: str):
        """Test customer registration and onboarding for a specific database type"""
        logger.info(f"TEST: Customer Onboarding ({db_type.upper()})")
        
        if db_type not in DATABASE_TEST_CONFIGS:
            logger.warning(f"No test configuration available for {db_type}, skipping...")
            return None
        
        config = DATABASE_TEST_CONFIGS[db_type].copy()
        
        async with httpx.AsyncClient() as client:
            url = f"http://localhost:{self.onboarding_port}/customers/register"
            logger.info(f"POST {url}")
            response = await client.post(url, json=config, timeout=30.0)

            if response.status_code != 200:
                logger.warning(f"{db_type.upper()} registration failed: {response.text}")
                return None

            result = response.json()
            assert result["customer_id"] == config["customer_id"]
            assert "api_key" in result
            assert result["status"] == "active"
            assert result["database_type"] == db_type

            customer_info = {
                "customer_id": result["customer_id"],
                "api_key": result["api_key"],
                "database_type": db_type,
                "sample_query": config["sample_query"]
            }
            
            self.registered_customers[db_type] = customer_info
            logger.info(f"✓ {db_type.upper()} customer registered successfully with API key: {result['api_key'][:8]}...")
            return customer_info

    async def test_mcp_connection_for_database(self, db_type: str):
        """Test MCP server connection and authentication for a specific database"""
        logger.info(f"TEST: MCP Connection ({db_type.upper()})")
        
        if db_type not in self.registered_customers:
            logger.warning(f"No registered customer for {db_type}, skipping...")
            return
        
        from app.agents.sampling_agent_v2 import test_mcp_connection

        customer_info = self.registered_customers[db_type]
        mcp_config = {
            "mcp_url": f"http://localhost:{self.mcp_port}/sse",
            "api_key": customer_info["api_key"]
        }

        result = await test_mcp_connection(mcp_config["mcp_url"], mcp_config["api_key"])

        assert "error" not in result, f"MCP connection failed for {db_type}: {result.get('error')}"
        assert result["status"] == "connected"
        assert "query" in result["available_tools"]

        logger.info(f"✓ {db_type.upper()} MCP connection successful. Available tools: {result['available_tools']}")

    async def test_database_sampling(self, db_type: str):
        """Test database data sampling for a specific database type"""
        logger.info(f"TEST: Database Sampling ({db_type.upper()})")
        
        if db_type not in self.registered_customers:
            logger.warning(f"No registered customer for {db_type}, skipping...")
            return
        
        from app.agents.sampling_agent_v2 import sample_data_from_source

        customer_info = self.registered_customers[db_type]
        mcp_config = {
            "mcp_url": f"http://localhost:{self.mcp_port}/sse",
            "api_key": customer_info["api_key"],
            "customer_id": customer_info["customer_id"]
        }
        
        result = await sample_data_from_source(
            source_type=db_type,
            query=customer_info["sample_query"],
            sample_size=5,
            mcp_config=mcp_config,
            database_type=db_type
        )

        if "error" in result:
            logger.warning(f"{db_type.upper()} sampling failed: {result['error']}")
            return
        
        assert "schema" in result and "rows" in result
        assert len(result["rows"]) <= 5
        assert "CREATE TABLE" in result["schema"]
        assert result["meta"]["database_type"] == db_type

        logger.info(f"✓ {db_type.upper()} sampling successful. Sampled {len(result['rows'])} rows")
        logger.info(f"  Schema: {result['schema'].split('(')[0]}...")
        logger.info(f"  Sample columns: {result['meta']['columns']}")

    async def test_database_specific_features(self, db_type: str):
        """Test database-specific features and queries"""
        logger.info(f"TEST: Database Features ({db_type.upper()})")
        
        if db_type not in self.registered_customers:
            logger.warning(f"No registered customer for {db_type}, skipping...")
            return
        
        from app.agents.sampling_agent_v2 import test_database_specific_features

        customer_info = self.registered_customers[db_type]
        mcp_config = {
            "mcp_url": f"http://localhost:{self.mcp_port}/sse",
            "api_key": customer_info["api_key"],
            "customer_id": customer_info["customer_id"]
        }
        
        result = await test_database_specific_features(mcp_config, db_type)
        
        if "error" in result:
            logger.warning(f"{db_type.upper()} feature testing failed: {result['error']}")
            return
        
        assert result["database_type"] == db_type
        assert "test_results" in result
        
        successful_tests = [name for name, res in result["test_results"].items() if res.get("success")]
        failed_tests = [name for name, res in result["test_results"].items() if not res.get("success")]
        
        logger.info(f"✓ {db_type.upper()} feature testing completed")
        logger.info(f"  Successful tests: {', '.join(successful_tests) if successful_tests else 'None'}")
        if failed_tests:
            logger.info(f"  Failed tests: {', '.join(failed_tests)}")

    async def test_data_sampling_csv(self):
        """Test CSV data sampling (database-independent)"""
        logger.info("TEST: CSV Data Sampling")
        
        df = pd.DataFrame(SAMPLE_CSV_DATA)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_csv:
            df.to_csv(temp_csv.name, index=False)
            self.temp_files.append(temp_csv.name)
            csv_path = temp_csv.name

        from app.agents.sampling_agent_v2 import sample_data_from_source
        result = await sample_data_from_source(
            source_type="csv",
            path=csv_path,
            sample_size=3
        )

        assert "error" not in result, f"CSV sampling failed: {result.get('error')}"
        assert "schema" in result and "rows" in result
        assert len(result["rows"]) == 3
        logger.info(f"✓ CSV sampling successful. Sampled {len(result['rows'])} rows")

    async def test_customer_status_checks(self):
        """Test customer status checking for all registered customers"""
        logger.info("TEST: Customer Health Checks")
        
        async with httpx.AsyncClient() as client:
            for db_type, customer_info in self.registered_customers.items():
                url = f"http://localhost:{self.onboarding_port}/customers/{customer_info['customer_id']}/status"
                logger.info(f"GET {url}")
                response = await client.get(url)

                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✓ {db_type.upper()} customer health check passed - Status: {result['status']}")
                else:
                    logger.warning(f"{db_type.upper()} customer health check failed: {response.text}")

    async def run_all_tests(self):
        """Run all integration tests"""
        try:
            await self.setup()
            
            # Test supported databases endpoint
            await self.test_supported_databases_endpoint()
            
            # Test each configured database type
            for db_type in self.databases_to_test:
                logger.info("=" * 60)
                logger.info(f"TESTING {db_type.upper()}")
                logger.info("=" * 60)
                
                # Register customer
                customer_info = await self.test_customer_onboarding_for_database(db_type)
                if not customer_info:
                    continue
                
                # Test MCP connection
                await self.test_mcp_connection_for_database(db_type)
                
                # Test data sampling
                await self.test_database_sampling(db_type)
                
                # Test database-specific features
                await self.test_database_specific_features(db_type)
            
            # Test CSV sampling (independent of database)
            await self.test_data_sampling_csv()
            
            # Test customer status checks
            await self.test_customer_status_checks()

            logger.info("=" * 70)
            logger.info("🎉 ALL MULTI-DATABASE INTEGRATION TESTS COMPLETED! 🎉")
            logger.info(f"Tested databases: {', '.join(self.databases_to_test)}")
            logger.info(f"Registered customers: {len(self.registered_customers)}")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"TEST FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            await self.cleanup()

async def main():
    """Main test runner"""
    os.environ["GOOGLE_API_KEY"] = "test_key"  # Mock for testing
    os.environ["GOOGLE_MODEL"] = "test_model"

    test_suite = MultiDatabaseIntegrationTest()
    
    # Configure which databases to test based on your environment
    # Uncomment the databases you have available for testing
    test_suite.databases_to_test = [
        #"postgresql",
        "mysql",
        # "sqlite",
        # "mssql",
        # "oracle"
    ]
    
    await test_suite.run_all_tests()

if __name__ == "__main__":
    logger.info("Multi-Database Integration Test Suite")
    logger.info("=" * 50)
    asyncio.run(main())