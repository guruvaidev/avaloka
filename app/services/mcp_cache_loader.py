import asyncio
import logging
import json
from typing import Dict, Any, Optional

from app.agents.sampling_agent_v2 import MCPConnectionManager

logger = logging.getLogger(__name__)

async def fetch_schema_from_mcp(mcp_url: str, api_key: str, dataset_id: str) -> Optional[Dict[str, Any]]:
    """
    Connects to the enterprise multi-tenant MCP server securely to extract 
    the schema representation and tables. 
    This is requested during a Domain Context (Layer 1) Cache Miss.
    """
    logger.info(f"Layer 1 Cache Miss! Fetching schema structure for {dataset_id} via MCP: {mcp_url}")
    try:
        connection_manager = MCPConnectionManager(mcp_url=mcp_url, api_key=api_key)
        
        async with connection_manager.get_session() as session:
            schema_resp = await session.call_tool("list_tables", {})
            
            if schema_resp and hasattr(schema_resp, "content") and schema_resp.content:
                text_content = schema_resp.content[0].text
                try:
                    payload = json.loads(text_content)
                    logger.info(f"Successfully retrieved Layer 1 Domain Context via MCP for {dataset_id}")
                    return payload
                except json.JSONDecodeError as decode_error:
                    logger.error(f"Failed to parse list_tables response from MCP: {decode_error}")
                    return {"raw_text": text_content, "error": "Invalid JSON format returned from MCP tool"}
                    
    except Exception as e:
        logger.error(f"Failed to fetch MCP schema for Domain Context Layer 1: {e}")
        
    return None

def fetch_schema_sync(mcp_url: str, api_key: str, dataset_id: str) -> Optional[Dict[str, Any]]:
    """Synchronous wrapper for memory_plane.py"""
    try:
        return asyncio.run(fetch_schema_from_mcp(mcp_url, api_key, dataset_id))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        res = loop.run_until_complete(fetch_schema_from_mcp(mcp_url, api_key, dataset_id))
        loop.close()
        return res
