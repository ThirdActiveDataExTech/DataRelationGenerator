from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import settings


async def get_neo4j_mcp_tools():
    """Get Neo4j MCP tools from the MCP server.

    Returns:
        List of LangChain tools for Neo4j operations (read_neo4j_cypher, write_neo4j_cypher).
    """
    client = MultiServerMCPClient(
        {
            "neo4j": {
                "url": settings.neo4j_mcp_url,
                "transport": settings.NEO4J_TRANSPORT,
            }
        }
    )

    tools = await client.get_tools()
    return [tool for tool in tools if tool.name != "get_neo4j_schema"]  # Exclude schema tool
