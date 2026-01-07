from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_neo4j_mcp_tools():
    """Get Neo4j MCP tools from the MCP server.

    Returns:
        List of LangChain tools for Neo4j operations (read_neo4j_cypher, write_neo4j_cypher, get_neo4j_schema).
    """
    client = MultiServerMCPClient(
        {
            "neo4j": {
                "url": "http://localhost:8001/api/mcp/",
                "transport": "http",
            }
        }
    )

    tools = await client.get_tools()
    return tools


