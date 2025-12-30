import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_fetch_server():    
    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-fetch", "--ignore-robots-txt"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            result = await session.call_tool(
                "fetch",
                arguments={
                    "url": "https://example.com",
                    "raw": True
                }
            )
            print("<<< RAW CONTENT >>>")
            print(result.content[0].text)


            result = await session.call_tool(
                "fetch",
                arguments={
                    "url": "https://example.com",
                    "raw": False
                }
            )
            print("<<< PARSED CONTENT >>>")
            print(result.content[0].text)
            

            result = await session.call_tool(
                "fetch",
                arguments={
                    "url": "https://www.allrecipes.com/",
                    "raw": False
                }
            )
            print("<<< PARSED CONTENT (AllRecipes) >>>")
            print(result.content[0].text)

if __name__ == "__main__":
    asyncio.run(test_fetch_server())

