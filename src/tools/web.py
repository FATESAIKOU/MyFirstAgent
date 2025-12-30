"""
網路搜尋相關工具
使用 MCP fetch server 抓取網頁內容
"""
from langchain_core.tools import tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# 預定義的食譜網站
RECIPE_SITES = {
    "allrecipes": {
        "name": "AllRecipes",
        "search_url": "https://www.allrecipes.com/search?q={query}",
        "base_url": "https://www.allrecipes.com"
    },
    "bbcgoodfood": {
        "name": "BBC Good Food",
        "search_url": "https://www.bbcgoodfood.com/search?q={query}",
        "base_url": "https://www.bbcgoodfood.com"
    },
    "foodnetwork": {
        "name": "Food Network",
        "search_url": "https://www.foodnetwork.com/search/{query}-",
        "base_url": "https://www.foodnetwork.com"
    }
}


@tool
async def fetch_webpage(url: str, max_length: int = 5000, use_markdown: bool = True) -> str:
    """
    抓取網頁內容。可以用來搜尋食譜、獲取營養資訊等。
    
    Args:
        url: 要抓取的網頁 URL
        max_length: 最大字元數（預設 5000）
        use_markdown: 是否轉換為 Markdown 格式（預設 True，更易讀）
    
    Returns:
        網頁內容（Markdown 或 HTML）
    
    Examples:
        >>> await fetch_webpage("https://www.allrecipes.com/search?q=chicken+salad")
        >>> await fetch_webpage("https://example.com", use_markdown=False)
    """
    # 設定 MCP server 參數
    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-fetch", "--ignore-robots-txt"]
    )
    
    # 連接並抓取
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            result = await session.call_tool(
                "fetch",
                arguments={
                    "url": url,
                    "max_length": max_length,
                    "raw": not use_markdown
                }
            )
            
            # 提取內容
            if result.content and len(result.content) > 0:
                content = result.content[0]
                if hasattr(content, 'text'):
                    return content.text
            
            return ""


@tool
async def search_recipes(query: str, site: str = "allrecipes", max_results: int = 5) -> str:
    """
    在食譜網站搜尋食譜。
    
    Args:
        query: 搜尋關鍵字（如 "chicken salad", "pasta"）
        site: 網站名稱（allrecipes, bbcgoodfood, foodnetwork）
        max_results: 最多返回幾個結果（預設 5）
    
    Returns:
        搜尋結果摘要，包含食譜標題和連結
    
    Examples:
        >>> await search_recipes("chicken salad")
        >>> await search_recipes("pasta carbonara", site="bbcgoodfood")
    """
    # 驗證網站
    if site not in RECIPE_SITES:
        available = ", ".join(RECIPE_SITES.keys())
        return f"錯誤：未知網站 '{site}'。可用網站：{available}"
    
    site_info = RECIPE_SITES[site]
    
    # 構建搜尋 URL
    search_url = site_info["search_url"].format(query=query.replace(" ", "+"))
    
    # 設定 MCP server 參數
    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-fetch", "--ignore-robots-txt"]
    )
    
    # 連接並抓取搜尋結果
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            result = await session.call_tool(
                "fetch",
                arguments={
                    "url": search_url,
                    "max_length": 8000,
                    "raw": False  # 使用 Markdown 更易讀
                }
            )
            
            # 提取內容
            content = ""
            if result.content and len(result.content) > 0:
                text_content = result.content[0]
                if hasattr(text_content, 'text'):
                    content = text_content.text
    
    # 返回內容讓 LLM 解析
    return f"""已抓取 {site_info['name']} 的搜尋結果：
搜尋關鍵字：{query}
搜尋 URL：{search_url}

搜尋結果內容：
{content}

請從上述內容中提取最相關的 {max_results} 個食譜，包含：
1. 食譜標題
2. 食譜連結（如果有）
3. 簡短描述（如果有）
"""


# 導出工具列表
web_tools = [
    fetch_webpage,
    search_recipes,
]
