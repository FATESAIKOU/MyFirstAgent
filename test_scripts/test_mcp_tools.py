"""
測試 web tools
"""
import asyncio
import sys
from pathlib import Path

# 添加專案根目錄到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.web import fetch_webpage, search_recipes


async def test_fetch_tool():
    """測試 fetch_webpage tool"""
    print("="*60)
    print("測試 1: fetch_webpage Tool")
    print("="*60)
    
    print("\n🧪 使用 tool 抓取網頁...")
    result = await fetch_webpage.ainvoke({
        "url": "https://example.com",
        "max_length": 1000,
        "use_markdown": True
    })
    print(f"✅ 成功！長度: {len(result)} 字元")
    print(f"內容預覽:\n{result[:200]}...\n")


async def test_search_tool():
    """測試 search_recipes tool"""
    print("="*60)
    print("測試 2: search_recipes Tool")
    print("="*60)
    
    print("\n🧪 搜尋食譜: chicken salad...")
    result = await search_recipes.ainvoke({
        "query": "chicken salad",
        "site": "allrecipes",
        "max_results": 3
    })
    print(f"✅ 成功！結果長度: {len(result)} 字元")
    print(f"\n搜尋結果預覽（前 500 字元）:")
    print(result[:500])
    print("...\n")


async def main():
    """執行所有測試"""
    try:
        await test_fetch_tool()
        await test_search_tool()
        
        print("="*60)
        print("✅ 所有測試通過！")
        print("="*60)
    except Exception as e:
        print(f"\n❌ 測試失敗：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
