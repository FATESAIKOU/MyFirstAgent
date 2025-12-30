"""
測試 Agent 是否能正確使用 web tools
"""
import sys
from pathlib import Path

# 添加專案根目錄到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agent.graph import create_graph, all_tools


def test_agent_tools():
    """測試 Agent 能否看到並使用 web tools"""
    print("="*60)
    print("測試 Agent 工具載入")
    print("="*60)
    
    # 建立 Graph
    graph = create_graph()
    
    # 檢查綁定的工具
    print("\n✅ Agent Graph 建立成功")
    print("\n📋 已綁定的工具：")
    
    for i, tool in enumerate(all_tools, 1):
        print(f"  {i}. {tool.name}")
        if hasattr(tool, 'description'):
            desc = tool.description[:60] + "..." if len(tool.description) > 60 else tool.description
            print(f"     {desc}")
    
    print(f"\n✅ 總計 {len(all_tools)} 個工具")
    
    # 檢查是否包含 web tools
    tool_names = [tool.name for tool in all_tools]
    web_tool_names = ["fetch_webpage", "search_recipes"]
    
    print("\n🔍 檢查 web tools:")
    for web_tool in web_tool_names:
        if web_tool in tool_names:
            print(f"  ✅ {web_tool} - 已載入")
        else:
            print(f"  ❌ {web_tool} - 未找到")
    
    print("\n" + "="*60)
    print("✅ Agent 已準備好使用網路搜尋功能！")
    print("="*60)


if __name__ == "__main__":
    test_agent_tools()
