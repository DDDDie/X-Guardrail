import copy
import inspect
import traceback
from typing import Annotated, get_origin
from types import GenericAlias
from pprint import pformat

# 工具存储
_TOOL_HOOKS = {}
_TOOL_DESCRIPTIONS = {}


def register_tool(func: callable):
    """
    装饰器：注册一个工具函数
    """
    tool_name = func.__name__
    tool_description = inspect.getdoc(func).strip() if func.__doc__ else ""
    python_params = inspect.signature(func).parameters
    tool_params = []

    for name, param in python_params.items():
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"参数 `{name}` 缺少类型注解")

        if get_origin(annotation) != Annotated:
            raise TypeError(f"参数 `{name}` 的注解必须是 typing.Annotated")

        typ, (description, required) = annotation.__origin__, annotation.__metadata__
        typ: str = str(typ) if isinstance(typ, GenericAlias) else typ.__name__
        if not isinstance(description, str):
            raise TypeError(f"参数 `{name}` 的描述必须是字符串")
        if not isinstance(required, bool):
            raise TypeError(f"参数 `{name}` 的 required 必须是 bool")

        tool_params.append({
            "name": name,
            "description": description,
            "type": typ,
            "required": required
        })

    tool_def = {
        "name": tool_name,
        "description": tool_description,
        "parameters": {
            "type": "object",
            "properties": {p["name"]: {"type": p["type"], "description": p["description"]} for p in tool_params},
            "required": [p["name"] for p in tool_params if p["required"]]
        }
    }

    print("[registered tool] " + pformat(tool_def))
    _TOOL_HOOKS[tool_name] = func
    _TOOL_DESCRIPTIONS[tool_name] = tool_def
    return func


def dispatch_tool(tool_name: str, tool_params: dict) -> str:
    if tool_name not in _TOOL_HOOKS:
        return f"❌ 工具 `{tool_name}` 未找到"
    tool_call = _TOOL_HOOKS[tool_name]
    try:
        ret = tool_call(**tool_params)
    except Exception:
        ret = traceback.format_exc()
    return str(ret)


def get_tools() -> dict:
    return copy.deepcopy(_TOOL_DESCRIPTIONS)


# ========== 示例工具 =============
@register_tool
def get_weather(
    city: Annotated[str, "要查询的城市名", True],
) -> str:
    """
    获取城市的当前天气信息
    """
    import requests
    try:
        resp = requests.get(f"https://wttr.in/{city}?format=3")
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        return f"获取天气失败: {e}"


@register_tool
def add_numbers(
    a: Annotated[int, "加数 a", True],
    b: Annotated[int, "加数 b", True],
) -> int:
    """
    两数相加
    """
    return a + b


# ========== 模拟 LLM 调用工具 =============
def llm_agent(prompt: str):
    """
    一个简单的 LLM 调用模拟器：
    根据 prompt 决定调用哪个工具
    """
    if "天气" in prompt:
        return {"tool_name": "get_weather", "tool_params": {"city": "Beijing"}}
    if "加法" in prompt:
        return {"tool_name": "add_numbers", "tool_params": {"a": 7, "b": 8}}
    return None


if __name__ == "__main__":
    # 模拟 LLM 输出
    task = llm_agent("帮我查一下北京的天气")
    if task:
        result = dispatch_tool(task["tool_name"], task["tool_params"])
        print("调用结果:", result)

    task = llm_agent("帮我做个加法 7+8")
    if task:
        result = dispatch_tool(task["tool_name"], task["tool_params"])
        print("调用结果:", result)
