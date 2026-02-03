import re
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List, Union

from .utils.AhoCorasick import ac_text_filter
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

# 获取当前文件的绝对路径并解析符号链接
current_path = Path(__file__).resolve()
current_dir = current_path.parent

# 构造目标文件路径（使用路径拼接运算符/）
MASK_WORDS_FILE = current_dir / "database" / "mask_words.json"

@register_validator(name="STAIR/CustomMask", data_type="string")
class CustomMask(Validator):
    """优化版屏蔽词验证器，支持海量词库和短语匹配"""

    def __init__(
        self,
        mask_list: List[str],  # 改为直接接收字符串列表
        replace_str: str = "<MASK>",
        on_fail: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(on_fail, **kwargs)
        # 转换为原始代码需要的字典格式
        self.mask_list = mask_list
        self.replace_str = replace_str
    
    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        try:
            masked_content, masked_report = ac_text_filter(value, str(MASK_WORDS_FILE), self.mask_list, self.replace_str)
            if not (masked_content == value):
                result = ""
                for word, info in masked_report.items():
                    result += "\n"
                    result += f"{word}: 出现{info['count']}次，原位置：{info['positions']}"
                return FailResult(
                    metadata=metadata,
                    error_message=json.dumps({
                        "result": result,
                        "category": 32,
                        "failure": "文本中含有屏蔽词",
                        "refactor_content": masked_content,
                    }, ensure_ascii=False),
                    fix_value=masked_content
                )
            return PassResult(metadata=metadata)
        except Exception as e:
            raise e

