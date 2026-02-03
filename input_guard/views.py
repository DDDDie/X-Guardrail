import json
import logging
import re
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .authentication import TokenAuthentication
from guardrails.errors import *

import hashlib
import time
import random
import importlib

class InputEvaluatoreView(APIView):
    # authentication_classes = [TokenAuthentication]
    # permission_classes = [IsAuthenticated]
    
    def post(self, request, *args, **kwargs):
        post_data = request.data
        timestamp = int(time.time() * 1000000)
        random_number = random.randint(1000000, 9999999)
        randon_string = "_" + str(random_number)
        data_to_hash = f"{timestamp}_{request.data}" + randon_string
        task_id = hashlib.sha3_256(data_to_hash.encode()).hexdigest()
        
        ##################################################################
        def evaluacte_content_task(data) -> str:
            try:
                content = data.get("content")
                extra_validators = data.get("Extra-validator", {})
                # 从配置字典中提取组件配置信息
                components = []

                for key, value in extra_validators.items():
                    try:
                        # 动态导入组件
                        module = importlib.import_module("input_guard.components")
                        component_class = getattr(module, key)
                        
                        # 检查并过滤无效参数
                        import inspect
                        component_sign = inspect.signature(component_class.__init__)
                        component_args = component_sign.parameters.keys()
                        valid_params = {k: v for k, v in value.items() if (k in component_args)}
                        
                        # 添加组件实例到列表
                        components.append((component_class, valid_params))
                    except (ImportError, AttributeError, TypeError) as e:
                        print(f"Error loading component {key}: {e}")
                        logging.exception(e)
                        raise e
            except Exception as e:
                raise e
            
            def execute_guard_validate(component_class, content, **args):
                try:
                    component = component_class(**args)
                    res = component(content)
                    print(res)
                    return res
                except Exception as e:
                    raise e
            
            result = content
            for component in components:
                try:
                    result = execute_guard_validate(component[0], result, **component[1])
                except ValidationError as ve:
                    content = str(ve)
                    def remove_validation_error(text):
                        # 查找 "Validation failed for field with errors: " 的位置
                        marker = "Validation failed for field with errors: "
                        
                        # 如果文本包含这个字符串
                        if marker in text:
                            # 截取并返回从该标记之后的部分
                            return text.split(marker, 1)[1]
                        return text  # 如果没有找到该标记，直接返回原始文本
                    cleaned_result = remove_validation_error(content)
                    def fix_json_format(text):
                        # 替换单引号为双引号，注意这里只是简单的替换，可能需要更复杂的规则
                        text = text.replace("'", '"')
                        text = text.replace('"{', '{').replace('}"', '}')
                        return text
                    def safe_json_loads(text):
                        try:
                            # 修复格式问题后加载 JSON
                            fixed_text = fix_json_format(text)
                            result = json.loads(fixed_text)
                            return result
                        except json.JSONDecodeError as e:
                            raise e
                    ve_dict = safe_json_loads(cleaned_result)
                    result = ve_dict["refactor_content"]
                except Exception as e:
                    raise e
            return result
        
        try:
            results = evaluacte_content_task(post_data)
            return Response({'status': 'success', 'message':results})
        except Exception as e:
            logging.exception(e)
            return Response({
                "code":500,
                "status":"error",
                "failCode": None,
                "failMessage": str(e)
            }, status=500)
        ##################################################################

