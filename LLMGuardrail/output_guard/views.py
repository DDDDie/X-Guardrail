import json
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .authentication import TokenAuthentication
from guardrails import Guard
from guardrails.errors import *

import hashlib
import time
import random
import importlib
import concurrent.futures
import re

class OutputEvaluatoreView(APIView):
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
        def evaluacte_content_task(data):
            try:
                content = data.get("content")
                extra_validators = data.get("Extra-validator", {})
                # 从配置字典中提取组件配置信息
                components = []

                for key, value in extra_validators.items():
                    try:
                        # 动态导入组件
                        module = importlib.import_module("output_guard.components")
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
                return Response({
                    "code":500,
                    "status":"error",
                    "failCode": None,
                    "failMessage": str(e)
                }, status=500)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                # 提交所有Guard对象的validate任务
                # 线程池执行的函数
                def execute_guard_validate(guard, content, **args):
                    try:
                        guard.validate(content, **args)
                    except Exception as e:
                        raise e
                
                for conponent in components:
                    future = executor.submit(execute_guard_validate, Guard().use(conponent[0], **conponent[1]), content)
                    futures.append(future)
                
                # 等待所有任务完成，并处理异常
                try:
                    for future in concurrent.futures.as_completed(futures):
                        print(future.result())
                        future.result()  # 如果有异常，将会在这里抛出
                except Exception as e:
                    # 处理验证失败
                    for future in futures:
                        future.cancel()
                    raise e  # 或者自定义处理
        
        try:
            evaluacte_content_task(post_data)
            return Response({
                    "code":200,
                    "status":"success",
                    "failCode": None,
                    "failMessage": None
                }, status=200)
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
            try:
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
                        print(f"Error decoding JSON: {e}")
                        return None
                result = safe_json_loads(cleaned_result)
                if result:
                    return Response({
                        "code":200,
                        "status":"fail",
                        "failCode": result.get("category"),
                        "failMessage": f"文本违反了主要风险中{result.get('failure', '未知错误')}",
                    }, status=200)
                else:
                    return Response({
                        "code":200,
                        "status":"fail",
                        "failCode": None,
                        "failMessage": cleaned_result
                    }, status=200)
            except Exception as e:
                logging.exception(e)
                return Response({
                    "code":500,
                    "status":"error",
                    "failCode": None,
                    "failMessage": str(e)
                }, status=500)
        except Exception as e:
            logging.exception(e)
            return Response({
                "code":500,
                "status":"error",
                "failCode": None,
                "failMessage": str(e)
            }, status=500)
        ##################################################################

