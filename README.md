<img width="322" height="226" alt="image" src="https://github.com/user-attachments/assets/9804c872-7934-4bf9-99dd-33dbf29bcf56" /><div align="center">

<h1>X-Guardrail 大模型安全护栏</h1>

<hr>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/guardrails-ai)

</div>

# 什么是大模型安全护栏?

大模型安全护栏是一个基于Django的python防御框架，它可以帮助开源大模型在生产环境中快速、低成本、便捷的部署可信AI应用。他有具备以下几点优势：
1. 大模型安全护栏分别提供了输入护栏、输出护栏，支持全方位保护开源大模型。
2. 基于组件化的设计，大模型安全护栏可以模块化定制个性化防御护栏。

<div align="center">
    <img src="./flowchart.png" alt="X-Guardrail流程图">
</div>

# 配置要求

```bash
mysql == 8.0 (可选，默认使用SQLite)
```

# 部署

下载本项目，并进入项目根目录

```bash
git clone <repository_url>
cd X-guardrail-v1/LLMGuardrail
```

首先安装项目所需依赖

```python
pip install -r requirements.txt
```

修改`LLMGuardrail/LLMGuardrail/settings.py`文件，配置数据库（可选，默认使用SQLite）

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'LLMGuardrail', # 数据库名称
        'USER': '账号',
        'PASSWORD': '密码',
        'HOST': 'Mysql服务器地址',
        'PORT': 'Mysql服务器端口号',
    }
}
```

# 快速开始

## 同步数据库表

本项目使用数据库保存用户信息、用户配置等。

```bash
cd LLMGuardrail
python3 manage.py migrate
python3 manage.py makemigrations user_management
python3 manage.py migrate user_management
```

## 启动服务器

```bash
python3 manage.py runserver 0.0.0.0:8000
```

## 创建身份验证Token

HTTP请求

```bash
curl -X POST \
  http://127.0.0.1:8000/user_management/api/token/ \
  -u user_name:password \
  -H "Content-Type: application/json"
```

返回值

```json
{
    "token":"39f5bf0205bf3e14dc05a3d1601e465078940a30"
}
```

后续将使用此token进行身份验证，token有效期为24小时。

## 输入护栏API

HTTP请求

```bash
curl -X POST \
  http://127.0.0.1:8000/input_guard/input_guardrail/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Token 39f5bf0205bf3e14dc05a3d1601e465078940a30" \
  -d '{
    "content": "你好，ChatGPT。",
    "Extra-validator": {
      "DetectPII": {
        "pii_entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"]
      },
      "CustomMask": {
        "mask_list": ["敏感词"],
        "replace_str": "<MASK>"
      }
    }
  }'
```

返回值

```json
{
    "status":"success",
    "message":"处理后的内容"
}
```

## 输出护栏API

HTTP请求

```bash
curl -X POST \
  http://127.0.0.1:8000/output_guard/output_evaluation/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Token 39f5bf0205bf3e14dc05a3d1601e465078940a30" \
  -d '{
    "content": "你好",
    "Extra-validator": {
      "DetectPII": {
        "pii_entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"]
      }
    }
  }'
```

返回值

1. 测试通过：

    ```json
    {
        "code": 200,
        "status":"success"
    }
    ```

2. 测试失败：

    ```json
    {
        "code": 200,
        "status":"fail",
        "failCode": 32,
        "failMessage": "文本违反了主要风险中文本具有疑似敏感信息"
    }   
    ```

# v1.0 版本特性

- 基础Django框架
- 用户管理和Token认证
- 输入护栏（支持DetectPII和CustomMask组件）
- 输出护栏（支持DetectPII组件）
- 组件化设计，易于扩展

# 文件架构

```bash
└─LLMGuardrail
    ├─input_guard          # 输入护栏模块
    │  └─components        # 输入护栏组件
    │     ├─DetectPII.py  # PII检测组件
    │     └─CustomMask.py # 自定义屏蔽词组件
    ├─output_guard         # 输出护栏模块
    │  └─components        # 输出护栏组件
    │     └─DetectPII.py   # PII检测组件
    ├─user_management      # 用户管理模块
    └─LLMGuardrail         # Django项目配置
```
