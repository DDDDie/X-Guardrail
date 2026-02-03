import asyncio
import time
import traceback
import gc
from typing import Dict, Any
import datetime
import hashlib
import os
import random
import torch
import json
import uvicorn
import asyncio

torch.random.manual_seed(0)
from pydantic import BaseModel, Field, validator
from typing import List, Literal, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
# from sse_starlette.sse import ServerSentEvent, EventSourceResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 全局字典保存已部署的模型信息
# 格式示例：
# deployed_models = {
#     "model_a": {
#         "instance": <model_instance>,
#         "gpu_id": 0,
#         "last_used": timestamp
#     }
# }
deployed_methods: Dict[str, Any] = {}
deployed_models: Dict[str, Dict[str, Any]] = {}
model_locks: Dict[str, asyncio.Lock] = {} # 针对每个模型的锁
MODEL_IDLE_TIMEOUT = 36000 # 定义模型空闲超时时间（单位秒），例如 1 小时
REQUEST_BATCH_SIZE = 4 # 定义请求批处理大小
REQUEST_BATCH_WAITING_TIME = 100 # 定义请求批处理等待时间（单位毫秒），例如 100 毫秒


def get_model_info(key: str, file_path: str = "./model_info.json"):
    with open(file_path, "r") as f:
        model_info = json.load(f)
    return model_info.get(key)

def extract_r1_qwen_conversation(text, pattern = r"<\｜User\｜>(.*?)<\｜Assistant\｜><think>(.*?)</think>\s*(.*)"):
    import re
    match = re.search(pattern, text, re.DOTALL)
    if match:
        user_input = match.group(1).strip()
        assistant_thinking = match.group(2).strip()
        assistant_output = match.group(3).strip()

        result = {
            "User": user_input,
            "Assistant_thinking": assistant_thinking,
            "Assistant_output": assistant_output
        }
        return result
    else:
        raise ValueError("Invalid conversation format")
    
def extract_r1_conversation(text, pattern = r"(.*?)</think>\s*(.*)"):
    import re
    match = re.search(pattern, text, re.DOTALL)
    if match:
        assistant_thinking = match.group(1).strip()
        assistant_output = match.group(2).strip()

        result = {
            "Assistant_thinking": assistant_thinking,
            "Assistant_output": assistant_output
        }
        return result
    else:
        raise ValueError("Invalid conversation format")

##########################################################################
#                         定义gpu算法推理方法                             #
##########################################################################
class llm_DeepSeek_R1_Distill_Qwen_pipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def _sync_generate_batch(self, messages_batch, **kwargs):
        device = self.model.device
        
        input_tensor = self.tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)  

        with torch.inference_mode():
            outputs = self.model.generate(
                input_tensor,
                max_new_tokens= kwargs.get("max_new_tokens", 1024),
                temperature= kwargs.get("temperature", 0.7)
            )
        generated_text_list = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        result = []
        
        for generated_text in generated_text_list:
            try:
                generated_text = extract_r1_qwen_conversation(generated_text)["Assistant_output"]
            except:
                generated_text = generated_text
            result.append({
                'generated_text': generated_text
            })
        
        return result
    
    def _sync_generate(self, messages, **kwargs):
        device = self.model.device
        
        input_tensor = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)  
        
        with torch.inference_mode():
            outputs = self.model.generate(
                input_tensor,
                max_new_tokens= kwargs.get("max_new_tokens", 1024),
                temperature= kwargs.get("temperature", 0.7),
                top_p = kwargs.get("top_p", 0.9)
            )    
        
        response = outputs[0][input_tensor.shape[1]:]
        
        response = self.tokenizer.decode(
                response,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        try:
            generated_text = extract_r1_qwen_conversation(response)["Assistant_output"]
        except:
            generated_text = response
        return [{
            'generated_text': generated_text
        }]
        
    async def __call__(self, messages, **generation_args):
        loop = asyncio.get_event_loop()
        # 使用 ThreadPoolExecutor 执行同步函数
        result = await loop.run_in_executor(None, self._sync_generate, messages)
        return result
                
class llm_default_pipeline:   
    def __init__(self, pipeline, **kwargs):
        self.pipe = pipeline
    
    def _sync_generate_batch(self, messages_batch, **kwargs):
        result = []
        with torch.inference_mode():
            kwargs["do_sample"] = True
            kwargs["return_full_text"] = False
            outputs = self.pipe(messages_batch, 
                                **kwargs)
        for output in  outputs:
            result.append({'generated_text': output[-1]['generated_text']})
        return result
    
    # 同步生成
    def _sync_generate(self, messages, **kwargs):
        with torch.inference_mode():
            outputs = self.pipe(messages, **kwargs)
        return outputs
    
    # 异步推理
    async def __call__(self, messages, **generation_args):
        loop = asyncio.get_event_loop()
        # 使用 ThreadPoolExecutor 执行同步函数
        result = await loop.run_in_executor(None, self._sync_generate, messages)
        return result

class llm_deepseek_moe_16b_chat_custom_pipeline:
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        
    def _sync_generate_batch(self, messages_batch, **kwargs):
        print("Request batch size:" + str(len(messages_batch)))
        inputs = []
        for messages in messages_batch:
            text = messages[-1]["content"]
            inputs.append(self.tokenizer(
                text, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ))
        
        batched_inputs = {
            'input_ids': torch.cat([i['input_ids'] for i in inputs], dim=0).to("cuda"),
            'attention_mask': torch.cat([i['attention_mask'] for i in inputs], dim=0).to("cuda")
        }
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **batched_inputs,
                max_new_tokens=32
            )
            
        responses = []
        for i, output in enumerate(outputs):
            decoded = self.tokenizer.decode(
                output[len(inputs[i]['input_ids'][0]):],
                skip_special_tokens=True
            )
            responses.append({'generated_text': decoded})
            
        return responses

class llm_ktransformers_custom_pipeline:
    def __init__(self, model, tokenizer, config, **kwargs):
        from transformers import (
            GenerationConfig,
        )
        try:
            model.generation_config = GenerationConfig.from_pretrained(kwargs["model_path"])
        except Exception as e:
            print(f"generation config can't auto create, make default. Message: {e}")
            gen_config = GenerationConfig(
                temperature=0.6,
                top_p=0.95,
                do_sample=True
            )
            model.generation_config = gen_config
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.args = kwargs

        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
        model.eval()
        
    def _sync_generate_batch(self, messages_batch, **kwargs):
        from ktransformers.server.config.config import Config
        from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled
        from ktransformers.util.utils import prefill_and_generate, get_compute_capability
        from ktransformers.util.vendors import device_manager, GPUVendor
        
        # messages_batch = [[
        #     {"role": "user", "content": "你好,什么是大语言模型？"}
        # ],
        # [
        #     {"role": "user", "content": "你好,什么是大语言模型？"}
        # ]]
        
        force_think = self.args["force_think"]
        mode = self.args["mode"]
        max_new_tokens = int(self.args["max_new_tokens"] / 10)
        chunk_prefill_size = int(self.args["chunk_prefill_size"] / 40)
        use_cuda_graph = self.args["use_cuda_graph"]

        response = []
        for msg in messages_batch:
            input_tensor = self.tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, return_tensors="pt"
            )
            if force_think:
                token_thinks = torch.tensor([self.tokenizer.encode("<think>\\n",add_special_tokens=False)],device=input_tensor.device)
                input_tensor = torch.cat(
                    [input_tensor, token_thinks], dim=1
                )
            if mode == 'long_context':
                assert Config().long_context_config['max_seq_len'] > input_tensor.shape[1] + max_new_tokens, \
                "please change max_seq_len in  ~/.ktransformers/config.yaml"
            try:
                if (self.config.architectures[0] == "DeepseekV2ForCausalLM" or self.config.architectures[0] == "DeepseekV3ForCausalLM") and flashinfer_enabled and get_compute_capability() >= 8 and device_manager.gpu_vendor == GPUVendor.NVIDIA:
                    generated = prefill_and_generate(
                        self.model, self.tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_prefill_size = chunk_prefill_size,
                        use_flashinfer_mla = True, num_heads = self.config.num_attention_heads, head_dim_ckv = self.config.kv_lora_rank, head_dim_kpe = self.config.qk_rope_head_dim, q_head_dim = self.config.qk_rope_head_dim + self.config.qk_nope_head_dim
                    )
                else:
                    generated = prefill_and_generate(
                        self.model, self.tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_prefill_size = chunk_prefill_size,
                    )
                text = self.tokenizer.decode(generated)
                
                try:
                    generated_text = extract_r1_qwen_conversation(text)["Assistant_output"]
                except:
                    generated_text = text
                response.append({'generated_text': generated_text})
            except Exception as e:
                    response.append({'generated_text': str(e)})
        return response
    
    def _sync_generate(self, messages, **kwargs):
        from ktransformers.server.config.config import Config
        from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled
        from ktransformers.util.utils import prefill_and_generate, get_compute_capability
        from ktransformers.util.vendors import device_manager, GPUVendor

        force_think = self.args["force_think"]
        mode = self.args["mode"]
        max_new_tokens = self.args["max_new_tokens"]
        chunk_prefill_size = self.args["chunk_prefill_size"]
        use_cuda_graph = self.args["use_cuda_graph"]
        
        input_tensor = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if force_think:
            token_thinks = torch.tensor([self.tokenizer.encode("<think>\\n",add_special_tokens=False)],device=input_tensor.device)
            input_tensor = torch.cat(
                [input_tensor, token_thinks], dim=1
            )
        if mode == 'long_context':
            assert Config().long_context_config['max_seq_len'] > input_tensor.shape[1] + max_new_tokens, \
            "please change max_seq_len in  ~/.ktransformers/config.yaml"
        
        if (self.config.architectures[0] == "DeepseekV2ForCausalLM" or self.config.architectures[0] == "DeepseekV3ForCausalLM") and flashinfer_enabled and get_compute_capability() >= 8 and device_manager.gpu_vendor == GPUVendor.NVIDIA:
            generated = prefill_and_generate(
                self.model, self.tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_prefill_size = chunk_prefill_size,
                use_flashinfer_mla = True, num_heads = self.config.num_attention_heads, head_dim_ckv = self.config.kv_lora_rank, head_dim_kpe = self.config.qk_rope_head_dim, q_head_dim = self.config.qk_rope_head_dim + self.config.qk_nope_head_dim
            )
        else:
            generated = prefill_and_generate(
                self.model, self.tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, chunk_prefill_size = chunk_prefill_size,
            )
        
        text = ""
        for token in generated:
            text += token.item()
            
        try:
            generated_text = extract_r1_qwen_conversation(text)["Assistant_output"]
        except:
            generated_text = text
        return [{'generated_text': generated_text}]

    async def __call__(self, messages, **generation_args):
        loop = asyncio.get_event_loop()
        # 使用 ThreadPoolExecutor 执行同步函数
        result = await loop.run_in_executor(None, self._sync_generate, messages)
        return result
        
#========================================================================#
## 组件模型
class MD_Judge_v1_custom_pipeline:
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        
    def _sync_generate_batch(self, messages_batch, **kwargs):
        print("Request batch size:" + str(len(messages_batch)))
        inputs = []
        for messages in messages_batch:
            text = messages[-1]["content"]
            inputs.append(self.tokenizer(
                text, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ))
        
        batched_inputs = {
            'input_ids': torch.cat([i['input_ids'] for i in inputs], dim=0).to("cuda"),
            'attention_mask': torch.cat([i['attention_mask'] for i in inputs], dim=0).to("cuda")
        }
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **batched_inputs,
                max_new_tokens=32
            )
            
        responses = []
        for i, output in enumerate(outputs):
            decoded = self.tokenizer.decode(
                output[len(inputs[i]['input_ids'][0]):],
                skip_special_tokens=True
            )
            responses.append({'generated_text': decoded})
            
        return responses
        
    def _sync_generate(self, messages, **kwargs):
        inputs = self.tokenizer(
            messages[-1]["content"],
            return_tensors="pt",
            add_special_tokens=True
        ).to("cuda")
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=32)
        resp = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return [{'generated_text': resp[0].removeprefix(messages[-1]["content"])}]

    async def __call__(self, messages, **generation_args):
        loop = asyncio.get_event_loop()
        # 使用 ThreadPoolExecutor 执行同步函数
        result = await loop.run_in_executor(None, self._sync_generate, messages)
        return result

class component_nlp_fluency_bert_pipeline:
    from utils.nlp_fluency import MaskedBert
    def __init__(self, model: MaskedBert):
        self.model = model
    
    def __call__(self, value: str, verbose=False, temperature=1.0, batch_size=100, check_length= 10, **kwds):
        ppl = self.model.perplexity(
            x=" ".join(value[-check_length:]),   # 每个字空格隔开或者输入一个list
            verbose= verbose,     # 是否显示详细的probability，default=False
            temperature= temperature,   # softmax的温度调节，default=1
            batch_size= batch_size,    # 推理时的batch size，可根据cpu或gpu而定，default=100
        )
        return ppl
##########################################################################
#                         定义gpu算法载入方法                             #
##########################################################################
def llm_default_mothod(model_name:str, **config):
    from transformers import BitsAndBytesConfig
    model_info = get_model_info(model_name)
    if model_info.get("accalerate") is None:
        try:
            if model_info.get("load_in_4bit") is None or model_info.get("load_in_4bit") == False:
                quantization_config = None
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit= True,
                    bnb_4bit_quant_type= "nf4",
                    bnb_4bit_use_double_quant= True,
                    bnb_4bit_compute_dtype=torch.bfloat16
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_info["model_path"],
                trust_remote_code=True,
                device_map='auto',
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config
            )

            tokenizer = AutoTokenizer.from_pretrained(model_info["model_path"], trust_remote_code=True)
            
            return llm_default_pipeline(pipeline("text-generation", model=model, tokenizer=tokenizer))
            
        except Exception as e:
            print(f"Error occurred during model initialization for {model_name}: {e}")
            raise e
    elif model_info.get("accalerate") == 'deepspeed':
        import deepspeed
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_info["model_path"],
                device_map={'': 4},
                torch_dtype="auto",
                trust_remote_code=True
            )

            model = deepspeed.init_inference(
                model,
                dtype=torch.float16,
                tensor_parallel={'tp_size': 1},
                replace_with_kernel_inject=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_info["model_path"], use_fast=False)
            def custom_pipeline(model, tokenizer, **kwargs):
                def generate(input_text, **kwargs):
                    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.module.device)
                    output = model.generate(input_ids, **kwargs)
                    return tokenizer.decode(output[0], skip_special_tokens=True)
                return generate
            pipe = custom_pipeline(model, tokenizer)
        except Exception as e:
            print(f"Error occurred during model initialization for {model_name}: {e}")
            raise e
    return pipe

def llm_deepseek_moe_16b_chat_transformers(model_name: str, **config):
    model_info = get_model_info(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_info["model_path"],
            device_map="auto" ,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,  
            load_in_8bit=config.get('load_in_8bit', True),
            use_flash_attention_2=False  
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_info["model_path"],
            use_fast=True,
            trust_remote_code=True
        )
        return llm_deepseek_moe_16b_chat_custom_pipeline(model,tokenizer)
    except Exception as e:
        print(f"Error occurred during model initialization for {model_name}: {e}")
        raise e

def llm_DeepSeek_R1_Distill_Qwen(model_name: str, **config):
    from transformers import BitsAndBytesConfig
    model_info = get_model_info(model_name)
    try:
        if model_info.get("load_in_4bit") is None or model_info.get("load_in_4bit") == False:
            quantization_config = None
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit= True,
                bnb_4bit_quant_type= "nf4",
                bnb_4bit_use_double_quant= True,
                bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        device_map = model_info.get("device_map", "auto")
        
        if model_name == "llm-DeepSeek-R1-Distill-Qwen-7B":
            model_path = model_info.get("model_path", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        elif model_name == "llm-DeepSeek-R1-Distill-Qwen-1.5B":
            model_path = model_info.get("model_path", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        elif model_name == "llm-Qwen2.5-1.5B-Instruct":
            model_path = model_info.get("model_path", "Qwen/Qwen2.5-1.5B-Instruct")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map, 
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True,
            pad_token='<|endoftext|>'  
        )
        
        return llm_DeepSeek_R1_Distill_Qwen_pipeline(model, tokenizer)
        
    except Exception as e:
        print(f"Error loading Qwen2-7B model: {e}")
        raise

def llm_ktransformers(
        model_name: str,
        max_new_tokens: int = 4096,
        cpu_infer: int = 24,
        use_cuda_graph: bool = True,
        optimize_config_path: str = None,
        mode: str = "normal",
        force_think: bool = False,
        chunk_prefill_size: int = 32768,
        **config
    ):
    from ktransformers.optimize.optimize import optimize_and_load_gguf
    from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
    from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
    from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
    from ktransformers.models.modeling_llama import LlamaForCausalLM
    from ktransformers.models.modeling_mixtral import MixtralForCausalLM
    from ktransformers.server.config.config import Config
    from transformers import (
        AutoTokenizer,
        AutoConfig,
        AutoModelForCausalLM,
    )
    
    
    custom_models = {
        "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
        "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
        "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
        "LlamaForCausalLM": LlamaForCausalLM,
        "MixtralForCausalLM": MixtralForCausalLM,
    }

    ktransformer_rules_dir = (
        os.path.dirname(os.path.abspath(__file__)) + "/optimize/optimize_rules/"
    )
    
    default_optimize_rules = {
        "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
        "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat.yaml",
        "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
        "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
        "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml",
    }

    model_info = get_model_info(model_name)
    model_path = model_info["model_path"]
    torch.set_grad_enabled(False)

    if cpu_infer is None:
        cpu_infer = Config().cpu_infer
    else:
        Config().cpu_infer = cpu_infer

    # try:
    #     force_think = model_info["force_think"]
    # except KeyError:
    #     force_think = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    if mode == 'long_context':
        assert config.architectures[0] == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            if (
                "Qwen2Moe" in config.architectures[0]
            ):  # Qwen2Moe must use flash_attention_2 to avoid overflow.
                config._attn_implementation = "eager"
            if "Llama" in config.architectures[0]:
                config._attn_implementation = "eager"
            if "Mixtral" in config.architectures[0]:
                config._attn_implementation = "eager"

            model = custom_models[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="eager"
            )

    if optimize_config_path is None:
        if config.architectures[0] in default_optimize_rules:
            print("using default_optimize_rule for", config.architectures[0])
            optimize_config_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_config_path = input(
                "please input the path of your rule file(yaml file containing optimize rules):"
            )
    gguf_path = model_info["model_gguf"]
    optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)
    
    return llm_ktransformers_custom_pipeline(
        model,
        tokenizer,
        config,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        cpu_infer=cpu_infer,
        use_cuda_graph=use_cuda_graph,
        mode=mode,
        force_think=force_think,
        chunk_prefill_size=chunk_prefill_size
    )

#========================================================================#
## 组件模型
def llm_MD_Judge_v1(model_name:str, **config):
    from transformers import BitsAndBytesConfig
    model_info = get_model_info(model_name)
    try:
        if model_info.get("load_in_4bit") is None or model_info.get("load_in_4bit") == False:
            quantization_config = None
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit= True,
                bnb_4bit_quant_type= "nf4",
                bnb_4bit_use_double_quant= True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_info.get("model_path", "OpenSafetyLab/MD-Judge-v0.1"),
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(model_info.get("model_path", "OpenSafetyLab/MD-Judge-v0.1"), trust_remote_code=True) 
        return MD_Judge_v1_custom_pipeline(model, tokenizer)
    except Exception as e:
        print(f"Error occurred during model initialization for {model_name}: {e}")
        raise e

def component_nlp_fluency_bert(model_name:str, **config):
    from utils.nlp_fluency import MaskedBert
    model_info = get_model_info("component-nlp_fluency_bert")
    model = MaskedBert.from_pretrained(model_info["model_path"],
                                       # 使用cpu或者cuda:0，default=cpu
                                       device="cpu",
                                       # 长句做切句处理，段落会被切成最大不超过该变量的句子集，default=50
                                       sentence_length=50)
    return component_nlp_fluency_bert_pipeline(model)

##########################################################################
#                         注册模型载入方法                                #
##########################################################################
deployed_methods["llm-default"] = llm_default_mothod
deployed_methods["llm-MD-Judge-v0.1"] = llm_MD_Judge_v1
deployed_methods["llm-DeepSeek-V2-Lite"] = llm_ktransformers
deployed_methods["llm-DeepSeek-V3"] = llm_ktransformers
deployed_methods["llm-DeepSeek-R1"] = llm_ktransformers
deployed_methods["llm-DeepSeek-R1-Distill-Qwen-7B"] = llm_DeepSeek_R1_Distill_Qwen
deployed_methods["llm-DeepSeek-R1-Distill-Qwen-1.5B"] = llm_DeepSeek_R1_Distill_Qwen
deployed_methods["llm-Qwen2.5-1.5B-Instruct"] = llm_DeepSeek_R1_Distill_Qwen
deployed_methods["component-nlp_fluency_bert"] = component_nlp_fluency_bert

# 模型载入工厂
class ModelFactory:
    def deploy_model(self, model_name:str, **config):
        if model_name == "llm-DeepSeek-R1":  # 新增条件判断
            func = deployed_methods.get("llm-DeepSeek-V2-Lite")
            try:
                model = func(model_name, **config)
                return model
            except Exception as e:
                print(f"Error loading DeepSeek-R1 with GGUF: {e}")
                raise e
        func = deployed_methods.get(model_name)
        if model_name.startswith("llm-"):
            if func is None:
                # 没有就是用默认pipeline导入
                func = deployed_methods.get("llm-default")
            try:
                model = func(model_name, **config)
                return model
            except Exception as e:
                print(f"Error occurred during model initialization for {model_name}: {e}")
                raise e
        elif model_name.startswith("component-"):
            if func is None:
                raise ValueError(f"Invalid component name: {model_name}")
            try:
                model = func(model_name, **config)
                return model
            except Exception as e:
                print(f"Error occurred during model initialization for {model_name}: {e}")
                raise e
        else:
            raise ValueError(f"Invalid model name: {model_name}")
##########################################################################

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://allowed-origin.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False

    @validator('temperature')
    def temperature_range(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('temperature must be between 0 and 1')
        return v

    @validator('top_p')
    def top_p_range(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('top_p must be between 0 and 1')
        return v


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))

class BatchHandler:
    def __init__(self, model_name, batch_size=4, max_wait=0.1):
        self.queue = asyncio.Queue()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.processing_task = asyncio.create_task(self._process_batches())

    async def add_request(self, messages, generation_args):
        future = asyncio.Future()
        await self.queue.put((messages, generation_args, future))
        return await future

    async def _process_batches(self):
        print("Batch handler started")
        while True:
            batch = []
            start_time = time.time()
            
            # 收集批次或等待超时（精确计算剩余时间）
            while len(batch) < self.batch_size:
                try:
                    remaining = self.max_wait - (time.time() - start_time)
                    if remaining <= 0:
                        break
                        
                    # 等待时使用剩余时间作为超时
                    item = await asyncio.wait_for(
                        self.queue.get(), 
                        timeout=remaining
                    )
                    batch.append(item)       
                except asyncio.TimeoutError:
                    break
            
            if batch:           
                await self._process_batch(batch)

    async def _process_batch(self, batch):
        try:
            # 获取模型实例
            pipeline = await deploy_model(self.model_name)
            
            # 准备批量输入
            messages_batch = [item[0] for item in batch]
            args_batch = [item[1] for item in batch]
            
            # 合并生成参数（取第一个请求的参数）
            merged_args = self._merge_generation_args(args_batch)
            
            # 执行批量推理
            loop = asyncio.get_event_loop()
            responses = await loop.run_in_executor(
                self.executor,
                self._run_batch_inference,
                pipeline,
                messages_batch,
                merged_args
            )
            
            # 分发结果
            for (_, _, future), response in zip(batch, responses):
                future.set_result(response)
                
        except Exception as e:
            for (_, _, future) in batch:
                future.set_exception(e)

    def _merge_generation_args(self, args_list):
        # 这里实现参数合并逻辑，示例取第一个参数
        return args_list[0] if args_list else {}

    def _run_batch_inference(self, pipeline, messages_batch, generation_args):
        # 扩展pipeline以支持批量处理
        if hasattr(pipeline, '_sync_generate_batch'):
            return pipeline._sync_generate_batch(messages_batch, **generation_args)
        # 回退到逐个处理
        return [pipeline._sync_generate(msg, **generation_args)[0] for msg in messages_batch]

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="OpenSource_LLM")
    return ModelList(data=[model_card])

##########################################################################
#                              gpu算法                                   #
##########################################################################
batch_handlers = {}

async def get_batch_handler(model_name: str) -> BatchHandler:
    if model_name not in batch_handlers:
        batch_handlers[model_name] = BatchHandler(
            model_name,
            batch_size=REQUEST_BATCH_SIZE,      # 可配置参数
            max_wait=REQUEST_BATCH_WAITING_TIME / 1000      # 100ms等待
        )
    return batch_handlers[model_name]

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if request.messages[-1].role!= "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        query = prev_messages.pop(0).content + query

    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and prev_messages[i + 1].role == "assistant":
                history.append([prev_messages[i].content, prev_messages[i + 1].content])

    def print_request_and_response(request, response, generation_args):
        # 打印请求中的消息
        print("Request Messages:")
        for msg in request.messages:
            print(f"Role: {msg.role}, Content: {msg.content}")

        # 打印生成参数
        print("\nGeneration Arguments:")
        print(json.dumps(generation_args, indent=4))

        # 打印响应生成的文本
        print("\nResponse:")
        print(response)

    messages = []
    for msg in request.messages:
        messages.append({
            "role": msg.role,
            "content": msg.content
        })

    generation_args = {
        "temperature": request.temperature if request.temperature is not None else 0.8,
        "top_p": request.top_p if request.top_p is not None else 0.95,
        "max_new_tokens": request.max_length if request.max_length is not None else 2056,
        "do_sample": True, 
        "return_full_text": False, 
    }

    try:
        # 部署或获取已部署的模型
        handler = await get_batch_handler("llm-" + request.model)
        response = await handler.add_request(messages, generation_args)
        # pipeline = await deploy_model("llm-" + request.model)
        # deployed_models["llm-" + request.model]["last_used"] = time.time()
        # # response = pipeline(messages, **generation_args)[0]['generated_text']
        # task = asyncio.create_task(pipeline(messages))
        # gather_responses = await asyncio.gather(task)
        # response = gather_responses[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {e}")
    
    print_request_and_response(request, response, generation_args)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response['generated_text']),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")

@app.post("/ppl_detector")
async def ppl_detector(request: Request):
    timestamp = int(time.time() * 1000000)
    random_number = random.randint(1000000, 9999999)
    randon_string = "_" + str(random_number)
    data_to_hash = f"{timestamp}_" + randon_string
    task_id = hashlib.sha3_256(data_to_hash.encode()).hexdigest()
    
    json_post = await request.json()
    
    content = json_post.get('content')
    
    ############################################################################     
    # 调用ppl_calculator进行ppl检测
    check_length = 15 if json_post.get('check_length') is None else json_post.get('check_length')
    ppl_detector = await deploy_model("component-nlp_fluency_bert")
    deployed_models["component-nlp_fluency_bert"]["last_used"] = time.time()
    ppl_score = ppl_detector(content, check_length= check_length)
   
    ############################################################################
    
    now = datetime.datetime.now()
    now_time = now.strftime("%Y-%m-%d %H:%M:%S")
    
    response = {
        "ppl_score": float(ppl_score),
        "status": 200,
        "time": now_time
    }
    log = f"[{now_time}] content:{content}, ppl_score:{ppl_score}"
    print(log)
    return response
    
##########################################################################
#                      处理模型加载、卸载等操作                            #
##########################################################################
def load_model(model_name: str, device: torch.device, **config):
    """
    模拟加载模型，可以替换为具体的模型加载逻辑
    """

    try:
        model_factory = ModelFactory()
        model = model_factory.deploy_model(model_name, device=device, **config)
    except Exception as e:
        raise e
    print(f"Model {model_name} loaded on {device}")
    return model

async def deploy_model(model_name: str, gpu_id: int = 0):
    """
    部署模型：如果模型未加载，则加载到指定的 GPU 上；否则直接返回已加载的模型。
    使用锁来防止多个协程同时部署同一模型。
    """
    lock = model_locks.setdefault(model_name, asyncio.Lock())
    async with lock:
        if model_name in deployed_models:
            return deployed_models[model_name]["instance"]

        try:
            
            model_factory = ModelFactory()
            model_instance = model_factory.deploy_model(
                model_name, 
                device_map=f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
            
            deployed_models[model_name] = {
                "instance": model_instance,
                "gpu_id": gpu_id,
                "last_used": time.time()
            }
            return model_instance
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise HTTPException(status_code=500, detail="GPU 内存不足，无法部署模型。")
            else:
                raise HTTPException(status_code=500, detail="部署模型时出错。")
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="未知错误，部署模型失败。")

def unload_model(model_name: str):
    """
    卸载模型：释放该模型占用的资源，不影响其他模型。
    为避免与其它操作冲突，同样需要通过锁来保护卸载过程。
    """
    async def _unload():
        lock = model_locks.setdefault(model_name, asyncio.Lock())
        async with lock:
            if model_name in deployed_models:
                model_info = deployed_models.pop(model_name)
                # 删除模型实例，并调用垃圾回收释放内存
                del model_info["instance"]
                gc.collect()
                # 如果需要，还可以调用 torch.cuda.empty_cache() 清理当前 GPU 上未使用的显存
                torch.cuda.empty_cache()
                print(f"模型 {model_name} 已卸载。")
    # 异步调度卸载任务，不阻塞当前线程
    asyncio.create_task(_unload())

##########################################################################
#                        启动任务和后台卸载任务                            #
##########################################################################
async def model_idle_checker():
    """
    后台任务：定时扫描所有已部署模型，
    如果模型长时间没有被调用，则卸载该模型释放显存。
    """
    while True:
        now = time.time()
        models_to_unload = []
        # 遍历已部署的模型，检查空闲时间
        for model_name, model_info in list(deployed_models.items()):
            if now - model_info["last_used"] > MODEL_IDLE_TIMEOUT:
                models_to_unload.append(model_name)
        # 卸载满足空闲条件的模型
        for model_name in models_to_unload:
            print(f"模型 {model_name} 长时间未调用，开始卸载。")
            unload_model(model_name)
        # 每 60 秒检测一次
        await asyncio.sleep(60)

async def init_deploy_models():
    """
    读取配置文件并部署初始模型
    """
    llms = get_model_info("LLMs", file_path="./config.json")
    components = get_model_info("Components", file_path="./config.json")
    
    for llm in llms:
        await deploy_model("llm-" + llm, 0)
    for component in components:
        await deploy_model("component-" + component, 0)

    print("所有初始模型已成功部署")

##########################################################################

@app.on_event("startup")
async def startup_event():
    """
    FastAPI 启动时，开启后台任务对闲置模型进行监控卸载。
    """
    asyncio.create_task(model_idle_checker())
    await init_deploy_models()

##########################################################################
#                            启动接口                                     #
##########################################################################
def start_api(host='localhost', port=53766, workers=1, CUDA_VISIBLE_DEVICES=None, TORCH_CUDA_ARCH_LIST=None):
    if CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    if TORCH_CUDA_ARCH_LIST is not None:
        os.environ['TORCH_CUDA_ARCH_LIST'] = TORCH_CUDA_ARCH_LIST
    uvicorn.run(app=app, host=host, port=port, workers=workers)

if __name__ == "__main__":
    start_api(port=8787, CUDA_VISIBLE_DEVICES="0")     
