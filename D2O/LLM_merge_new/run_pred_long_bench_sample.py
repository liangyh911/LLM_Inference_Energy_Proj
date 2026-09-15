
import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import time
import pynvml
import threading

import argparse
os.environ["WANDB_DISABLED"] = "true"
from typing import Optional
from utils.process_args import process_args
from transformers import LlamaConfig, MistralConfig, AutoTokenizer
import os
from dataclasses import dataclass, field
import transformers

from transformers import AutoConfig

import multiprocessing as mp
from zeus.monitor import ZeusMonitor
from zeus.monitor.power import PowerMonitor
from transformers import LogitsProcessor, LogitsProcessorList


# for llama 3
from LMEval_kv_token_merge.modeling_llama3_7b_13b_drop import Llama3ForCausalLM_drop, Llama3Attention_drop
from LMEval_kv_token_merge.modeling_llama3_7b_13b_d2o import Llama3ForCausalLM_D2O, LlamaAttention3_D2O  # 
from LMEval_kv_token_merge.modeling_llama3_full import Llama3ForCausalLM
from LMEval_kv_token_merge.modeling_llama3_streaming import Llama3Attention_streaming, Llama3ForCausalLM_streaming


TAGET_MODULE_FOR_LLAMA_3 = {
    "llama3": None,
    "real_drop": Llama3Attention_drop,
    "real_d2o": LlamaAttention3_D2O,
    "real_stream": Llama3Attention_streaming
}

class GPUFreqMonitor:
    def __init__(self, gpu_id=0, interval=0.05):
        self.gpu_id = gpu_id
        self.interval = interval
        self.is_running = False
        self.sm_samples = []
        self.mem_samples = []
        self.thread = None
        
        # 初始化 NVML
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)

    def begin_window(self):
        self.sm_samples = []
        self.mem_samples = []
        self.is_running = True
        self.thread = threading.Thread(target=self._poll)
        self.thread.start()

    def _poll(self):
        while self.is_running:
            # NVML_CLOCK_GRAPHICS 即为 SM 核心计算频率
            sm_freq_mhz = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
            mem_freq_mhz = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            
            self.sm_samples.append(sm_freq_mhz)
            self.mem_samples.append(mem_freq_mhz)
            
            time.sleep(self.interval)

    def end_window(self):
        self.is_running = False
        if self.thread is not None:
            self.thread.join()
        
        # 兜底机制：如果窗口太短没采到样，直接取一次瞬时频率
        if len(self.sm_samples) == 0:
            # return pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
            avg_sm_freq = 0
        else:
            avg_sm_freq = sum(self.sm_samples) / len(self.sm_samples)

        if len(self.mem_samples) == 0:
            # return pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
            avg_mem_freq = 0
        else:
            avg_mem_freq = sum(self.mem_samples) / len(self.mem_samples)
        
        return avg_sm_freq, avg_mem_freq

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass

class ZeusEnergyTracker(LogitsProcessor):
    def __init__(self, monitor, power_monitor=None, measure_interval=64, gpu_id=0):
        self.monitor = monitor
        self.power_monitor = power_monitor
        self.gpu_id = gpu_id

        self.is_first = True
        self.is_finished = False

        self.token_count = 0
        self.measure_interval = measure_interval

        self.prefill_energy = 0.0
        self.prefill_time = 0.0

        # === 频率监控：新增状态变量 ===
        self.prefill_avg_freq = 0.0
        self.decode_avg_freq = 0.0

        self.prefill_start_time = None
        self.prefill_end_time = None
        
        self.decode_start_time = None
        self.decode_end_time = None

        self.token_start_time = None
        self.token_end_time = None

        self.token_energy_list = []
        self.token_time_list = []
        self.tpj_list = []
        # === 频率监控：新增列表 ===
        self.tokenSM_freq_list = []
        self.tokenMem_freq_list = []

        # === 频率监控：实例化 Monitor ===
        # 1. 负责总体 (Prefill & Decode)
        self.main_freq_monitor = GPUFreqMonitor(gpu_id=self.gpu_id, interval=0.05)
        # 2. 负责局部 Token 块 (稍微加快采样率到 20ms，因为 8 个 token 可能只有一百多毫秒)
        self.chunk_freq_monitor = GPUFreqMonitor(gpu_id=self.gpu_id, interval=0.02)

    def __call__(self, input_ids, scores):
        # 当这个方法第一次被调用时，意味着 Prefill 阶段刚刚完成
        if self.is_first:
            # 停止 Prefill 测量
            measurement = self.monitor.end_window("prefill")
            self.prefill_end_time = time.time()

            self.prefill_energy = measurement.total_energy
            self.prefill_time = measurement.time
            
            # 立即开始 Decode 测量
            self.monitor.begin_window("decode")
            # === 频率监控：开启全局 Decode 频率测量 ===
            self.main_freq_monitor.begin_window()
            self.decode_start_time = time.time()

            self.monitor.begin_window("token")
            # === 频率监控：开启第一个 Token 块频率测量 ===
            self.chunk_freq_monitor.begin_window()
            self.token_start_time = time.time()

            self.is_first = False
            self.token_count += 1
        else:
            self.token_count += 1

            if self.token_count % self.measure_interval == 0:
                measurement = self.monitor.end_window("token")
                # === 频率监控：结束当前 Token 块频率测量 ===
                chunk_avg_sm_freq, chunk_avg_mem_freq = self.chunk_freq_monitor.end_window()
                self.token_end_time = time.time()

                self.token_energy_list.append(measurement.total_energy)
                self.token_time_list.append(measurement.time)
                # === 频率监控：记录频率 ===
                self.tokenSM_freq_list.append(chunk_avg_sm_freq)
                self.tokenMem_freq_list.append(chunk_avg_mem_freq)

                tpj = (self.measure_interval / measurement.total_energy) if measurement.total_energy != 0 else 0
                self.tpj_list.append(tpj)

                self.monitor.begin_window("token")
                # === 频率监控：开启下一个 Token 块频率测量 ===
                self.chunk_freq_monitor.begin_window()

        return scores
    
    def end_generation(self):
        if not self.is_first and not self.is_finished:
            # 处理最后不足 10 个的尾部 Token (Remainder)
            remainder = self.token_count % self.measure_interval
            
            measurement = self.monitor.end_window("token")
            # === 频率监控：结束最后一个 Token 块频率测量 ===
            chunk_avg_sm_freq, chunk_avg_mem_freq = self.chunk_freq_monitor.end_window()

            # 如果最后还剩几个 token，把它们记录下来
            if remainder != 0:
                self.token_energy_list.append(measurement.total_energy)
                self.token_time_list.append(measurement.time)
                self.tokenSM_freq_list.append(chunk_avg_sm_freq)
                self.tokenMem_freq_list.append(chunk_avg_mem_freq)

                tpj = (remainder / measurement.total_energy) if measurement.total_energy != 0 else 0
                self.tpj_list.append(tpj)

            # 闭合整个 Decode 窗口
            decode_measurement = self.monitor.end_window("decode")
            # === 频率监控：结束全局 Decode 频率测量 ===
            self.decode_avg_freq = self.main_freq_monitor.end_window()
            self.decode_end_time = time.time()
            
            self.is_finished = True
            return decode_measurement

    def get_prefill_max_power(self, gpu_id):
        timeline_dict = self.power_monitor.get_power_timeline(
            power_domain="device_instant",
            start_time=self.prefill_start_time,
            end_time=self.prefill_end_time
        )

        samples = timeline_dict.get(gpu_id, [])

        if samples:
            max_power_w = 0
            for s in samples:
                power_w = s[1]
                if power_w > max_power_w:
                    max_power_w = power_w
            return max_power_w, samples
        else:
            return None, None
    
    
    def get_decode_max_power(self, gpu_id):
        timeline_dict = self.power_monitor.get_power_timeline(
            power_domain="device_instant",
            start_time=self.decode_start_time,
            end_time=self.decode_end_time
        )

        samples = timeline_dict.get(gpu_id, [])

        if samples:
            max_power_w = 0
            for s in samples:
                power_w = s[1]
                if power_w > max_power_w:
                    max_power_w = power_w
        
            return max_power_w, samples
        else:
            return None, None

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Output model local path, do not set manually"}
    )

    
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative manifold path"}
    )
    load_quant: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a quantized model"},
    )
    w_bit: Optional[int] = field(
        default=4,
        metadata={"help": "The model weight bit width."},
    )
    lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use LoRA"},
    )
    lora_mode: Optional[str] = field(
        default="q",
        metadata={"help": "LoRA mode"},
    )
    lora_r: Optional[int] = field(
        default=1,
        metadata={"help": "LoRA r"},
    )
    lora_alpha: Optional[float] = field(
        default=1.,
        metadata={"help": "LoRA alpha"},
    )
    lora_dropout: Optional[float] = field(
        default=0.,
        metadata={"help": "LoRA dropout"},
    )
    hh_ratio: Optional[float] = field(
        default=0.1,
        metadata={"help": "important ratio"},
    )

    recent_ratio: Optional[float] = field(
        default=0.1,
        metadata={"help": "recent ratio"},
    )

    hh_size: Optional[int] = field(
        default=200,
        metadata={"help": "important windows size"},
    )

    recent_size: Optional[int] = field(
        default=200,
        metadata={"help": "recent window size"},
    )
    
    use_full: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use our d2o method."},
    )

    use_real_drop: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use our d2o method."},
    )

    use_d2o: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use our d2o method."},
    )

    use_real_streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use our d2o method."},
    )

    model_type: Optional[str] = field(
        default='llama',
        metadata={"help": "The path to a quantized model"},
    )


    action_name: str = field(
        default='full', metadata={"help": "Output model local path, do not set manually"}
    )

    alpha: Optional[float] = field(
        default=0.5,
        metadata={"help": "recent ratio"},
    )

    belta: Optional[float] = field(
        default=0.5,
        metadata={"help": "recent ratio"},
    )


@dataclass
class DataArguments:
    dataset: Optional[str] = field(
        default='c4',
        metadata={"help": "The dataset used for fine-tuning the model."},
    )
    eval_tasks: Optional[str] = field(
        default='wikitext',
        metadata={"help": "The dataset used for evaluation."},
    )
    tasks: Optional[str] = field(
        default='wikitext',
        metadata={"help": "The dataset used for evaluation."},
    )
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "The batch size."},
    )
    num_fewshot: Optional[int] = field(
        default=0,
        metadata={"help": "The number of fewshot examples."},
    )
    output_path: Optional[str] = field(
        default='./outputs',
        metadata={"help": "The output path."},
    )
    e: Optional[bool] = field(
        default=False,
        metadata={"help": "Evaluate on LongBench-E."},
    )
    use_our_imp: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use our KV cache quantization implementation."},
    )



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default='/fs/scratch/PAS2473/zhongwei_models')
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="./outputs")
    model_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). 512 or 1024"
        },
    )
    num_train_epochs: Optional[int] = field(default=1)
    n_train_samples: Optional[int] = field(default=None)
    n_eval_samples: Optional[int] = field(default=None)
    qat: Optional[bool] = field(default=False)
    exp_name: Optional[str] = field(default="test")


def process_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)

    model_args.output_model_local_path = os.path.join(
        training_args.output_dir, "models", str(model_args.output_model_filename)
    )

    return model_args, data_args, training_args


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # For results in KIVI paper (Llama, Llama-Chat, Mistral-7B-v0.1), we do not apply any special treatment to the prompt.
    # For lmsys/longchat-7b-v1.5-32k and mistralai/Mistral-7B-Instruct-v0.2, we need to rewrite the prompt a little bit.
    if "longchat" in model_name.lower():
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "mistral-v0.2-instruct" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name):
    preds = []
    energy = []
    # for json_obj in tqdm(data):
    for json_obj in tqdm(data.select(range(100))):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        # if "chatglm3" in model:
        #     tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        
        # 2. 实例化跟踪器
        # tracker = ZeusEnergyTracker(monitor, power_monitor)
        tracker = ZeusEnergyTracker(monitor)
        processors = LogitsProcessorList([tracker])

        # 3. 开启 Prefill 能耗测量窗口
        monitor.begin_window("prefill")
        tracker.prefill_start_time = time.time()

        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                # min_length=context_length+1,
                # eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                min_new_tokens=max_gen,
                eos_token_id=None,
                suppress_tokens=[tokenizer.eos_token_id],
                logits_processor=processors # <-- 注入处理器
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_new_tokens=max_gen,
                eos_token_id=None,
                suppress_tokens=[tokenizer.eos_token_id],
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=processors # <-- 注入处理器
            )[0]
        
        # 4. 结束 Decode 能耗测量
        if not tracker.is_first:
            # 正常生成了 token，关闭 decode 窗口（同时会关闭仍处于打开状态的 "token" 窗口）
            decode_measurement = tracker.end_generation()

            decode_energy = decode_measurement.total_energy
            decode_time = decode_measurement.time
        else:
            # 极端异常情况：如果模型一个 token 都没生成直接退出
            measurement = monitor.end_window("prefill")
            tracker.decode_end_time = time.time()

            tracker.prefill_energy = measurement.total_energy
            tracker.prefill_time = measurement.time
            decode_energy = 0.0
            decode_time = 0.0

        ################## modified here #############################
        
        json_obj["length"]

        if model_args.model_type == 'llama3':

            if model_args.use_real_drop:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_LLAMA_3['real_drop']):
                        m._clean_cache()

            elif model_args.use_d2o:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_LLAMA_3['real_d2o']):
                        # breakpoint()
                        m._clean_cache()


                 ##### clean and reset the dynamic cache size ####################

                num_layers = len(model.model.layers)

                for layer_idx in range(num_layers):

                    model.model.layers[layer_idx].self_attn.layer_hh_ratio = 0
                    model.model.layers[layer_idx].self_attn.layer_recent_ratio = 0

                    model.model.layers[layer_idx].self_attn.multimodal_entropy = 0

                    model.model.layers[layer_idx].self_attn.prfill_size = 0

                    model.model.layers[layer_idx].self_attn.prfill_score = None

            elif model_args.use_real_streaming:
                for name, m in model.named_modules():
                    if isinstance(m, TAGET_MODULE_FOR_LLAMA_3['real_stream']):
                        # breakpoint()
                        m._clean_cache()

    
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})

        # --- 计算生成的 Token 数量 ---
        gen_token_ids = output[context_length:]
        # num_gen_tokens = (gen_token_ids != tokenizer.pad_token_id).sum().item()
        # if tokenizer.pad_token_id is not None:
        #     num_gen_tokens = (gen_token_ids != tokenizer.pad_token_id).sum().item()
        # else:
        #     num_gen_tokens = gen_token_ids.numel()
        num_gen_tokens = gen_token_ids.shape[0]

        # --- 新增的能耗指标 ---
        seq_prefill_energy = tracker.prefill_energy 
        seq_prefill_time = tracker.prefill_time / data_args.batch_size
        seq_decode_energy = decode_energy / data_args.batch_size
        seq_decode_time = decode_time / data_args.batch_size

        # --- 计算 TPJ 和 TPS ---
        tpj = num_gen_tokens / seq_decode_energy if seq_decode_energy > 0 else 0.0
        tps = num_gen_tokens / seq_decode_time if seq_decode_time > 0 else 0.0

        # --- Maximum power for prefill and decode ---
        # prefill_max_power, prefill_power = tracker.get_prefill_max_power(gpu_id=0)
        # decode_max_power, decode_power = tracker.get_decode_max_power(gpu_id=0)

        energy.append({
            "prefill_energy_J": seq_prefill_energy, "prefill_time_s": seq_prefill_time, "decode_energy_J": seq_decode_energy, "decode_time_s": seq_decode_time,
            # "prefill_max_power": prefill_max_power, "prefill_power": prefill_power, "decode_max_power": decode_max_power, "decode_power": decode_power,
            "input_tokens": context_length,
            "generated_tokens": num_gen_tokens, "TPJ": tpj, "TPS": tps,
            "energy_list": tracker.token_energy_list, "time_list": tracker.token_time_list, "tpj_list": tracker.tpj_list,
            "SM_frequency_list": tracker.tokenSM_freq_list, "Mem_frequency_list": tracker.tokenMem_freq_list}) 
                    
    return preds, energy

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    
    # 1. 初始化 Zeus Monitor (假设跑在 cuda:0 上，如果是多卡请调整 gpu_indices)
    monitor = ZeusMonitor(gpu_indices=[0])
    # power_monitor = PowerMonitor(
    #     gpu_indices=[0],
    #     power_domains=["device_instant"],
    #     update_period=0.1
    # )

    seed_everything(42)
    # args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_name = args.model

    # define your model
    model_args, data_args, training_args = process_args()
    # print(model_args, data_args, training_args)
    model_name = model_args.model_name_or_path.split("/")[-1]
    # dtype = torch.bfloat16 if training_args.bf16 else torch.float
    # dtype = torch.float16
    dtype = torch.bfloat16
    
    if 'llama' in model_args.model_name_or_path.lower() or 'longchat' in model_args.model_name_or_path.lower():

        if model_args.model_type == 'llama3':

            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                                use_fast=False, 
                                                trust_remote_code=True, 
                                               )
        else:
        

            config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                                use_fast=False, 
                                                trust_remote_code=True, 
                                                tokenizer_type='llama')
                                            # model_max_length=training_args.model_max_length)

    if model_args.model_type == 'llama3' :
           
        if model_args.use_full:


            config.hh_size = model_args.hh_size
            config.recent_size = model_args.recent_size
            config.hh_ratio = model_args.hh_ratio
            config.recent_ratio = model_args.recent_ratio

            model = Llama3ForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                # use_flash_attention_2=True,
                device_map="auto",
            )
        
        elif model_args.use_real_drop:

            config.hh_size = model_args.hh_size
            config.recent_size = model_args.recent_size
            config.hh_ratio = model_args.hh_ratio
            config.recent_ratio = model_args.recent_ratio

            config.alpha = model_args.alpha
            config.belta = model_args.belta
            
            
            model = Llama3ForCausalLM_drop.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                # use_flash_attention_2=True,
                device_map="auto",
            )

        elif model_args.use_d2o:

            config.hh_size = model_args.hh_size
            config.recent_size = model_args.recent_size
            config.hh_ratio = model_args.hh_ratio
            config.recent_ratio = model_args.recent_ratio
            config.alpha = model_args.alpha
            config.belta = model_args.belta


            model = Llama3ForCausalLM_D2O.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                # use_flash_attention_2=True,
                device_map="auto",
            )

        elif model_args.use_real_streaming:

            config.hh_size = model_args.hh_size
            config.recent_size = model_args.recent_size
            config.hh_ratio = model_args.hh_ratio
            config.recent_ratio = model_args.recent_ratio

            model = Llama3ForCausalLM_streaming.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                # use_flash_attention_2=True,
                device_map="auto",
            )


    model.eval()
    max_length = model2maxlen[model_name]
    if data_args.e:
        # datasets = ["triviaqa"]
        datasets = ["narrativeqa", "qmsum", "trec"]
    else:
        datasets = ["narrativeqa"]
        # datasets = ["narrativeqa", "qmsum", "trec"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if data_args.e:

            if dataset in ["narrativeqa","qmsum", "musique"]:
                
                data = load_dataset('THUDM/LongBench', f"{dataset}", split='test')

            else:
                
                data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')

            if not os.path.exists(f"pred_e/{model_name}_{model_args.action_name}"):
                os.makedirs(f"pred_e/{model_name}_{model_args.action_name}")
            out_path = f"pred_e/{model_name}_{model_args.action_name}/{dataset}.jsonl"
            out_energy_path = f"pred_e/{model_name}_{model_args.action_name}/{dataset}_energy.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}_{model_args.action_name}"):
                os.makedirs(f"pred/{model_name}_{model_args.action_name}")
            out_path = f"pred/{model_name}_{model_args.action_name}/{dataset}.jsonl"
            out_energy_path = f"pred/{model_name}_{model_args.action_name}/{dataset}_energy.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        
        preds, energy = get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
        
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')
        
        with open(out_energy_path, "w", encoding="utf-8") as f:
            for e in energy:
                json.dump(e, f, ensure_ascii=False)
                f.write('\n')



"""
### running sample #######################

CUDA_VISIBLE_DEVICES=0 python run_pred_long_bench_sample.py --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --cache_dir /fs/scratch/PAS2473/zhongwei_new_models \
    --use_d2o True \
    --model_type llama3 \
    --hh_ratio 0.1 \
    --recent_ratio 0.1 \
    --action_name d2o_0.2 \
    --e True 

"""

 
