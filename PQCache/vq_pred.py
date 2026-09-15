import os
from datasets import load_dataset
import torch
import json
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
from vq_method.llama_patch import VQLlamaForCausalLM
from vq_method.llama31_patch import VQLlama31ForCausalLM

from vq_method.mistral_patch import VQMistralForCausalLM
from h2o_method.h2o_attention import H2OLlamaForCausalLM, H2OLlamaAttention
from vq_method.retrieval_based.pq_search import initialize_objects, del_objects

import torch.distributed as dist
import torch.multiprocessing as mp
import time
from loguru import logger

from zeus.monitor import ZeusMonitor
from zeus.monitor.power import PowerMonitor
from transformers import LogitsProcessor, LogitsProcessorList

class ZeusEnergyTracker(LogitsProcessor):
    def __init__(self, monitor, power_monitor=None):
        self.monitor = monitor
        self.power_monitor = power_monitor

        self.is_first = True
        self.prefill_energy = 0.0
        self.prefill_time = 0.0

        self.prefill_start_time = None
        self.prefill_end_time = None
        
        self.decode_start_time = None
        self.decode_end_time = None

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
            self.decode_start_time = time.time()

            self.is_first = False
        return scores
    
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

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    compressor_choices = ["h2o", "original", "no_drop_lb", "pq_search","sparq_f"]
    parser.add_argument('--model', type=str, default=None, choices=[
        "llama-7b", "llama2-7b-chat-4k", "llama2-7b-32K", "mistral-7b-Instruct-32k", "llama-3", "llama-3.1","longchat-v1.5-7b-32k",
        "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k"])
    parser.add_argument('--e', action='store_true',
                        help="Evaluate on LongBench-E")
    parser.add_argument("--compress_ratio", type=float, default=1)
    parser.add_argument("--important_ratio", type=float, default=0)
    parser.add_argument("--recent_ratio", type=float, default=1)
    parser.add_argument('--enable_vq_cache', action='store_true')
    parser.add_argument('--enable_h2o_cache', action='store_true')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sink-size", type=int, default=2)
    parser.add_argument("--keyformer_mode",type=int, default=0)
    parser.add_argument("--drop_ratio", type=float, default=0)
    parser.add_argument("--exp_name", type=str, default="dafault_exp")
    parser.add_argument("--preserve_layer", type=int, default=0)
    parser.add_argument("--score_func", type=str, default="sum")
    parser.add_argument("--compressor", type=str, default="off", choices=compressor_choices)
    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument("--n_subvec_per_head", type=int, default=0)
    parser.add_argument("--n_subbits", type=int, default=0)
    parser.add_argument("--topr", type=int, default=32)
    parser.add_argument("--recent_size", type=int, default=32)
    parser.add_argument("--gqa", type=str, default="True")
    parser.add_argument("--sparq_mean_v_trick", type=str, default="False")
    parser.add_argument("--max_iter", type=int, default=0)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument('--dp', action='store_true',
                        help='whether data parallel')
    parser.add_argument('--pp-size', type=int, choices=[1,2,4,8])
    parser.add_argument('--test_mode', action='store_true')
    return parser.parse_args(args)

def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama" in model_name:
        if "3" in model_name:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            prompt = tokenizer.apply_chat_template(
                                    messages,
                                    tokenize=False,
                                    add_generation_prompt=True
                                )
        else:
            prompt = f"[INST]{prompt}[/INST]"
    elif "mistral" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(args, model, tokenizer, rank, world_size, data, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path, out_energy_path):
    data_idx = None
    device = model.device
    
    min_ctx_length = 100000
    max_ctx_length = 0
    line_num = 0
    all_time_elapsed = 0
    all_tt2t = 0
    all_token_generated = 0
    if not os.path.exists(out_path):
        line_num = 0
    else:
        with open(out_path, "r", encoding="utf-8") as f:
            while True:
                l = f.readline()
                if l == "":
                    break
                line_num += 1
    for i, json_obj in tqdm(enumerate(data)):
        if i < line_num:
            continue

        if data_idx is not None and i != (data_idx - 1):
            continue
        
        prompt = prompt_format.format(**json_obj)

        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        original_token_cnt = len(tokenized_prompt)
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(
                tokenized_prompt[-half:], skip_special_tokens=True)
            original_token_cnt = max_length

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False,
                                    return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False,
                                return_tensors="pt").to(device)
            
        context_length = input.input_ids.shape[-1]
        min_ctx_length = min(min_ctx_length, context_length)
        max_ctx_length = max(max_ctx_length, context_length)

        # 2. 实例化跟踪器
        # tracker = ZeusEnergyTracker(monitor, power_monitor)
        tracker = ZeusEnergyTracker(monitor)
        processors = LogitsProcessorList([tracker])

        # 3. 开启 Prefill 能耗测量窗口
        monitor.begin_window("prefill")
        tracker.prefill_start_time = time.time()

        begin_gen = time.perf_counter()
        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode(
                    "\n", add_special_tokens=False)[-1]],
                logits_processor=processors # <-- 注入处理器
            )[0]
        else:
            output = model.generate(
                input_ids=input.input_ids,
                attention_mask=input.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                
                min_new_tokens=max_gen,
                eos_token_id=None,
                suppress_tokens=[tokenizer.eos_token_id],

                logits_processor=processors # <-- 注入处理器
            )[0]
        
        # 4. 结束 Decode 能耗测量
        if not tracker.is_first:
            # 正常生成了 token，关闭 decode 窗口
            decode_measurement = monitor.end_window("decode")
            tracker.decode_end_time = time.time()

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

        end_gen = time.perf_counter()
        all_time_elapsed += end_gen - begin_gen
        all_token_generated += output[context_length:].shape[0]

        if args.enable_h2o_cache:
            for name, m in model.named_modules():
                if isinstance(m, H2OLlamaAttention):
                    m._clean_cache()

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        
        if data_idx is not None and i == (data_idx - 1):
            print("output:",pred)
            exit()
        
        if eval(os.environ.get("JUST_ONE_SAMPLE","False")):
            print(json_obj["answers"])
            exit()

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], 
                        "all_classes": json_obj["all_classes"], "length": json_obj["length"], 
                        "request_time": {"batch_time": 0, "batch_size": 1}, 
                        "input_tokens":int(original_token_cnt)}, f, ensure_ascii=False)
            f.write('\n')
        
        # --- 计算生成的 Token 数量 ---
        gen_token_ids = output[context_length:]
        # num_gen_tokens = (gen_token_ids != tokenizer.pad_token_id).sum().item()
        if tokenizer.pad_token_id is not None:
            num_gen_tokens = (gen_token_ids != tokenizer.pad_token_id).sum().item()
        else:
            num_gen_tokens = gen_token_ids.numel()

        # --- 新增的能耗指标 ---
        seq_prefill_energy = tracker.prefill_energy 
        seq_prefill_time = tracker.prefill_time
        seq_decode_energy = decode_energy
        seq_decode_time = decode_time

        # --- 计算 TPJ 和 TPS ---
        tpj = num_gen_tokens / seq_decode_energy if seq_decode_energy > 0 else 0.0
        tps = num_gen_tokens / seq_decode_time if seq_decode_time > 0 else 0.0

        # --- Maximum power for prefill and decode ---
        # prefill_max_power, prefill_power = tracker.get_prefill_max_power(gpu_id=0)
        # decode_max_power, decode_power = tracker.get_decode_max_power(gpu_id=0)

        with open(out_energy_path, "a", encoding="utf-8") as f:
            # json.dump({"prefill_energy_J": seq_prefill_energy, "prefill_time_s": seq_prefill_time, "decode_energy_J": seq_decode_energy, "decode_time_s": seq_decode_time,
            #             "prefill_max_power": prefill_max_power, "prefill_power": prefill_power, "decode_max_power": decode_max_power, "decode_power": decode_power,
            #             "generated_tokens": num_gen_tokens, "TPJ": tpj, "TPS": tps}, f, ensure_ascii=False)
            json.dump({"prefill_energy_J": seq_prefill_energy, "prefill_time_s": seq_prefill_time, "decode_energy_J": seq_decode_energy, "decode_time_s": seq_decode_time,
                        "input_tokens": context_length, "generated_tokens": num_gen_tokens, "TPJ": tpj, "TPS": tps}, f, ensure_ascii=False)
            f.write('\n')

    print("minimum length is ", min_ctx_length, "maximum", max_ctx_length)
    print("It takes", all_time_elapsed, "s to generate", all_token_generated, "tokens. tt2t is ", all_tt2t)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(args, path, model_name, device, pp_size = 1):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model = model.half().eval().to(device)
    elif "mistral" in model_name:
        config = AutoConfig.from_pretrained(path)
        config.recent_ratio = args.recent_ratio
        config.compress_ratio = args.compress_ratio
        config.important_ratio = args.important_ratio
        config.pp_size = pp_size
        config.sink_size = args.sink_size
        config.keyformer_mode = (args.keyformer_mode == 1)
        config.drop_ratio = args.drop_ratio
        config.preserve_layer = args.preserve_layer
        config.score_func = args.score_func
        config.compressor = args.compressor
        config.threshold = args.threshold
        config.n_subvec_per_head = args.n_subvec_per_head
        config.n_subbits = args.n_subbits
        config.topr = args.topr
        config.gqa = (args.gqa == "True")
        config.mean_v_trick = (args.sparq_mean_v_trick == "True")
        config.max_iter = args.max_iter
        config.device = torch.device("cuda:0")

        if config.compressor == "pq_search":
            config.max_seq_len = 33000
            config.cache_block_size = 128
            config.global_cache_size = 4096
            config.cache_topk = 32
            initialize_objects(config, model="mistral")

        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, config=config)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = VQMistralForCausalLM.from_pretrained(path, config=config)
        model = model.half().eval()
    elif "llama" in model_name and "2." in model_name:
        config = AutoConfig.from_pretrained(path)
        config.compress_ratio = args.compress_ratio
        config.important_ratio = args.important_ratio
        config.pp_size = pp_size
        config.sink_size = args.sink_size
        config.keyformer_mode = (args.keyformer_mode == 1)
        config.drop_ratio = args.drop_ratio
        config.preserve_layer = args.preserve_layer
        config.score_func = args.score_func
        config.compressor = args.compressor
        config.threshold = args.threshold
        config.n_subvec_per_head = args.n_subvec_per_head
        config.n_subbits = args.n_subbits
        config.topr = args.topr
        config.gqa = (args.gqa == "True")
        config.max_iter = args.max_iter
        config.device = torch.device("cuda:0")
        config.mean_v_trick = (args.sparq_mean_v_trick == "True")
        config.recent_ratio = args.recent_ratio
        if args.enable_vq_cache:
            config.compress_ratio = args.compress_ratio
            config.important_ratio = args.important_ratio
        elif args.enable_h2o_cache:
            config.hh_ratio = args.important_ratio
        
        if config.compressor == "pq_search":
            config.max_seq_len = 32768
            config.cache_block_size = 128
            config.global_cache_size = 4096
            config.cache_topk = 32
            initialize_objects(config, model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        if args.enable_vq_cache:
            model = VQLlamaForCausalLM.from_pretrained(path, config=config)
        elif args.enable_h2o_cache:
            model = H2OLlamaForCausalLM.from_pretrained(path, config=config)
        model = model.half().eval().to(device)
    elif "llama" in model_name and "3" in model_name:
        config = AutoConfig.from_pretrained(path)
        config.compress_ratio = args.compress_ratio
        config.important_ratio = args.important_ratio
        config.pp_size = pp_size
        config.sink_size = args.sink_size
        config.keyformer_mode = (args.keyformer_mode == 1)
        config.drop_ratio = args.drop_ratio
        config.preserve_layer = args.preserve_layer
        config.score_func = args.score_func
        config.compressor = args.compressor
        config.threshold = args.threshold
        config.n_subvec_per_head = args.n_subvec_per_head
        config.n_subbits = args.n_subbits
        config.topr = args.topr
        config.gqa = (args.gqa == "True")
        config.max_iter = args.max_iter
        config.device = torch.device("cuda:0")
        config.mean_v_trick = (args.sparq_mean_v_trick == "True")
        config.recent_ratio = args.recent_ratio
        if args.enable_vq_cache:
            config.compress_ratio = args.compress_ratio
            config.important_ratio = args.important_ratio
        elif args.enable_h2o_cache:
            config.hh_ratio = args.important_ratio
        
        if config.compressor == "pq_search":
            config.max_seq_len = 70000
            config.cache_block_size = 128
            config.global_cache_size = 4096
            config.cache_topk = 32
            initialize_objects(config, model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        if args.enable_vq_cache:
            model = VQLlama31ForCausalLM.from_pretrained(path, config=config, torch_dtype=torch.bfloat16)
        elif args.enable_h2o_cache:
            model = H2OLlamaForCausalLM.from_pretrained(path, config=config, torch_dtype=torch.bfloat16)
        model = model.eval()
        # model = model.half().eval()
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, use_fast=False)
        model = model.half().eval().to(device)
    
    return model, tokenizer

def get_config_str_list(args):
    if args.compressor in ["original"]:
        config_str = [
            "original"
        ]
    if args.compressor in ["pq_search","infllm"]:
        config_str = [
            f"budget_{args.compress_ratio}",
            f"rec_{args.recent_ratio}",
            f"sink_{args.sink_size}",
            f"mode_{args.compressor}",
            f"gqa_{args.gqa}",
            f"subvec_{args.n_subvec_per_head}",
            f"subbit_{args.n_subbits}",
            f"max_iter_{args.max_iter}"
        ]
    if args.compressor in ["sparq","sparq_f"]:
        config_str = [
            f"budget_{args.compress_ratio}",
            f"rec_{args.recent_ratio}",
            f"sink_{args.sink_size}",
            f"mode_{args.compressor}",
            f"gqa_{args.gqa}",
            f"topr_{args.topr}",
            f"mean_v_trick_{args.sparq_mean_v_trick}"
        ]
    elif args.compressor in ["only_h2o_hw", "only_h2o_lw", "no_drop_vb"]:
        config_str = [
            f"threshold_{args.threshold}",
            f"score_{args.score_func}",
            f"mode_{args.compressor}",
            f"sink_{args.sink_size}",
        ]
    elif args.compressor in ["off","vq_h2o","no_drop_lb","snapkv","h2o"]:
        config_str = [ 
            f"budget_{args.compress_ratio}",
            f"topk_{args.important_ratio}",
            f"rec_{args.recent_ratio}",
            f"sink_{args.sink_size}",
            f"gumbel_{args.keyformer_mode}",
            f"drop_{args.drop_ratio}",
            f"mode_{args.compressor}",
            f"skip_layer_{args.preserve_layer}",
            f"score_{args.score_func}",
            f"gqa_{args.gqa}",
        ]
    return config_str


if __name__ == '__main__':
    # 1. 初始化 Zeus Monitor (假设跑在 cuda:0 上，如果是多卡请调整 gpu_indices)
    monitor = ZeusMonitor(gpu_indices=[0])
    # power_monitor = PowerMonitor(
    #     gpu_indices=[0],
    #     power_domains=["device_instant"] # 显式指定我们要测的是 GPU 瞬时功率
    # )

    seed_everything(42)
    args = parse_args()
    assert args.enable_vq_cache + args.enable_h2o_cache == 1
    world_size = torch.cuda.device_count()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model_name = args.model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    
    # Tasks presented in paper
    # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
    #                 "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
    #                 "passage_count", "passage_retrieval_en"]
    # datasets = ["narrativeqa", "qmsum", "trec"]
    datasets = ["narrativeqa"]

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
        
    device = torch.device("cuda:0")
    model, tokenizer = load_model_and_tokenizer(args, model2path[model_name], model_name, device, args.pp_size)
    for dataset in datasets:
        logger.info(f"Yes we are evaluating {dataset}")
        if args.e:
            data = load_dataset('./data', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('json', data_files='./data/' +
                                dataset+'.jsonl', split='train')
            exp_name = args.exp_name
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            if not os.path.exists(f"pred/{model_name}/{dataset}"):
                os.makedirs(f"pred/{model_name}/{dataset}")
            if not os.path.exists(f"pred/{model_name}/{dataset}/{exp_name}"):
                os.makedirs(f"pred/{model_name}/{dataset}/{exp_name}")
            config_str_list = get_config_str_list(args)
            if args.enable_h2o_cache:
                out_path = f"pred/{model_name}/{dataset}/{exp_name}/h2o_hh_{args.important_ratio}_recent_{args.recent_ratio}.jsonl"
            elif args.enable_vq_cache:
                out_path = f"pred/{model_name}/{dataset}/{exp_name}/{'_'.join(config_str_list)}.jsonl"

        out_energy_path = f"pred/{model_name}/{dataset}/PQCache_energy.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        
        get_pred(args, model, tokenizer, 0, world_size, data_all, max_length, max_gen,
                    prompt_format, dataset, model_name, model2path, out_path, out_energy_path)
        
    print("All evaluation done.")
    del_objects()
