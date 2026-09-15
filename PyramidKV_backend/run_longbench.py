import os
import json
import random
import argparse
import time
import pynvml
import threading

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from zeus.monitor import ZeusMonitor
from zeus.monitor.power import PowerMonitor
from transformers import LogitsProcessor, LogitsProcessorList

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


datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

dataset2maxlen = {
    "narrativeqa": 1024,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

# model2maxlen = {
#     "Llama-2-7b-chat-hf": 3950,
#     "Llama-3-8B-Instruct": 7950,
#     "Meta-Llama-3-70B-Instruct": 7950,
#     "Meta-Llama-3-8B-Instruct-32k": 31500,
#     "Llama-2-7B-32K-Instruct": 31500,
#     "Mistral-7B-Instruct-v0.2": 31500,
#     "Mistral-7B-Instruct-v0.1": 31500,

# }

model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 512,
    "llama-3": 512,
    # "llama3": 7500,
    # "llama-3": 7500,
    "mistral": 31500
}

datatype = torch.bfloat16



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(prompt):
        prompt = f"[INST] {prompt} [/INST]"
        return prompt

# def build_prompt(prompt, dataset):
    
#     SYSTEM_PROMPT = model2prompt[dataset]

#     prompt = f"<<SYS>>\n {SYSTEM_PROMPT} \n<</SYS>>\n\n{prompt}"
#     return prompt

def main(args):
    

    print("Loading data...")
    
    test_data = []
    
    prompts = []
    inputs = []
    contexts = []
    answerss = []
    lengths = []
    datasets = []
    languages = []
    all_classess = []
    _ids = []
    
    input_max_len = 0
    
    model_path = args.model_path.lower()

    
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]
            

    
    output_max_len = dataset2maxlen[args.dataset]
    
    with open(args.data_file) as fp:
        for line in fp:
            example = json.loads(line)
            
            
            length = example["length"]
            if length > input_max_len: input_max_len = length
            
            template = model2prompt[args.dataset]
            prompt = template.format(**example)
            
            if "llama2" in args.model_path.lower():
                prompt = build_chat(prompt)
                
            example["prompt"] = prompt
                
            test_data.append(example)
        
    print(f"Max Length is {input_max_len}")
        
    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]
    
    
    for example in test_data:
        
        prompts.append(example["prompt"])
        inputs.append(example["input"])
        contexts.append(example["context"])
        answerss.append(example["answers"])
        lengths.append(example["length"])
        datasets.append(example["dataset"])
        languages.append(example["language"])
        all_classess.append(example["all_classes"])
        _ids.append(example["_id"])

    print("Finish loading model and tokenizer")
    
    model_name = model_path.split("/")[-1]

    os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset), exist_ok=True)

    fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, f"{args.method}.json"), "w")
    fout_energy = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, f"{args.method}_energy.json"), "w")
     
    for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
        
        batch_prompts = prompts[i:i+args.eval_batch_size]
        batch_inputs = inputs[i:i+args.eval_batch_size]
        batch_contexts = contexts[i:i+args.eval_batch_size]
        batch_answerss = answerss[i:i+args.eval_batch_size]
        batch_lengths = lengths[i:i+args.eval_batch_size]
        
        batch_datasets = datasets[i:i+args.eval_batch_size]
        batch_languages = languages[i:i+args.eval_batch_size]
        batch_all_classess = all_classess[i:i+args.eval_batch_size]
        batch__ids = _ids[i:i+args.eval_batch_size]
        
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        # Original: eval_bacth = 1
        # if len(batch_input_ids[0]) > model_max_len:
        #     half = int(model_max_len/2)
        #     prompt = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True)+tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True)

        #     tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        #     batch_input_ids = tokenized_prompts.input_ids
        #     attention_mask = tokenized_prompts.attention_mask
        
        # eval_bacth > 1, decode+re-encode 会有 token 数漂移 (511/512不稳定)，改为 token id 级别截断
        # if batch_input_ids.shape[1] > model_max_len:
        #     half = int(model_max_len/2)
        #     truncated_prompts = []
        #     for row, mask in zip(batch_input_ids, attention_mask):
        #         ids = row[mask.bool()]
        #         if len(ids) > model_max_len:
        #             truncated_prompts.append(
        #                 tokenizer.decode(ids[:half], skip_special_tokens=True)
        #                 + tokenizer.decode(ids[-half:], skip_special_tokens=True)
        #             )
        #         else:
        #             truncated_prompts.append(tokenizer.decode(ids, skip_special_tokens=True))
        #
        #     tokenized_prompts = tokenizer(truncated_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        #     batch_input_ids = tokenized_prompts.input_ids
        #     attention_mask = tokenized_prompts.attention_mask

        if batch_input_ids.shape[1] > model_max_len:
            half = int(model_max_len/2)
            truncated_ids_list = []
            for row, mask in zip(batch_input_ids, attention_mask):
                ids = row[mask.bool()]
                if len(ids) > model_max_len:
                    ids = torch.cat([ids[:half], ids[-half:]])
                truncated_ids_list.append(ids)

            tokenized_prompts = tokenizer.pad(
                {"input_ids": truncated_ids_list},
                padding="longest",
                return_tensors="pt",
            ).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask

        num_tokens = attention_mask.sum().item()

        # # default to True
        # if args.method == "DynamicKV":
        #     args.output_attentions = True
        # else:
        #     args.output_attentions=False

        if args.max_capacity_prompts != -1:
            max_capacity_prompts = args.max_capacity_prompts
        elif args.max_capacity_prompts_ratio != -1:
            max_capacity_prompts = round(batch_input_ids.shape[1] * args.max_capacity_prompts_ratio)
        
        
        if args.method != "FullKV":
            if args.method.lower() in ["snapkv","pyramidkv","h2o","cam", "l2norm", "adakv", "headkv", "think"]:
                window_sizes = 8
            elif args.method.lower() in ["streamingllm"]:
                window_sizes = max_capacity_prompts - 4

            if args.method.lower() =='headkv':
                with open(args.head_path, 'r') as file:
                    head_list = json.loads(file.readline())
                head_score_list = [np.mean(l[1]) for l in head_list.items()]
                head_score_list = torch.tensor(head_score_list / sum(head_score_list))
                total_attention = head_score_list.reshape(model.config.num_hidden_layers, model.config.num_attention_heads)
                total_pool_capacity = (args.max_capacity_prompts // args.head_beta) * model.config.num_hidden_layers * model.config.num_attention_heads
                min_num = (args.max_capacity_prompts - args.max_capacity_prompts // args.head_beta)
                head_capacity = torch.round(total_attention * total_pool_capacity + min_num).int()
                model.model.config.head_capacity = head_capacity 
                # model.model.config.head_choice = 'reason'

            kernel_sizes = 7
            pooling = "maxpool"
            ratio = args.pruning_ratio
            recent_size = args.recent_size

            layers = len(model.model.layers)
            # check if window_sizes is a list
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            if not isinstance(max_capacity_prompts, list):
                max_capacity_prompts = [max_capacity_prompts] * layers
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers
            if not isinstance(ratio, list):
                ratio = [ratio] * layers
            if not isinstance(recent_size, list):
                recent_size = [recent_size] * layers
            for i in range(layers):
                model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                model.model.layers[i].self_attn.config.pooling = pooling
                model.model.layers[i].self_attn.config.merge = args.merge
                model.model.layers[i].self_attn.config.floor = args.floor
                model.model.layers[i].self_attn.config.ratio = ratio[i]
                model.model.layers[i].self_attn.config.recent_size = recent_size[i]
            

        context_length = batch_input_ids.shape[-1]

        # 2. 实例化跟踪器
        # tracker = ZeusEnergyTracker(monitor, power_monitor)
        tracker = ZeusEnergyTracker(monitor)
        processors = LogitsProcessorList([tracker])

        # 3. 开启 Prefill 能耗测量窗口
        monitor.begin_window("prefill")
        tracker.main_freq_monitor.begin_window()
        tracker.prefill_start_time = time.time()

        if args.quant_method == None:        
            output = model.generate(
                **tokenized_prompts,
                output_attentions = args.output_attentions,
                max_new_tokens=output_max_len,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                # min_length=context_length+1,
                # eos_token_id=[tokenizer.eos_token_id],
                min_new_tokens=output_max_len,
                eos_token_id=None,
                suppress_tokens=[tokenizer.eos_token_id],
                logits_processor=processors # <-- 注入处理器
            )
        else:
            output = model.generate(
                **tokenized_prompts,
                output_attentions = args.output_attentions,
                max_new_tokens=output_max_len,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                # min_length=context_length+1,
                # eos_token_id=[tokenizer.eos_token_id],
                min_new_tokens=output_max_len,
                eos_token_id=None,
                suppress_tokens=[tokenizer.eos_token_id],
                cache_implementation="quantized", 
                cache_config={
                    "nbits": args.nbits, 
                    "backend": "HQQ",
                    "device":"cuda",
                    "residual_length":output_max_len,
                    "axis_key":1,
                    "q_group_size":64, 
                    "compute_dtype": datatype
                },
                logits_processor=processors # <-- 注入处理器
            )

        # 4. 结束 Decode 能耗测量
        if not tracker.is_first:
            decode_measurement = tracker.end_generation()
            if decode_measurement:
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
        
        # batch_outputs =tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)
        batch_outputs = tokenizer.batch_decode([output[k][context_length:] for k in range(output.shape[0])], skip_special_tokens=True)
        
        # print(f"debbug batch_outputs {batch_outputs}")
        
        batch_generations = batch_outputs

        torch.cuda.empty_cache()

        for j in range(args.eval_batch_size):
            
            example = {}
            
            example["prompt"] = batch_prompts[j]
            example["input"] = batch_inputs[j]
            example["context"] = batch_contexts[j]
            example["answers"] = batch_answerss[j]
            example["pred"] = batch_generations[j]
            example["length"] = batch_lengths[j]
            
            example["dataset"] = batch_datasets[j]
            example["language"] = batch_languages[j]
            example["all_classes"] = batch_all_classess[j]
            example["_id"] = batch__ids[j]

            # print(f'{batch_generations[j]}')
            fout.write(json.dumps(example) + "\n")

        example_energy = {}
        # --- 计算生成的 Token 数量 ---
        # 从输出张量中截取新生成的部分，并排除可能存在的 pad_token
        gen_token_ids = output[j][context_length:]
        # num_gen_tokens = (gen_token_ids != tokenizer.pad_token_id).sum().item()
        num_gen_tokens = gen_token_ids.shape[0]

        # --- 新增的能耗指标 ---
        # 注意：如果你使用 batched inference (eval_batch_size > 1)，
        # 这个测量值是整个 batch 的。这里将其平均分配给单个 example。
        # 将整个 Batch 的能耗和时间平均分配给单个 sample
        seq_prefill_energy = tracker.prefill_energy
        seq_prefill_time = tracker.prefill_time
        seq_decode_energy = decode_energy
        seq_decode_time = decode_time

        # --- 计算 TPJ 和 TPS ---
        # 避免除以 0 的情况
        tpj = num_gen_tokens / seq_decode_energy if seq_decode_energy > 0 else 0.0
        tps = num_gen_tokens / seq_decode_time if seq_decode_time > 0 else 0.0

        # --- Maximum power for prefill and decode ---
        
        example_energy["prefill_energy_J"] = seq_prefill_energy
        example_energy["prefill_time_s"] = seq_prefill_time

        example_energy["decode_energy_J"] = seq_decode_energy
        example_energy["decode_time_s"] = seq_decode_time

        # 记录具体的吞吐量与能效指标
        example_energy["input_tokens"] = num_tokens // args.eval_batch_size
        example_energy["generated_tokens"] = num_gen_tokens
        example_energy["TPJ"] = tpj  # Tokens Per Joule (越高越省电)
        example_energy["TPS"] = tps  # Tokens Per Second (越高越快)

        example_energy["energy_list"] =  tracker.token_energy_list
        example_energy["time_list"] = tracker.token_time_list
        example_energy["tpj_list"] = tracker.tpj_list 
        example_energy["SM_frequency_list"] = tracker.tokenSM_freq_list
        example_energy["Mem_frequency_list"] = tracker.tokenMem_freq_list  

        # print(f'{batch_generations[j]}')
        fout_energy.write(json.dumps(example_energy) + "\n")
    

if __name__ == "__main__":

    # 1. 初始化 Zeus Monitor (假设跑在 cuda:0 上，如果是多卡请调整 gpu_indices)
    monitor = ZeusMonitor(gpu_indices=[0])
    # power_monitor = PowerMonitor(
    #     gpu_indices=[0],
    #     power_domains=["device_instant"] # 显式指定我们要测的是 GPU 瞬时功率
    # )

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
    
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--quant_method",type=str,default=None,choices=["kivi","kvquant"])
    parser.add_argument("--nbits", type=int, default=8, help="")
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")
    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--merge", type=str, default=None, help="kv merge method(look-m)")
    parser.add_argument('--floor', type=float, default=0.2, help='hyper-parameter used in AdaKV')
    parser.add_argument('--head_path', type=str, default='./data/heads_score/Meta-Llama-3-8B-Instruct_retrieval_reasoning_heads.json', help='Path to head score (HeadKV)')
    parser.add_argument('--head_beta', type=float, default=1.01, help='hyper-parameter used on HeadKV')
    parser.add_argument("--recent_size", type=int, default=32, help="")
    parser.add_argument("--pruning_ratio", type=float, default=0.4, help="pruning ratio of Key Cache")

    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    if args.quant_method == "kvquant":
        from pyramidkv.quantcache import KVQuantizedCache
        from transformers import cache_utils
        cache_utils.HQQQuantizedCache = KVQuantizedCache
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )


    from pyramidkv.monkeypatch import replace_llama,replace_mistral
    replace_llama(args.method.lower())
    replace_mistral(args.method.lower())
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        # torch_dtype=torch.float16,
        torch_dtype=datatype,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation
    )
        

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    

        
    model.eval()
    
    save_dir = args.save_dir
    
        
    max_capacity_prompts = args.max_capacity_prompts

    # selected_datasets = ["narrativeqa", "qmsum", "trec"]
    
    for idx, dataset in enumerate(datasets):
        
        print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(datasets)}")

        if dataset not in ["narrativeqa"]:
            continue

        args.dataset = dataset
        
        args.data_file = f"data/LongBench/{args.dataset}.jsonl"
        
        main(args)
    
    # power_monitor.stop()
