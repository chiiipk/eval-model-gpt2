import torch
import os
import json
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm

from utils import log_rank
from typing import Dict, Optional
from transformers import AutoTokenizer


class DistillDataset(Dataset):
    def __init__(
        self, 
        args, 
        split: str,
        student_tokenizer: Dict[str, AutoTokenizer], 
        teacher_tokenizers: Optional[Dict[str, AutoTokenizer]] = {},
    ):
        self.args = args
        self.split = split
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizers = teacher_tokenizers
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.dataset = self._load_and_process_data()
        # log_rank(f"Num of data instances: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)
   
    def __getitem__(self, index):
        return self.dataset[index]
    
    def _load_and_process_data(self):
        dataset = []
        path = os.path.join(self.args.data_dir, f"{self.split}.jsonl")

        if os.path.exists(path):
            with open(path) as f:
                raw_data = [json.loads(l) for l in f.readlines()]
                self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in raw_data]
            
            log_rank("Processing dataset for student model (and all teacher models)...")  
            seg = np.iinfo(np.int32).max * 2 + 1        
            for data in tqdm(raw_data, disable=(dist.get_rank() != 0)):
                student_prompt_ids = self.student_tokenizer.encode(
                    data["prompt"], add_special_tokens=False
                )
                student_prompt_ids = student_prompt_ids[:self.max_prompt_length]
                student_response_ids = self.student_tokenizer.encode(
                    data["output"], add_special_tokens=False
                )
                student_response_ids = student_response_ids \
                                     + [self.student_tokenizer.eos_token_id]
                tokenized_data = {
                    "student_input_ids": student_prompt_ids + [seg] + student_response_ids,
                }
        
                for model_type in self.teacher_tokenizers:
                    if self.teacher_tokenizers[model_type] is None: continue
                        
                    teacher_prompt_ids = self.teacher_tokenizers[model_type].encode(
                        data["prompt"], add_special_tokens=False
                    )
                    teacher_prompt_ids = teacher_prompt_ids[:self.max_prompt_length]
                    teacher_response_ids = self.teacher_tokenizers[model_type].encode(
                        data["output"], add_special_tokens=False
                    )
                    teacher_response_ids = teacher_response_ids \
                                            + [self.teacher_tokenizers[model_type].eos_token_id]
                    tokenized_data[f"teacher_{model_type}_input_ids"] = \
                        teacher_prompt_ids + [seg] + teacher_response_ids

                dataset.append(tokenized_data)
            return dataset
        else:
            raise FileNotFoundError(f"No such file named {path}")
        
    def _process_lm(
        self, i, samp, model_data, no_model_data, gen_data, 
        teacher_model_data, teacher_no_model_data
    ):
        seg = np.iinfo(np.int32).max * 2 + 1
        input_ids = np.array(samp["student_input_ids"])
        source_len = np.where(input_ids == seg)[0][0]
        prompt = input_ids[:source_len]
        input_ids = np.concatenate(
            [input_ids[:source_len], input_ids[source_len+1:]], axis=0
        )
        input_ids = input_ids[:self.max_length]
        input_len = len(input_ids)
        model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:input_len-1] = 1.0
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        no_model_data["label"][i][:source_len-1] = -100
        no_model_data["loss_mask"][i][:input_len-1] = 1.0
        no_model_data["loss_mask"][i][:source_len-1] = 0
        
        gen_data["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
        gen_data["attention_mask"][i][-len(prompt):] = 1.0

        for model_type in self.teacher_tokenizers:
            t_input_ids = np.array(samp[f"teacher_{model_type}_input_ids"])
            t_source_len = np.where(t_input_ids == seg)[0][0]
            t_input_ids = np.concatenate(
                [t_input_ids[:t_source_len], t_input_ids[t_source_len+1:]], axis=0
            )
            t_input_ids = t_input_ids[:self.max_length]
            t_input_len = len(t_input_ids)
            teacher_model_data[model_type]["input_ids"][i][:t_input_len-1] = \
                torch.tensor(t_input_ids[:-1], dtype=torch.long)
            teacher_model_data[model_type]["attention_mask"][i][:t_input_len-1] = 1.0
            if model_type in ["gpt2"]:
                teacher_model_data[model_type]["position_ids"][i][:t_input_len-1] = \
                    torch.arange(0, t_input_len-1, dtype=torch.long)
            teacher_no_model_data[model_type]["label"][i][:t_input_len-1] = \
                torch.tensor(t_input_ids[1:], dtype=torch.long)
            teacher_no_model_data[model_type]["label"][i][:t_source_len-1] = -100
            teacher_no_model_data[model_type]["loss_mask"][i][:t_input_len-1] = 1.0
            teacher_no_model_data[model_type]["loss_mask"][i][:t_source_len-1] = 0

    def move_to_device(self, datazip, device):
        for data in datazip:
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)
                elif isinstance(data[k], dict):
                    for kk in data[k]:
                        data[k][kk] = data[k][kk].to(device)

    def collate(self, samples):
        bs = len(samples)
        max_length = self.max_length

        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) \
                        * self.student_tokenizer.eos_token_id,
            "attention_mask": torch.zeros(bs, max_length),
        }
        
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
            
        no_model_data = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length)
        }
        
        gen_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) \
                        * self.student_tokenizer.eos_token_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }

        teacher_model_data = {
            model_type: {
                "input_ids": torch.ones(bs, max_length, dtype=torch.long) \
                            * self.teacher_tokenizers[model_type].eos_token_id,
                "attention_mask": torch.zeros(bs, max_length),
            } for model_type in self.teacher_tokenizers
        }

        for model_type in self.teacher_tokenizers:
            if model_type in ["gpt2"]:
                teacher_model_data[model_type]["position_ids"] = torch.zeros(
                    bs, max_length, dtype=torch.long
                )

        teacher_no_model_data = {
            model_type: {
                "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
                "loss_mask": torch.zeros(bs, max_length),
            } for model_type in self.teacher_tokenizers
        }

        for i, samp in enumerate(samples):
            self._process_lm(
                i, samp, model_data, no_model_data, gen_data, 
                teacher_model_data, teacher_no_model_data
            )

        for model_type in teacher_model_data:
            prefix = f"teacher_{model_type}_"
            for key in teacher_model_data[model_type]:
                model_data[f"{prefix}{key}"] = teacher_model_data[model_type][key]
                
            for key in teacher_no_model_data[model_type]:
                no_model_data[f"{prefix}{key}"] = teacher_no_model_data[model_type][key]
        
        return model_data, no_model_data, gen_data


class SelfCorrectionDistillDataset(Dataset):
    def __init__(
        self, 
        args, 
        split: str,
        student_tokenizer: Dict[str, AutoTokenizer], 
        teacher_tokenizers: Optional[Dict[str, AutoTokenizer]] = {},
    ):
        self.args = args
        self.split = split
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizers = teacher_tokenizers
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.dataset = self._load_and_process_data()
        self.answers = self._extract_reference_answers()

    def __len__(self):
        return len(self.dataset)
   
    def __getitem__(self, index):
        return self.dataset[index]
    
    def _load_and_process_data(self):
        """
        Expected data format:
        {
            "prompt": "x",
            "initial_response": "y1", 
            "feedback": "f",
            "corrected_output": "y*",
            "original_logits": [...] # Optional: original model's logits for p(y1|x)
        }
        """
        dataset = []
        # For self-correction tasks, use appropriate files
        if hasattr(self.args, 'task') and self.args.task == "self_correction_dskd":
            if self.split == "train":
                filename = "train_stage2.jsonl"  # Use the main self-correction training data
            else:
                filename = f"{self.split}.jsonl"  # Use dev.jsonl/test.jsonl (created from train_stage2.jsonl)
        else:
            filename = f"{self.split}.jsonl"
        path = os.path.join(self.args.data_dir, filename)

        if os.path.exists(path):
            with open(path) as f:
                raw_data = [json.loads(l) for l in f.readlines()]
            
            log_rank("Processing self-correction dataset...")  
            seg = np.iinfo(np.int32).max * 2 + 1        
            for data in tqdm(raw_data, disable=(dist.get_rank() != 0)):
                # Original prompt (x)
                x_ids = self.student_tokenizer.encode(
                    data["prompt"], add_special_tokens=False
                )[:self.max_prompt_length]
                
                # Initial response (y1)
                y1_ids = self.student_tokenizer.encode(
                    data["initial_response"], add_special_tokens=False
                ) + [self.student_tokenizer.eos_token_id]
                
                # Feedback (f)
                f_ids = self.student_tokenizer.encode(
                    data["feedback"], add_special_tokens=False
                )
                
                # Corrected output (y*)
                y_star_ids = self.student_tokenizer.encode(
                    data["corrected_output"], add_special_tokens=False
                ) + [self.student_tokenizer.eos_token_id]
                
                # x* = cat(x, y1, f) - extended prompt
                x_star_ids = x_ids + y1_ids + f_ids
                x_star_ids = x_star_ids[:self.max_prompt_length]
                
                tokenized_data = {
                    # For (y1|x) - original task
                    "original_input_ids": x_ids + [seg] + y1_ids,
                    "original_y1": y1_ids,
                    
                    # For (y*|x*) - corrected task with DSKD+DTW
                    "corrected_input_ids": x_star_ids + [seg] + y_star_ids,
                    "corrected_output": y_star_ids,
                    
                    # For easy access
                    "x_ids": x_ids,
                    "y1_ids": y1_ids, 
                    "x_star_ids": x_star_ids,
                    "y_star_ids": y_star_ids
                }
                
                # Store original model logits if available (for KL regularization)
                if "original_logits" in data:
                    tokenized_data["original_logits"] = data["original_logits"]
        
                # Process teacher tokenizers for DSKD
                for model_type in self.teacher_tokenizers:
                    if self.teacher_tokenizers[model_type] is None: 
                        continue
                        
                    # Teacher tokenization for corrected task
                    t_x_star_ids = self.teacher_tokenizers[model_type].encode(
                        data["prompt"] + data["initial_response"] + data["feedback"], 
                        add_special_tokens=False
                    )[:self.max_prompt_length]
                    
                    t_y_star_ids = self.teacher_tokenizers[model_type].encode(
                        data["corrected_output"], add_special_tokens=False
                    ) + [self.teacher_tokenizers[model_type].eos_token_id]
                    
                    tokenized_data[f"teacher_{model_type}_corrected_input_ids"] = \
                        t_x_star_ids + [seg] + t_y_star_ids

                dataset.append(tokenized_data)
            return dataset
        else:
            raise FileNotFoundError(f"No such file named {path}")

    def _extract_reference_answers(self):
        """Extract corrected outputs as reference answers for ROUGE evaluation"""
        # Use same file selection logic as _load_and_process_data
        if hasattr(self.args, 'task') and self.args.task == "self_correction_dskd":
            if self.split == "train":
                filename = "train_stage2.jsonl"
            else:
                filename = f"{self.split}.jsonl"  # dev.jsonl created from train_stage2.jsonl
        else:
            filename = f"{self.split}.jsonl"
        path = os.path.join(self.args.data_dir, filename)
        
        if os.path.exists(path):
            with open(path) as f:
                raw_data = [json.loads(l) for l in f.readlines()]
            
            # Extract corrected outputs as reference answers
            answers = []
            for data in raw_data:
                corrected_output = data["corrected_output"]
                # Format as expected by evaluation (list of strings)
                if isinstance(corrected_output, list):
                    answers.append(corrected_output)
                else:
                    answers.append([corrected_output])
            
            return answers
        else:
            return []

    def _process_self_correction_data(
        self, i, samp, model_data, no_model_data, gen_data, 
        teacher_model_data, teacher_no_model_data
    ):
        seg = np.iinfo(np.int32).max * 2 + 1
        
        # Process original task (y1|x)
        orig_input_ids = np.array(samp["original_input_ids"])
        orig_source_len = np.where(orig_input_ids == seg)[0][0]
        orig_input_ids = np.concatenate(
            [orig_input_ids[:orig_source_len], orig_input_ids[orig_source_len+1:]], axis=0
        )[:self.max_length]
        orig_input_len = len(orig_input_ids)
        
        model_data["original_input_ids"][i][:orig_input_len-1] = torch.tensor(orig_input_ids[:-1], dtype=torch.long)
        model_data["original_attention_mask"][i][:orig_input_len-1] = 1.0
        if self.args.model_type in ["gpt2"]:
            model_data["original_position_ids"][i][:orig_input_len-1] = torch.arange(0, orig_input_len-1, dtype=torch.long)
        no_model_data["original_label"][i][:orig_input_len-1] = torch.tensor(orig_input_ids[1:], dtype=torch.long)
        no_model_data["original_label"][i][:orig_source_len-1] = -100
        no_model_data["original_loss_mask"][i][:orig_input_len-1] = 1.0
        no_model_data["original_loss_mask"][i][:orig_source_len-1] = 0
        
        # Process corrected task (y*|x*) - for DSKD+DTW
        corr_input_ids = np.array(samp["corrected_input_ids"])
        corr_source_len = np.where(corr_input_ids == seg)[0][0]
        corr_prompt = corr_input_ids[:corr_source_len]
        corr_input_ids = np.concatenate(
            [corr_input_ids[:corr_source_len], corr_input_ids[corr_source_len+1:]], axis=0
        )[:self.max_length]
        corr_input_len = len(corr_input_ids)
        
        model_data["input_ids"][i][:corr_input_len-1] = torch.tensor(corr_input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:corr_input_len-1] = 1.0
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"][i][:corr_input_len-1] = torch.arange(0, corr_input_len-1, dtype=torch.long)
        no_model_data["label"][i][:corr_input_len-1] = torch.tensor(corr_input_ids[1:], dtype=torch.long)
        no_model_data["label"][i][:corr_source_len-1] = -100
        no_model_data["loss_mask"][i][:corr_input_len-1] = 1.0
        no_model_data["loss_mask"][i][:corr_source_len-1] = 0
        
        # Generation data for corrected prompt
        gen_data["input_ids"][i][-len(corr_prompt):] = torch.tensor(corr_prompt, dtype=torch.long)
        gen_data["attention_mask"][i][-len(corr_prompt):] = 1.0
        
        # Store original prompt for KL computation
        x_ids = np.array(samp["x_ids"])[:self.max_prompt_length]
        gen_data["original_prompt_ids"][i][-len(x_ids):] = torch.tensor(x_ids, dtype=torch.long)
        gen_data["original_prompt_mask"][i][-len(x_ids):] = 1.0
        
        # Process original logits if available
        if "original_logits" in samp and "original_logits" in no_model_data:
            orig_logits = torch.tensor(samp["original_logits"], dtype=torch.float)
            # Ensure the logits fit within our max_length
            logits_len = min(orig_logits.shape[0], orig_input_len-1)
            no_model_data["original_logits"][i][:logits_len] = orig_logits[:logits_len]

        # Process teacher data for DSKD
        for model_type in self.teacher_tokenizers:
            if f"teacher_{model_type}_corrected_input_ids" not in samp:
                continue
                
            t_input_ids = np.array(samp[f"teacher_{model_type}_corrected_input_ids"])
            t_source_len = np.where(t_input_ids == seg)[0][0]
            t_input_ids = np.concatenate(
                [t_input_ids[:t_source_len], t_input_ids[t_source_len+1:]], axis=0
            )[:self.max_length]
            t_input_len = len(t_input_ids)
            
            teacher_model_data[model_type]["input_ids"][i][:t_input_len-1] = \
                torch.tensor(t_input_ids[:-1], dtype=torch.long)
            teacher_model_data[model_type]["attention_mask"][i][:t_input_len-1] = 1.0
            if model_type in ["gpt2"]:
                teacher_model_data[model_type]["position_ids"][i][:t_input_len-1] = \
                    torch.arange(0, t_input_len-1, dtype=torch.long)
            teacher_no_model_data[model_type]["label"][i][:t_input_len-1] = \
                torch.tensor(t_input_ids[1:], dtype=torch.long)
            teacher_no_model_data[model_type]["label"][i][:t_source_len-1] = -100
            teacher_no_model_data[model_type]["loss_mask"][i][:t_input_len-1] = 1.0
            teacher_no_model_data[model_type]["loss_mask"][i][:t_source_len-1] = 0

    def collate(self, samples):
        bs = len(samples)
        max_length = self.max_length

        model_data = {
            # For corrected task (DSKD+DTW)
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.student_tokenizer.eos_token_id,
            "attention_mask": torch.zeros(bs, max_length),
            
            # For original task (KL div)
            "original_input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.student_tokenizer.eos_token_id,
            "original_attention_mask": torch.zeros(bs, max_length),
        }
        
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
            model_data["original_position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)
            
        no_model_data = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length),
            "original_label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "original_loss_mask": torch.zeros(bs, max_length),
        }
        
        # Check if any sample has original logits
        has_original_logits = any("original_logits" in samp for samp in samples)
        if has_original_logits:
            # Initialize with zeros, will be filled for samples that have logits
            vocab_size = len(self.student_tokenizer)  # Approximate vocab size
            no_model_data["original_logits"] = torch.zeros(bs, max_length, vocab_size)
        
        gen_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.student_tokenizer.eos_token_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
            "original_prompt_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.student_tokenizer.eos_token_id,
            "original_prompt_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }

        teacher_model_data = {
            model_type: {
                "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.teacher_tokenizers[model_type].eos_token_id,
                "attention_mask": torch.zeros(bs, max_length),
            } for model_type in self.teacher_tokenizers
        }

        for model_type in self.teacher_tokenizers:
            if model_type in ["gpt2"]:
                teacher_model_data[model_type]["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)

        teacher_no_model_data = {
            model_type: {
                "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
                "loss_mask": torch.zeros(bs, max_length),
            } for model_type in self.teacher_tokenizers
        }

        for i, samp in enumerate(samples):
            self._process_self_correction_data(
                i, samp, model_data, no_model_data, gen_data, 
                teacher_model_data, teacher_no_model_data
            )

        # Flatten teacher data into model_data and no_model_data
        for model_type in teacher_model_data:
            prefix = f"teacher_{model_type}_"
            for key in teacher_model_data[model_type]:
                model_data[f"{prefix}{key}"] = teacher_model_data[model_type][key]
                
            for key in teacher_no_model_data[model_type]:
                no_model_data[f"{prefix}{key}"] = teacher_no_model_data[model_type][key]
        
        return model_data, no_model_data, gen_data

    def move_to_device(self, datazip, device):
        """Move tensor data to specified device"""
        for data in datazip:
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)
                elif isinstance(data[k], dict):
                    for kk in data[k]:
                        data[k][kk] = data[k][kk].to(device)
