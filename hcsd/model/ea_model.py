import copy
import json
import time

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM


from .utils import *
from .kv_cache import initialize_past_key_values
from .kv_cache_partial_offload import initialize_past_key_values as initialize_past_key_values_partial_offload

from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download
from .cnets import Model
from .cnets_eagle import Model as Model_eagle
from .cnets_tiny_sheared import Model as Model_tiny_sheared
from .configs import EConfig
from huggingface_hub import hf_hub_download

class EaModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            draft_model_type,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            ngl,
            partial_offload,
            partial_offload_SD
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path,use_fast=False)
        
        self.ngl = ngl
        self.partial_offload = partial_offload
        self.partial_offload_SD = partial_offload_SD

        if not partial_offload:
            print("partial_offload",partial_offload)
            print("partial_offload_SD",partial_offload_SD)
            config = EConfig.from_pretrained(ea_model_path)
            with open(ea_model_path,"r") as f:
                con=json.loads(f.read())
            try:
                bias=con["bias"]
            except:
                bias=True
            # self.ea_layer = Model(config,bias=bias,total_tokens=total_token,depth=depth,top_k=top_k,threshold=threshold)
            if draft_model_type == "eagle":
                self.ea_layer = Model_eagle(config,bias=bias,path=base_model_name_or_path,total_tokens=total_token,depth=depth,top_k=top_k,threshold=threshold)
            elif draft_model_type == "tinyllama" or draft_model_type == "shearedllama" :
                self.ea_layer = Model_tiny_sheared(config,bias=bias,total_tokens=total_token,depth=depth,top_k=top_k,threshold=threshold)
            else:
                self.ea_layer = Model(config,bias=bias,path=base_model_name_or_path,total_tokens=total_token,depth=depth,top_k=top_k,threshold=threshold)
      

        # device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        # if device!=base_model.lm_head.weight.device:
        #     self.ea_layer.diff_device = True
        #     if not low_memory:
        #         # self.ea_layer.head=nn.Linear(base_model.lm_head.in_features,base_model.lm_head.out_features,bias=False)
        #         # self.ea_layer.head.weight=copy.deepcopy(base_model.lm_head.weight)
        #         # self.ea_layer.head.to(device)
        #         self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
        #     else:
        #         self.ea_layer.layer_device = device

        # else:
        #     self.ea_layer.diff_device = False
        
        # self.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
        
            self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
             # ##########
            self.ea_layer.to(dtype=torch.float16, device="cuda:0")
            #############
            self.ea_layer.init_tree()
        low_memory=False

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            draft_model_type=None,
            partial_offload=False,
            partial_offload_SD=False,
            ngl=0,
            total_token=59,
            depth=5,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        #assert Type=="LLaMA" or "Mixtral"
        Type=AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type=='LlamaForCausalLM':

            if partial_offload or partial_offload_SD:
                print("partial_offload_SD",partial_offload_SD)
                base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs)

                base_model.model.embed_tokens.to(dtype=torch.float16,device='cuda:0')
                for i in range(0,ngl):
                    base_model.model.layers[i].to(dtype=torch.float16,device='cuda:0')

            else :
                base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs)

        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )


        configpath=os.path.join(ea_model_path,"config.json")
        # if not os.path.exists(configpath):
        #     configpath = hf_hub_download(ea_model_path, "config.json")
        # load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
        # if not os.path.exists(load_model_path):
        #     load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
        # ea_layer_state_dict = torch.load(load_model_path,
        #                                  map_location="cpu")
        # print("base_model.device",base_model.device)
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")
        try:
            load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location="cpu")
            new_state_dict = {}
            
            for key, value in ea_layer_state_dict.items():
                new_key = key.replace("model.", "") if key.startswith("model.") else key
                new_state_dict[new_key] = value
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
            new_state_dict = {}
            
            for key, value in ea_layer_state_dict.items():
                new_key = key.replace("model.", "") if key.startswith("model.") else key
                new_state_dict[new_key] = value

        model = cls(
            base_model,
            base_model_path,
            draft_model_type,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            # ea_layer_state_dict
            new_state_dict,
            ngl,
            partial_offload,
            partial_offload_SD

        )

        if total_token==-1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans=[40,48,50,56,60]
            x=[1,1.05,1.07,1.1,1.13]
            times=[]

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                # torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    # torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    # torch.cuda.synchronize()
                # torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token=cans[times.index(min(times))]
            model.ea_layer.total_tokens=total_token-1
            print(total_token)




        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
                ngl = self.ngl
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length=max_length-self.ea_layer.total_tokens-10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        if self.partial_offload_SD:
            # Initialize the past key and value states
            if hasattr(self, "past_key_values"):
                past_key_values = self.past_key_values
                past_key_values_data = self.past_key_values_data
                current_length_data = self.current_length_data
                # Reset the past key and value states
                current_length_data.zero_()
            else:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values_partial_offload(self.base_model)
                self.past_key_values = past_key_values
                self.past_key_values_data = past_key_values_data
                self.current_length_data = current_length_data
        
        else:
            # Initialize the past key and value states
            if hasattr(self, "past_key_values"):
                past_key_values = self.past_key_values
                past_key_values_data = self.past_key_values_data
                current_length_data = self.current_length_data
                # Reset the past key and value states
                current_length_data.zero_()
            else:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model)
                self.past_key_values = past_key_values
                self.past_key_values_data = past_key_values_data
                self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        
        
        draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0

        for idx in range(max_length):
            
            #with Timer("all"):
            ######################

            if not self.partial_offload_SD :
                retrieve_indices = retrieve_indices.to(input_ids.device)
                tree_mask = tree_mask.to(input_ids.device)
                tree_position_ids = tree_position_ids.to(input_ids.device)
                draft_tokens=draft_tokens.to(input_ids.device)

            ######################
            

            draft_tokens=draft_tokens.to(input_ids.device)

            self.base_model.model.tree_mask = tree_mask

            #with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
                self.partial_offload_SD,
            )
            
            

            draft_tokens=torch.cat((draft_tokens,padding),dim=1)
            candidates=draft_tokens[0,retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            #with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx
