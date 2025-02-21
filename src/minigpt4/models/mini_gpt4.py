import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import sys

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer



@registry.register_model("mini_gpt4")
class MiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "../configs/minigpt4.yaml", # "configs/models/minigpt4.yaml",
    }

    def __init__(
        self,
        llama_model="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading LLAMA')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.esm_struct_llama_proj = nn.Linear(
            512, self.llama_model.config.hidden_size
        )

        self.esm_seq_llama_proj = nn.Linear(
            # 1280, self.llama_model.config.hidden_size
            2560, self.llama_model.config.hidden_size
        )
        
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        
        self.prompt_template = prompt_template

        
    def encode_protein_struct(self, protein_struct_encode):
        device = protein_struct_encode.device
        protein_embeds = protein_struct_encode.to(device)

        # input llama shape: [B, 32, 5120]
        inputs_llama = self.esm_struct_llama_proj(protein_embeds.squeeze(dim=2))
        # atts_llama shape: [B, 32]
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
        return inputs_llama, atts_llama

    def encode_protein_seq(self, protein_seq_encode):
        device = protein_seq_encode.device
        protein_embeds = protein_seq_encode.to(device)

        # input llama is of shape [B, 32, 5120]
        inputs_llama = self.esm_seq_llama_proj(protein_embeds.squeeze(dim=2))
        # atts_llama is of shape [B, 32]
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<proteinHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            # print(p_before_embeds.shape, img_embeds.shape, p_after_embeds.shape)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img
    
    def forward(self, samples):
        # structure
        pdb_encode = samples["pdb_encoder_out"]
        pdb_device = pdb_encode.device
        pdb_encode = pdb_encode[0]
        pdb_encode = pdb_encode.permute(1, 0, 2) # Reshape [X, 1, Y] -> [1, X, Y]
        pdb_embeds, atts_pdb = self.encode_protein_struct(pdb_encode)

        # sequence
        seq_encode = samples["seq_encoder_out"]
        seq_device = seq_encode.device
        seq_encode = seq_encode[0]
        seq_embeds, atts_seq = self.encode_protein_seq(seq_encode)

        img_embeds = torch.cat([pdb_embeds, seq_embeds], dim=1)
        atts_img = torch.cat([atts_pdb, atts_seq], dim=1)

        # skips over this branch for stage 1 and 2 
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            vqa_prompt = '###Human: <protein><proteinHere></protein> '
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        # TO check: print out when needed (run stage 2 and print out some stuff to see which branch it goes to)
        elif "q_input" in samples: # prompt path (alignment.txt provided) then takes this path to random choose form the list
            prompt = self.prompt_template.format("<protein><proteinHere></protein> " + samples["q_input"][0])
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

        # stage 1 directly skip the branches above

        self.llama_tokenizer.padding_side = "right"

        text = []
        if "q_input" in samples: 
            text = [t + self.end_sym for t in samples["a_input"]]
        else: 
            text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(pdb_device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(pdb_device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id

        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        return {"loss": loss} 

    @classmethod
    def from_config(cls, cfg):

        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")



        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_protein_encoder = cfg.get("freeze_protein_encoder", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
        
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        model = cls(
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )


        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
