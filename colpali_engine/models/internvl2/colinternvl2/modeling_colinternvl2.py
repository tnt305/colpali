from typing import ClassVar, List, Optional
from typing import Any, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers.models.qwen2_vl import Qwen2VLConfig, Qwen2VLForConditionalGeneration
from .internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel, InternVLChatConfig
import math

class ColInternVL2(InternVLChatModel):
    """
    ColInternVL2 model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    """

    # main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: InternVLChatConfig):
        super().__init__(config=config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.language_model.model.config.hidden_size, self.dim ) #, bias=False)
        self.padding_side = "left"
        self.img_context_token_id = 151667
        # self.post_init()
        self.init_linear()
    
    def init_linear(self): 
        print(self.language_model.model.embed_tokens.weight)
        stdv = 1. / math.sqrt(self.custom_text_proj.weight.size(1))
        self.custom_text_proj.weight.data = self.custom_text_proj.weight.data.uniform_(-stdv, stdv)
        if self.custom_text_proj.bias is not None:
            self.custom_text_proj.bias.data = self.custom_text_proj.bias.data.uniform_(-stdv, stdv)


    def forward(
            self,
            pixel_values: torch.FloatTensor = None,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            statistics: Optional[torch.LongTensor] = None,
            loss_weight: Optional[List] = None,
            loss_reduction_all_gather: Optional[bool] = False,
            **kwargs
    ) -> torch.Tensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        B, N, C = input_embeds.shape
        
        if pixel_values is not None:
            
            pixel_values = pixel_values.type(self.vision_model.embeddings.patch_embedding.weight.dtype)
            vit_embeds = self.extract_feature(pixel_values)
            # image_flags = image_flags.squeeze(-1)
            # vit_embeds = vit_embeds[image_flags == 1]
            vit_batch_size = pixel_values.shape[0]
            
            input_embeds = input_embeds.reshape(B * N, C)
            
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
                if statistics is not None:
                    num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
                    self.num_samples += num_samples
                    print(f'total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}')

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            try:
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
                ignore_flag = False
            except Exception as e:
                
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}')
                n_token = selected.sum()
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
                ignore_flag = True
    
            input_embeds = input_embeds.reshape(B, N, C)
        
        outputs = self.language_model.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        
        last_hidden_states = outputs[0].type(self.custom_text_proj.weight.dtype)
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        proj = proj * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)
        return proj

    
    @property
    def get_patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
