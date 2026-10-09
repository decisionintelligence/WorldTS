import numpy as np
import torch
import torch.nn as nn
from torch import optim

# from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
# from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from utils.rev_in import RevIn

class Model(nn.Module):
    
    def __init__(self, configs, patch_size = 16, stride=8):
        super().__init__()
        # self.is_gpt = configs.is_gpt
        self.revin = False
        if patch_size > configs.pred_len :
            self.patch_size = configs.pred_len
        else:
            self.patch_size = patch_size
        # self.pretrain = configs.pretrain
        self.stride = stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        self.gpt_layers = 6
        # if configs.is_gpt:
        self.gpt2_config = GPT2Config.from_pretrained('language_model/openai-community/gpt2')

        # self.gpt2_config.num_hidden_layers = configs.llm_layers
        self.gpt2_config.output_attentions = True
        self.gpt2_config.output_hidden_states = True

        self.gpt2 = GPT2Model.from_pretrained('language_model/openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=True,
                config=self.gpt2_config,
            )

        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        print("gpt2 = {}".format(self.gpt2))

        self.relu = nn.ReLU()
        self.in_layer = nn.Linear(self.patch_size, configs.d_model)
        self.prompt_layer = nn.Linear(configs.d_model, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * (self.patch_num), configs.pred_len)
        
        # if configs.freeze and configs.pretrain:
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer, self.prompt_layer):
            layer.to(device=configs.gpu)
            layer.train()
        
        self.device = configs.gpu
        self.rev_in = RevIn(num_features=1) # channel independent

    def get_emb(self, x, tokens=None):
        if tokens is None:
            x = self.gpt2(inputs_embeds=x).last_hidden_state
            return x
        else:
            [a, b, c] = x.shape
            # print(x.shape, tokens.shape)
            tokens = tokens.to(self.prompt_layer.weight.device)
            prompt_x = self.relu(self.prompt_layer(tokens))
            x_all = torch.cat((prompt_x, x), dim=1)
            x = self.gpt2(inputs_embeds=x_all).last_hidden_state
            return x[: , -b:, :]

    def get_patch(self, x):
        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x) # b, 1, seq_len
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) #b, 1, patch_num, patch_size
        x = rearrange(x, 'b m n p -> (b m) n p') # b, patch_num, patch_size

        return x

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, summary):
        summary = rearrange(summary, 'b l m -> b m l') # [b, 768, 15]
        summary = self.padding_patch_layer(summary) # [b, 768, 19] 
        summary = summary.unfold(dimension=-1, size=self.patch_size, step=self.stride) # [b, 768, 3, 8]
        summary = summary.mean(dim=-1) # [b, 768, 3]
        print(summary.shape)
        summary = rearrange(summary, 'b l m -> b m l') # [b, 3, 768]

        B, L, M = x.shape # 4, 512, 1
        # print(x.shape)

        if self.revin:
            x = self.rev_in(x, 'norm').to(self.device)
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach()
            x /= stdev
       
        x = self.get_patch(x)
        x = self.in_layer(x)

        outputs = self.get_emb(x, summary)
        # print(outputs.shape, B,M)
        outputs = self.out_layer(outputs.reshape(B*M, -1)) 
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        
        if self.revin:
            outputs = self.rev_in(outputs, 'denorm').to(self.device)
        else:
            outputs = outputs * stdev
            outputs = outputs + means

        return outputs
