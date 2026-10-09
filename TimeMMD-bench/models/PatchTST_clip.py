import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
import torch.nn.functional as F


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=4, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride
        if patch_len>configs.pred_len:
            patch_len = configs.pred_len
        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

        if configs.multimodal:
            self.tr_sea = configs.tr_sea
            patch_num = int((self.seq_len - patch_len) / stride + 1)
            # if padding_patch == 'end':  # can be modified to general case
            #     patch_num += 1
            # print('patch num: ', patch_num)
            self.t = torch.tensor(configs.clip_t)
            self.ts_text_attn_trend = Ts_Text_Cross_Attention(d_model_text = configs.llm_dim, d_model_ts = configs.d_model)
            if self.tr_sea:
                self.ts_text_attn_seasonal = Ts_Text_Cross_Attention(d_model_text=configs.llm_dim, d_model_ts=configs.d_model)

            decomp_w = None
            # self.decomp_w = nn.Parameter((torch.tensor([1-decomp_w, 1-decomp_w, decomp_w, decomp_w]) / 100).reshape(4, 1, 1, 1))
            # self.decomp_w = nn.Parameter((torch.tensor([1-decomp_w, 1-decomp_w, 1-decomp_w, decomp_w, decomp_w, decomp_w]) / 10).reshape(6, 1, 1, 1))
            self.decomp_w = nn.Parameter((torch.tensor([0.5]*6) / 10).reshape(6, 1, 1, 1))

            # print('self.decomp_w: ', self.decomp_w)

            self.avg_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)

    def forecast_1(self, x):
        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        # do patching and embedding
        x = x.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def cross_attn(self, x, text_emb):

        text_emb_cross = self.ts_text_attn_trend(x, text_emb)
        if self.tr_sea:
            text_emb_seasonal = self.ts_text_attn_seasonal(x, text_emb)
            text_emb_cross = (text_emb_cross, text_emb_seasonal)

        # print(text_emb, x)
        # print(x, text_emb_trend, text_emb_seasonal)

        return text_emb_cross

    def forecast_clip(self, x, text_emb):
        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        # do patching and embedding
        x = x.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        text_emb = self.cross_attn(enc_out, text_emb)

        return enc_out, text_emb

    def forecast_2(self, x, text_emb):
        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        # do patching and embedding
        x = x.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        text_emb = self.cross_attn(enc_out, text_emb)

        text_emb = text_emb[0] + text_emb[1]
        # Decoder
        dec_out = self.head(0.5 * enc_out + 0.5 * text_emb)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out


    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark, text_emb, forecast = 1):
        if forecast == 1:
            return self.forecast_1(x)
        elif forecast == 0:
            return self.forecast_clip(x, text_emb)
        elif forecast == 2:
            return self.forecast_2(x, text_emb)

    # def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_emb, forecast = 1, mask = None):
    #     if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
    #         dec_out = self.forecast_1(x_enc)
    #         return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    #     if self.task_name == 'imputation':
    #         dec_out = self.imputation(
    #             x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
    #         return dec_out  # [B, L, D]
    #     if self.task_name == 'anomaly_detection':
    #         dec_out = self.anomaly_detection(x_enc)
    #         return dec_out  # [B, L, D]
    #     if self.task_name == 'classification':
    #         dec_out = self.classification(x_enc, x_mark_enc)
    #         return dec_out  # [B, N]
    #     return None



class Attention_text(nn.Module):
    def __init__(self, d_model_text=768, d_model_output = 768, num_heads=8, patch_num=8, nvars=4):
        super().__init__()
        self.patch_num = patch_num
        self.nvars = nvars
        self.d_model_text = d_model_text
        self.d_model_output = d_model_output
        # self.ln = nn.LayerNorm([192, self.d_model_text], eps=1e-6)
        self.query = nn.Parameter(torch.randn(1, self.patch_num * self.nvars, self.d_model_output))
        self.key_proj = nn.Linear(self.d_model_text, self.d_model_output)  # 用于生成 K
        self.value_proj = nn.Linear(self.d_model_text, self.d_model_output)  # 用于生成 V
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.d_model_output, num_heads=num_heads, batch_first=True)

    def forward(self, text):
        """
        text: [bs x seq_len x d_model']
        output: [bs x d_model']
        """
        # 投影生成 Q, K, V
        # text (32, 1024, 768)
        # query (32, self.patch_num * self.nvars, 768) key (32, 1024, 768) value (32, 1024, 768)
        # if torch.isnan(text).any():
        #     print("NaN")

        # text = self.ln(text)
        # print(text)
        # if torch.isnan(text).any():
        #     print("NaN")
        query = self.query.expand(text.shape[0], -1, -1)  # (batch_size, self.patch_num * self.nvars, d_model_output)
        key = self.key_proj(text)  # (batch_size, seq_len * max_length, d_model_output)
        value = self.value_proj(text)  # (batch_size, seq_len * max_length, d_model_output)
        # print('q: ', query, 'k: ', key, 'v: ', value)
        # 前向传播
        # query (32, 2, 768)
        output, _ = self.attention_layer(query, key, value)
        # print(text, output)

        # output = self.mlp(text.transpose(1,2)).transpose(1,2)
        return output

class Ts_Text_Cross_Attention(nn.Module):
    def __init__(self, d_model_text = 2048, d_model_ts = 16, num_heads = 8, patch_num = 8, nvars = 4):
        super().__init__()
        self.d_model_text = d_model_text
        self.d_model_ts = d_model_ts
        self.patch_num = patch_num
        self.nvars = nvars
        self.query_proj = nn.Linear(self.d_model_text, self.d_model_text) # 用于生成 Q
        self.key_proj = nn.Linear(self.d_model_text, self.d_model_text)  # 用于生成 K
        self.value_proj = nn.Linear(self.d_model_text, self.d_model_text)  # 用于生成 V
        self.attention_text = Attention_text(d_model_text=self.d_model_text, d_model_output=self.d_model_text, num_heads=8, patch_num=self.patch_num,nvars=self.nvars)
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.d_model_text, num_heads=num_heads, batch_first=True)

        mlp_sizes_before = [self.d_model_ts, int((self.d_model_text + self.d_model_ts) / 2), self.d_model_text]
        self.mlp_before = MLP(mlp_sizes_before,)
        mlp_sizes_after = [self.d_model_text, int((self.d_model_text + self.d_model_ts) / 2), self.d_model_ts]
        self.mlp_after = MLP(mlp_sizes_after)

    def forward(self, ts, text):
        """
        # ts: [bs x nvars x d_model x patch_num]
        # text: [bs x (nvars x patch_num) x d_model]
        # output: [bs x nvars x patch_num x d_model]
        """
        # 投影生成 Q, K, V
        bs, nvars, d_model, patch_num = ts.shape # 32*1*128*2
        ts = self.mlp_before(ts.transpose(-1,-2)).transpose(-1,-2)
        ts = ts.transpose(-1,-2).reshape((bs, -1, self.d_model_text)) # 32, n*p, 768

        bs, words_all, d_model_text = text.shape
        # print(text)
        text = self.attention_text(text) # (32, 2, 768)

        # mean_ts = ts.mean(dim = -1)
        # std_ts = ts.std(dim = -1, unbiased = False)
        # text =
        # print(text)
        scale_factor = torch.std(ts) / torch.std(text)
        offset = torch.mean(ts) - scale_factor * torch.mean(text)

        text = text * scale_factor + offset
        # print(ts, text)
        # here
        query = self.query_proj(ts)  # (batch_size, self.patch_num * self.nvars, d_model_ts)
        key = self.key_proj(text)  # (batch_size, seq_len * max_length, d_model_ts)
        value = self.value_proj(text)  # (batch_size, seq_len * max_length, d_model_ts)

        output, _ = self.attention_layer(query, key, value)
        output = output.reshape((bs, nvars, patch_num, self.d_model_text)).transpose(-1,-2)


        output = self.mlp_after(output.transpose(-1,-2)).transpose(-1,-2)
        # print(ts,text,output)
        return output

class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

