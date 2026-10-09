from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, EarlyStopping_Clip, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from transformers import AutoConfig, AutoModel, AutoTokenizer,LlamaForCausalLM
import datetime
from datetime import datetime, timedelta
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
import pandas as pd
from datetime import datetime
import re
import json

warnings.filterwarnings('ignore')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

class Exp_Long_Term_Forecast_Clip(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Clip, self).__init__(args)
        configs=args
        # self.text_path=configs.text_path
        # self.prompt_weight=configs.prompt_weight
        # self.attribute="final_sum"
        # self.type_tag=configs.type_tag
        # self.text_len=configs.text_len
        self.d_llm = configs.llm_dim
        self.pred_len=configs.pred_len
        # self.text_embedding_dim = configs.text_emb
        # self.pool_type=configs.pool_type
        # self.use_fullmodel=configs.use_fullmodel
        # self.hug_token=configs.huggingface_token
        # mlp_sizes=[self.d_llm,int(self.d_llm/8),self.text_embedding_dim]
        self.Doc2Vec=False
        if configs.llm_model == 'Doc2Vec':
            print('Now using Doc2Vec')
            print("Training Doc2Vec model")

            # from gensim.test.utils import common_texts
            # from gensim.test.utils import common_texts
            # from gensim.models.doc2vec import Doc2Vec, TaggedDocument
            # def read_csv_column(file_path, column_name):
            #     df = pd.read_csv(file_path)
            #
            #     column_data = df[column_name].replace('', np.nan).fillna('null')
            #
            #     return column_data.to_list()
            # result  = read_csv_column(file_path=os.path.join(configs.root_path,
            #                               configs.data_path), column_name='Final_Search_4')
            # train_len=int(len(result)*0.8)
            # documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(result[:train_len])]
            # text_model = Doc2Vec(documents, vector_size=configs.llm_dim, window=2, min_count=1, workers=4)
            # self.text_model=text_model
            # self.Doc2Vec=True
        else:
            if configs.llm_model == 'LLAMA2':
                # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
                self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
                self.llama_config.num_hidden_layers = configs.llm_layers
                self.llama_config.output_attentions = True
                self.llama_config.output_hidden_states = True
                try:
                    self.llm_model = LlamaModel.from_pretrained(
                        # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llama_config,
                        # load_in_4bit=True
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = LlamaModel.from_pretrained(
                        # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.llama_config,
                        # load_in_4bit=True
                    )
                try:
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                        'huggyllama/llama-7b',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'LLAMA3':
                # Automatically load the configuration, model, and tokenizer for LLaMA-3-8B
                llama3_path = "meta-llama/Meta-Llama-3-8B-Instruct"
                cache_path = "./"

                # Load the configuration with custom adjustments
                self.config =  LlamaConfig.from_pretrained(llama3_path,token=self.hug_token,cache_dir=cache_path)

                self.config.num_hidden_layers = configs.llm_layers
                self.config.output_attentions = True
                self.config.output_hidden_states = True

                self.llm_model  = LlamaModel.from_pretrained(
                    llama3_path,
                    config=self.config,
                    token=self.hug_token,cache_dir=cache_path
                )
                self.tokenizer = AutoTokenizer.from_pretrained(llama3_path,use_auth_token=self.hug_token,cache_dir=cache_path)
            elif configs.llm_model == 'GPT2':
                self.gpt2_config = GPT2Config.from_pretrained('language_model/openai-community/gpt2')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'language_model/openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'language_model/openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'language_model/openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'language_model/openai-community/gpt2',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'GPT2M':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-medium')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-medium',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'GPT2L':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-large')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-large',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'GPT2XL':
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2-xl')

                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True
                try:
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.gpt2_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                    )

                try:
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        'openai-community/gpt2-xl',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            elif configs.llm_model == 'BERT':
                self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

                self.bert_config.num_hidden_layers = configs.llm_layers
                self.bert_config.output_attentions = True
                self.bert_config.output_hidden_states = True
                try:
                    self.llm_model = BertModel.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.bert_config,
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = BertModel.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.bert_config,
                    )

                try:
                    self.tokenizer = BertTokenizer.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except EnvironmentError:  # downloads the tokenizer from HF if not already done
                    print("Local tokenizer files not found. Atempting to download them..")
                    self.tokenizer = BertTokenizer.from_pretrained(
                        'google-bert/bert-base-uncased',
                        trust_remote_code=True,
                        local_files_only=False
                    )
            
            else:
                raise Exception('LLM model is not defined')

            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

            for param in self.llm_model.parameters():
                param.requires_grad = False
            self.llm_model=self.llm_model.to(self.device)
        # if args.init_method == 'uniform':
        #     self.weight1 = nn.Embedding(1, self.args.pred_len)
        #     self.weight2 = nn.Embedding(1, self.args.pred_len)
        #     nn.init.uniform_(self.weight1.weight)
        #     nn.init.uniform_(self.weight2.weight)
        #     self.weight1.weight.requires_grad = True
        #     self.weight2.weight.requires_grad = True
        # elif args.init_method == 'normal':
        #     self.weight1 = nn.Embedding(1, self.args.pred_len)
        #     self.weight2 = nn.Embedding(1, self.args.pred_len)
        #     nn.init.normal_(self.weight1.weight)
        #     nn.init.normal_(self.weight2.weight)
        #     self.weight1.weight.requires_grad = True
        #     self.weight2.weight.requires_grad = True
        # else:
        #     raise ValueError('Unsupported initialization method')

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.llm_model, self.tokenizer)
        return data_set, data_loader

    def _select_optimizer(self, learning_rate):
        params_to_update = self.model.parameters()
        # 创建一个包含要更新参数的列表
        param_groups = []

        # 遍历要更新的参数
        for param in params_to_update:
            # 检查参数是否属于 self.model.decomp_w
            if self.args.multimodal and param is self.model.decomp_w:
                # 为 self.model.decomp_w 设置特定的学习率
                param_groups.append({'params': [param], 'lr': 0.0001})
            else:
                # 其他参数使用默认的学习率
                param_groups.append({'params': [param], 'lr': learning_rate})

        #    定义优化器，使用参数组来指定不同的学习率
        model_optim = optim.Adam(param_groups)
        return model_optim

    def _select_optimizer_clip(self):
        all_params = self.model.parameters()

        # 获取 self.model.model 的参数（你不想更新的部分）
        model_params1 = set(self.get_model(self.model).patch_embedding.parameters())
        model_params2 = set(self.get_model(self.model).encoder.parameters())
        model_params = model_params1.union(model_params2)

        # 使用 list comprehension 筛选出你想要更新的参数
        # 即所有参数减去 self.model.model 的参数
        params_to_update = [param for param in all_params if param not in model_params]

        # 定义优化器，只更新 params_to_update 中的参数
        model_optim = optim.Adam(params_to_update, lr=self.args.learning_rate_clip)
        return model_optim
    def _select_criterion(self, forecast = 1):
        if forecast != 0:
            criterion = nn.MSELoss()
        else:
            if self.args.tr_sea:
                criterion = self.contrastiveLoss_tr_sea
            else:
                criterion = self.contrastiveLoss
        return criterion

    def vali(self, vali_data, vali_loader, criterion, forecast = 1):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            if forecast == 2 and self.args.decomp_w != 0:
                text_pred = vali_data.get_text_pred()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,index) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if forecast == 1:
                    prompt_emb = None
                else:
                    batch_text = vali_data.get_text(index)
                    prompt_emb = self.get_text_emb(batch_x, batch_text)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_emb, forecast = forecast)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_emb, forecast = forecast)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_emb, forecast = forecast)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_emb, forecast = forecast)


                if forecast == 0:
                    loss = criterion(outputs[0], outputs[1], self.get_model(self.model).t)
                else:
                    if forecast == 2 and self.args.decomp_w != 0:
                        outputs = self.correction(outputs, text_pred, index)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        # self.train_forecast(setting, forecast=2)
        if self.args.multimodal:
            path_2 = os.path.join(self.args.checkpoints, setting) + '/clip/checkpoint.pth'
            if not os.path.exists(path_2):
                print('clip not exist, can not skip')
                self.train_forecast(setting, forecast = 1)
                self.train_forecast_clip(setting, forecast = 0)
            self.train_forecast(setting, forecast = 2)
        else:
            self.train_forecast(setting, forecast=1)

    def train_forecast(self, setting, forecast = 1):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        if forecast == 2 and self.args.decomp_w != 0:
            text_pred_train = train_data.get_text_pred()


        path = os.path.join(self.args.checkpoints, setting)
        if forecast == 2:
            path = os.path.join(self.args.checkpoints, setting) + '/decomp_w_' + str(self.args.decomp_w).split('.')[-1]
            path_clip = os.path.join(self.args.checkpoints, setting) + '/clip/checkpoint.pth'
            # self.model.load_state_dict(torch.load(path_clip))
            self.model.load_state_dict(torch.load(path_clip, map_location=self.device))
            if self.args.decomp_w != 0:
                decomp_w = self.args.decomp_w
                self.model.decomp_w.data = (torch.tensor([1-decomp_w, 1-decomp_w, 1-decomp_w, decomp_w, decomp_w, decomp_w])).reshape(6, 1, 1, 1).to(self.model.decomp_w.device)

        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        learning_rate = self.args.learning_rate if forecast == 1 else self.args.learning_rate_corr
        model_optim = self._select_optimizer(learning_rate)
        criterion = self._select_criterion(forecast = forecast)
        train_epochs = self.args.train_epochs if forecast == 1 else self.args.train_epochs_corr

        if self.args.if_lradj:
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=train_epochs,
                                                max_lr=learning_rate)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,index) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                prompt_emb = None
                if forecast == 2:
                    batch_text = train_data.get_text(index)
                    prompt_emb = self.get_text_emb(batch_x, batch_text)
                else:
                    prompt_emb = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_emb, forecast = forecast)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_emb, forecast = forecast)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_emb, forecast = forecast)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_emb, forecast = forecast)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                if forecast == 2 and self.args.decomp_w != 0:
                    outputs = self.correction(outputs, text_pred_train, index).float()


                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, forecast = forecast)
            test_loss = self.vali(test_data, test_loader, criterion, forecast = forecast)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)

            if self.args.if_lradj:
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                scheduler.step()

            if early_stopping.early_stop:
                print("Early stopping")
                break

            #adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def train_forecast_clip(self, setting, forecast = 0):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting) + "/clip/"
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping_Clip(patience=self.args.patience, verbose=True)
        criterion = self._select_criterion(forecast = forecast)
        model_optim = self._select_optimizer_clip()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs_clip):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                batch_text = train_data.get_text(index)
                prompt_emb = self.get_text_emb(batch_x, batch_text)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_emb, forecast=forecast)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_emb, forecast=forecast)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_emb, forecast=forecast)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_emb, forecast=forecast)


                loss = criterion(outputs[0], outputs[1], self.get_model(self.model).t)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | clip_loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, forecast=forecast)
            test_loss = self.vali(test_data, test_loader, criterion, forecast=forecast)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        early_stopping.save_checkpoint(path)

            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0, forecast = 2):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            path = os.path.join(self.args.checkpoints, setting) + '/decomp_w_' + str(self.args.decomp_w).split('.')[-1]
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            if forecast == 2 and self.args.decomp_w != 0:
                text_pred_test = test_data.get_text_pred()
            # start_time  = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,index) in enumerate(test_loader):
                # if i == 10:
                #     end_time = time.time()
                #     break
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_text=test_data.get_text(index)


                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if forecast == 2:
                    batch_text = test_data.get_text(index)
                    prompt_emb = self.get_text_emb(batch_x, batch_text)
                else:
                    prompt_emb = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_emb, forecast=forecast)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_emb, forecast=forecast)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_emb, forecast=forecast)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt_emb, forecast=forecast)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                #outputs=(1-self.prompt_weight)*outputs+self.prompt_weight*prompt_y
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]

                if forecast == 2 and self.args.decomp_w != 0:
                    outputs = self.correction(outputs, text_pred_test, index)

                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                # if test_data.scale and self.args.inverse:
                #     shape = outputs.shape
                #     outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                #     batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                #
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # preds = np.array(preds)
        # trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # print('time: ', (end_time - start_time)/10)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        
        dtw = -999
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, dtw:{}'.format(mse, mae, rmse, dtw))
        f = open(self.args.save_name, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return mse

    def get_model(self, model):
        "Return the model maybe wrapped inside `model`."
        return model.module if isinstance(model, nn.DataParallel) else model

    def get_text_emb(self, batch_ts, batch_text):
        summary_Template = """- Complete data: {ot_values}
        - Start value: {start_value:.2f}
        - End value: {end_value:.2f}
        - Maximum fluctuation: {max_delta:.2f} (occurred at hour {max_pos})
        - Trend slope: {trend:.2f}/h
        - Standard deviation: {std:.2f}"""
        bs, seq_len, channel = batch_ts.shape
        batch_ts = np.array(batch_ts.to('cpu'))
        prompt = []
        for i in range(bs):
            seq_x_value = batch_ts[i,:].reshape(-1)
            # print(batch_ts.shape, seq_x_value.shape)
            stats = {
                "start_value": seq_x_value[0],
                "end_value": seq_x_value[-1],
                "max_delta": np.max(np.abs(np.diff(seq_x_value))),
                "max_pos": np.argmax(np.abs(np.diff(seq_x_value))) + 1,
                "trend": np.polyfit(range(seq_len), seq_x_value, 1)[0],
                "std": np.std(seq_x_value)
            }

            seq_text = "<|start_prompt|Make predictions about the future based on the following information:" + summary_Template.format(ot_values=[f"{v:.2f}" for v in seq_x_value],**stats) + "<|<end_prompt>|>"
            prompt.append(seq_text)
        max_length = 192
        # prompt = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True,
        #                         max_length=max_length).input_ids
        # prompt_embeddings = self.llm_model.get_input_embeddings()(
        #     prompt.to(self.device))  # (batch, prompt_token, dim)

        prompt = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True,
                                max_length=max_length)
        prompt_emb = self.llm_model(
            input_ids=prompt['input_ids'].to(self.device),attention_mask=prompt['attention_mask'].to(self.device)).last_hidden_state  # (batch, prompt_token, dim)

        # print(prompt_embeddings.shape)
        # prompt_emb = self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
        # print(prompt_emb.shape)

        return prompt_emb

    # def correction(self, ts_pred, text_pred, batch_idx):
    #     avg_pool = self.get_model(self.model).avg_pool
    #     tt = [text_pred[i].reshape(1,-1) for i in batch_idx]
    #     text_pred = (torch.cat(tt, dim=0).to(ts_pred.device).unsqueeze(-1).transpose(-1, -2))
    #     ts_pred = ts_pred.transpose(-1, -2)
    #
    #     ts_pred_trend = avg_pool(ts_pred)
    #     ts_pred_seasonal = (ts_pred - ts_pred_trend)
    #     text_pred_trend = avg_pool(text_pred)
    #     text_pred_seasonal = (text_pred - text_pred_trend)
    #
    #     stacked_ts = torch.stack([ts_pred_trend, ts_pred_seasonal, text_pred_trend, text_pred_seasonal], dim=0)  # 在第 1 维堆叠
    #
    #     # weights = torch.softmax(self.model.decomp_w * 100, dim=0).view(-1, 1, 1, 1)
    #     weights = self.model.decomp_w.view(-1, 1, 1, 1) * 100
    #
    #     # 在堆叠的维度上求和
    #     ts = torch.sum(stacked_ts * weights, dim=0)  # 在第 1 维求和
    #
    #     return ts.transpose(-1, -2)

        # 频域分解函数
    def frequency_decomposition(self, signal, low_freq_ratio=0.1, high_freq_ratio=0.2):
        """
        对信号进行频域分解
        Args:
            signal: 输入信号 [batch, features, seq_len]
            low_freq_ratio: 低频分量占总频率的比例
            high_freq_ratio: 高频分量占总频率的比例
        Returns:
            low_freq: 低频分量
            high_freq: 高频分量
            mid_freq: 其余分量（中频）
        """
        # FFT变换到频域 - 在序列维度(dim=2)上进行
        fft_signal = torch.fft.rfft(signal, dim=2)
        seq_len = signal.shape[2]
        freq_len = fft_signal.shape[2]

        # 计算频率分割点 - 确保至少有1个频率点
        low_cutoff = max(1, int(freq_len * low_freq_ratio))
        high_cutoff = min(freq_len - 1, int(freq_len * (1 - high_freq_ratio)))

        # 确保 low_cutoff < high_cutoff
        if low_cutoff >= high_cutoff:
            high_cutoff = min(freq_len, low_cutoff + 1)

        # 创建频域掩码
        low_mask = torch.zeros_like(fft_signal)
        high_mask = torch.zeros_like(fft_signal)
        mid_mask = torch.zeros_like(fft_signal)

        # 低频分量（保留低频部分）- 正确的索引方式
        low_mask[:, :, :low_cutoff] = 1

        # 高频分量（保留高频部分）
        high_mask[:, :, high_cutoff:] = 1

        # 中频分量（保留中间频率部分）
        mid_mask[:, :, low_cutoff:high_cutoff] = 1

        # 分离各频率分量
        low_fft = fft_signal * low_mask
        high_fft = fft_signal * high_mask
        mid_fft = fft_signal * mid_mask

        # 变换回时域
        low_freq = torch.fft.irfft(low_fft, n=seq_len, dim=2)
        high_freq = torch.fft.irfft(high_fft, n=seq_len, dim=2)
        mid_freq = torch.fft.irfft(mid_fft, n=seq_len, dim=2)

        return low_freq, high_freq, mid_freq

    def frequency_decomposition_debug(self, signal, low_freq_ratio=0.1, high_freq_ratio=0.3):
        """调试版本的频域分解"""
        print(f"Input signal shape: {signal.shape}")
        print(f"Signal stats: mean={signal.mean():.6f}, std={signal.std():.6f}")

        # FFT变换到频域
        fft_signal = torch.fft.rfft(signal, dim=2)
        seq_len = signal.shape[2]
        freq_len = fft_signal.shape[2]

        print(f"FFT shape: {fft_signal.shape}, freq_len: {freq_len}")

        # 计算频率分割点 - 确保至少有1个频率点
        low_cutoff = max(1, int(freq_len * low_freq_ratio))
        high_cutoff = min(freq_len - 1, int(freq_len * (1 - high_freq_ratio)))

        # 确保 low_cutoff < high_cutoff
        if low_cutoff >= high_cutoff:
            high_cutoff = min(freq_len, low_cutoff + 1)

        print(f"Cutoffs - low: {low_cutoff}, high: {high_cutoff}")
        print(
            f"Frequency ranges - low: [0:{low_cutoff}], mid: [{low_cutoff}:{high_cutoff}], high: [{high_cutoff}:{freq_len}]")

        # 检查各频段的能量 - 使用正确的索引
        low_energy = torch.sum(torch.abs(fft_signal[:, :, :low_cutoff]) ** 2)
        print(f"Low freq energy: {low_energy:.4f}")

        high_energy = torch.sum(torch.abs(fft_signal[:, :, high_cutoff:]) ** 2)
        print(f"High freq energy: {high_energy:.4f}")

        mid_energy = torch.sum(torch.abs(fft_signal[:, :, low_cutoff:high_cutoff]) ** 2)
        print(f"Mid freq energy: {mid_energy:.4f}")

        # 创建频域掩码
        low_mask = torch.zeros_like(fft_signal)
        high_mask = torch.zeros_like(fft_signal)
        mid_mask = torch.zeros_like(fft_signal)

        # 低频分量 - 使用正确的索引
        low_mask[:, :, :low_cutoff] = 1

        # 高频分量
        high_mask[:, :, high_cutoff:] = 1

        # 中频分量
        mid_mask[:, :, low_cutoff:high_cutoff] = 1

        # 分离各频率分量
        low_fft = fft_signal * low_mask
        high_fft = fft_signal * high_mask
        mid_fft = fft_signal * mid_mask

        # 变换回时域
        low_freq = torch.fft.irfft(low_fft, n=seq_len, dim=2)
        high_freq = torch.fft.irfft(high_fft, n=seq_len, dim=2)
        mid_freq = torch.fft.irfft(mid_fft, n=seq_len, dim=2)

        print(
            f"Results - low: {low_freq.abs().sum():.4f}, high: {high_freq.abs().sum():.4f}, mid: {mid_freq.abs().sum():.4f}")

        # 验证重构
        reconstructed = low_freq + high_freq + mid_freq
        reconstruction_error = torch.mean((reconstructed - signal) ** 2)
        print(f"Reconstruction error (MSE): {reconstruction_error:.8f}")

        return low_freq, high_freq, mid_freq

    # def correction(self, ts_pred, text_pred, batch_idx):
    #     return ts_pred
    def correction(self, ts_pred, text_pred, batch_idx):
        """
        使用频域分析进行时序融合
        将ts_pred和text_pred分别分解为低频、高频和其余分量，然后加权融合
        """
        # 处理text_pred维度\
        # print(batch_idx[0], text_pred[batch_idx[0]].reshape(1, -1))
        tt = [text_pred[i].reshape(1, -1) for i in batch_idx]
        text_pred = (torch.cat(tt, dim=0).to(ts_pred.device).unsqueeze(-1).transpose(-1, -2))
        ts_pred = ts_pred.transpose(-1, -2)

        # 对于序列长度较短的情况，可能需要调整频率比例
        seq_len = ts_pred.shape[2]
        freq_len = seq_len // 2 + 1  # rfft后的频率点数

        # 自适应调整频率比例，确保每个频段至少有一些频率点
        if freq_len < 10:  # 如果频率点太少
            low_freq_ratio = min(0.3, 2.0 / freq_len)  # 至少2个频率点
            high_freq_ratio = min(0.3, 2.0 / freq_len)
            # print(f"Adjusted freq ratios for short sequence: low={low_freq_ratio:.2f}, high={high_freq_ratio:.2f}")
        else:
            low_freq_ratio = 0.1
            high_freq_ratio = 0.3

        # 对ts_pred进行频域分解
        ts_low, ts_high, ts_mid = self.frequency_decomposition(ts_pred, low_freq_ratio, high_freq_ratio)

        # 对text_pred进行频域分解
        text_low, text_high, text_mid = self.frequency_decomposition(text_pred, low_freq_ratio, high_freq_ratio)

        # 堆叠所有分量 [6, batch, features, seq_len]
        stacked_components = torch.stack([
            ts_low,  # ts预测的低频分量
            ts_high,  # ts预测的高频分量
            ts_mid,  # ts预测的中频分量
            text_low,  # text预测的低频分量
            text_high,  # text预测的高频分量
            text_mid  # text预测的中频分量
        ], dim=0)

        # 加权融合
        # decomp_w: [6] -> [6, 1, 1, 1] for broadcasting
        weights = self.model.decomp_w.view(-1, 1, 1, 1)
        # with torch.no_grad():
        #     weights[:3, :, :, :] = -1

        # 归一化权重（可选）
        # print(self.model.decomp_w)
        # weights = torch.softmax(weights, dim=0)

        # 加权求和
        fused_signal = torch.sum(stacked_components * weights, dim=0)

        return fused_signal.transpose(-1, -2)



    # def contrastiveLoss_tr_sea(self, ts, text, temperature=torch.tensor(0.0)):
    #     # print(ts, text)
    #     avg_pool = self.get_model(self.model).avg_pool
    #     b,n,d,p = ts.shape
    #     ts = ts.reshape((b, n, -1))
    #     ts_trend = avg_pool(ts)
    #     ts_seasonal = ts - ts_trend
    #     ts_trend = ts_trend.reshape((b,n,d,p))
    #     ts_seasonal = ts_seasonal.reshape((b,n,d,p))
    #
    #     ts_mu = torch.mean(ts_trend.transpose(-1,-2).reshape((b,-1,d)).transpose(-1,-2),-1)
    #     text_mu = torch.mean(text[0].transpose(-1,-2).reshape((b,-1,d)).transpose(-1,-2),-1)
    #
    #     logits_mu = (ts_mu @ text_mu.T) * torch.exp(temperature)
    #     labels_mu = torch.arange(b, device=logits_mu.device)  # 标签
    #     loss_i_mu = nn.functional.cross_entropy(logits_mu, labels_mu)  # 图像到文本的交叉熵损失
    #     loss_t_mu = nn.functional.cross_entropy(logits_mu.T, labels_mu)  # 文本到图像的交叉熵损失
    #     loss_contrast_mu = (loss_i_mu + loss_t_mu) / 2  # 对称损失
    #
    #     # ts_std = torch.std(ts_seasonal.transpose(-1, -2).reshape((b, -1, d)).transpose(-1, -2), -1)
    #     # text_std = torch.std(text[1].transpose(-1, -2).reshape((b, -1, d)).transpose(-1, -2), -1)
    #     #
    #     # logits_std = (ts_std @ text_std.T) * torch.exp(temperature)
    #     # labels_std = torch.arange(b, device=logits_std.device)  # 标签
    #     # loss_i_std = nn.functional.cross_entropy(logits_std, labels_std)  # 图像到文本的交叉熵损失
    #     # loss_t_std = nn.functional.cross_entropy(logits_std.T, labels_std)  # 文本到图像的交叉熵损失
    #     # loss_contrast_std = (loss_i_std + loss_t_std) / 2  # 对称损失
    #     #
    #     # loss_contrast = loss_contrast_mu + loss_contrast_std
    #     # print('loss: ', loss_contrast_mu, loss_contrast_std, loss_contrast)
    #     return loss_contrast_mu

    def contrastiveLoss_tr_sea(self, ts, text, temperature=torch.tensor(0.0)):
        # print(ts, text)
        avg_pool = self.get_model(self.model).avg_pool
        b,n,d,p = ts.shape
        ts = ts.reshape((b, n, -1))
        ts_trend = avg_pool(ts)
        ts_seasonal = ts - ts_trend
        ts_trend = ts_trend.reshape((b,n,d,p))
        ts_seasonal = ts_seasonal.reshape((b,n,d,p))

        ts_trend = torch.mean(ts_trend.transpose(-1,-2).reshape((b,-1,d)).transpose(-1,-2),-1)
        text_trend = torch.mean(text[0].transpose(-1,-2).reshape((b,-1,d)).transpose(-1,-2),-1)

        logits_trend = (ts_trend @ text_trend.T) * torch.exp(temperature)
        labels_trend = torch.arange(b, device=logits_trend.device)  # 标签
        loss_i_trend = nn.functional.cross_entropy(logits_trend, labels_trend)  # 图像到文本的交叉熵损失
        loss_t_trend = nn.functional.cross_entropy(logits_trend.T, labels_trend)  # 文本到图像的交叉熵损失
        loss_contrast_trend = (loss_i_trend + loss_t_trend) / 2  # 对称损失

        ts_seasonal = torch.mean(ts_seasonal.transpose(-1, -2).reshape((b, -1, d)).transpose(-1, -2), -1)
        text_seasonal = torch.mean(text[1].transpose(-1, -2).reshape((b, -1, d)).transpose(-1, -2), -1)

        logits_seasonal = (ts_seasonal @ text_seasonal.T) * torch.exp(temperature)
        labels_seasonal = torch.arange(b, device=logits_seasonal.device)  # 标签
        loss_i_seasonal = nn.functional.cross_entropy(logits_seasonal, labels_seasonal)  # 图像到文本的交叉熵损失
        loss_t_seasonal = nn.functional.cross_entropy(logits_seasonal.T, labels_seasonal)  # 文本到图像的交叉熵损失
        loss_contrast_seasonal = (loss_i_seasonal + loss_t_seasonal) / 2  # 对称损失

        loss_contrast = loss_contrast_trend + loss_contrast_seasonal
        # print('loss: ', loss_contrast_trend, loss_contrast_seasonal, loss_contrast)
        return loss_contrast

    def contrastiveLoss(self, ts, text, temperature=torch.tensor(0.0)):
        b,n,d,p = ts.shape

        # logits_mu = (ts_mu @ text_mu.T) * torch.exp(temperature)
        logits = (ts.reshape((b, -1)) @ text.reshape((b, -1)).T) * torch.exp(temperature)

        labels = torch.arange(b, device=logits.device)  # 标签
        loss_i= nn.functional.cross_entropy(logits, labels)  # 图像到文本的交叉熵损失
        loss_t = nn.functional.cross_entropy(logits.T, labels)  # 文本到图像的交叉熵损失
        loss_contrast = (loss_i + loss_t) / 2  # 对称损失

        return loss_contrast

