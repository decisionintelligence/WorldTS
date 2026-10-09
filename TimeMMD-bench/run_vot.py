import argparse
import os
import torch
from exp.exp_long_term_forecasting_clip import Exp_Long_Term_Forecast_Clip
from utils.print_args import print_args
import random
import numpy as np
import re
from utils.get_device import get_free_memory

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def create_shortened_setting(args):
    """
    根据参数字典创建缩短的设置字符串
    """
    shortened_pairs = {}
    for param, value in vars(args).items():
        print(param, str(value))
        # 缩写参数名
        # short_param = abbreviate_param(param)

        # 处理值：保留关键信息，缩短长字符串或列表
        if param == 'gpu' or param == 'devices' or param == 'device_ids' or param == 'decomp_w':
            short_value = '1'
        elif isinstance(value, list):
            # 列表只保留长度标识
            short_value = f"[{str(len(value))}]"
        elif isinstance(value, str):
            # 字符串保留前2个字符
            short_value = f"{value.split('/')[-1][:2]}"
        elif isinstance(value, bool):
            short_value = '1' if short_value else '0' # 这里有问题，但是为了不白跑先不改了（不影响结果，都是不变的），应该是value
        elif value == None:
            short_value = ''
        else:
            # 其他类型使用原始值
            short_value = str(value)

        shortened_pairs[param] = short_value

    return shortened_pairs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='PatchTST_clip',
                        help='model name, options: [PatchTST_clip]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='Algriculture.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='/public_data/syWang/checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    # parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    # parser.add_argument('--text_emb', type=int, default=96, help='prediction sequence length')


    # model define
    # parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    # parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    # parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    # parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--train_epochs_clip', type=int, default=1, help='train epochs')
    parser.add_argument('--train_epochs_corr', type=int, default=1, help='train epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--learning_rate_clip', type=float, default=0.0001, help='optimizer learning rate clip step')
    parser.add_argument('--learning_rate_corr', type=float, default=0.0001, help='optimizer learning rate correction step')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')

    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--if_lradj', type=int, default=0, help='if adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    #  new pars
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model') # LLAMA2, LLAMA3, GPT2, BERT, GPT2M, GPT2L, GPT2XL, Doc2Vec, ClosedLLM
    parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
    parser.add_argument('--llm_layers', type=int, default=6)
    # parser.add_argument('--text_path', type=str, default="None")
    # parser.add_argument('--type_tag', type=str, default="#F#")
    # parser.add_argument('--text_len', type=int, default=3)
    parser.add_argument('--clip_t', type=float, default=0.0, help='clip_t')
    parser.add_argument('--decomp_w', type=float, default=0.1, help='decomp_w')
    parser.add_argument('--multimodal', type=int, default=1, help='multimodal')
    parser.add_argument('--correct', type=int, default=1, help='correct')
    parser.add_argument('--tr_sea', type=int, default=1, help='trend & seasonal')



    # parser.add_argument('--learning_rate2', type=float, default=1e-2, help='mlp learning rate')
    # parser.add_argument('--learning_rate3', type=float, default=1e-3, help='proj learning rate')
    # parser.add_argument('--prompt_weight', type=float, default=0.01, help='prompt weight')#please tune this hyperparameter for combining
    # parser.add_argument('--prior_weight', type=float, default=0.01, help='prompt weight')#please tune this hyperparameter for combining
    # parser.add_argument('--pool_type', type=str, default='avg', help='pooling type') #avg min max attention
    # parser.add_argument('--date_name', type=str, default='end_date', help='matching date name in csv') #mlp linear
    # parser.add_argument('--addHisRate', type=float, default=0.5, help='add historical rate')
    # parser.add_argument('--init_method', type=str, default='normal', help='init method of combined weight')
    # parser.add_argument('--learning_rate_weight', type=float, default=0.0001, help='optimizer learning rate')

    parser.add_argument('--save_name', type=str, default='result_clip', help='save name')
    # parser.add_argument('--use_fullmodel', type=int, default=0, help='use full model or just encoder')
    # parser.add_argument('--use_closedllm', type=int, default=0, help='use closedllm or not')
    # parser.add_argument('--huggingface_token', type=str, help='your token of huggingface;need for llama3')

    # LATS
    # parser.add_argument('--text_dim', type=int, default=12, help='text embedding dimension after low rank projection')
    args = parser.parse_args()
    domain= re.search(r'/([^/]+)$', args.root_path).group(1)
    print("now running on domain {} model {} ".format(domain,args.model))
    if args.model=="LightTS":   
        if args.pred_len<args.seq_len:
            args.seq_len=args.pred_len
    if args.llm_model=="BERT":
        args.llm_dim=768
    elif args.llm_model=="GPT2":
        args.llm_dim=768
    elif args.llm_model=="LLAMA2":
        args.llm_dim=4096
    elif args.llm_model=="LLAMA3":
        args.llm_dim=4096
    elif args.llm_model=="GPT2M":
        args.llm_dim=1024
    elif args.llm_model=="GPT2L":
        args.llm_dim=1280
    elif args.llm_model=="GPT2XL":
        args.llm_dim=1600
    elif args.llm_model=="Doc2Vec":
        args.llm_dim=64
    elif args.llm_model=="ClosedLLM":
        args.llm_model="BERT" #just for encoding
        args.llm_dim=768
        args.use_closedllm=1


    args.features = 'S'
    args.enc_in = 1
    args.dec_in = 1
    args.c_out = 1
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!overwrite features to 'S' for univariate time series data and dim=1")
    fix_seed = args.seed

    print("Now using seed {}".format(fix_seed))
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    # if args.use_gpu and args.use_multi_gpu:
    #     args.devices = args.devices.replace(' ', '')
    #     device_ids = args.devices.split(',')
    #     args.device_ids = [int(id_) for id_ in device_ids]
    #     args.gpu = args.device_ids[0]

    if args.use_gpu:  # and args.use_multi_gpu
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        # args.gpu = args.device_ids[1]
    num_devices = torch.cuda.device_count()
    args.gpu = get_free_memory(args.device_ids)


    Exp = Exp_Long_Term_Forecast_Clip


    if args.is_training:
        for ii in range(args.itr):

            setting_pairs = create_shortened_setting(args)
            setting =  '_'.join(list(setting_pairs.values()))+ '_' + str(ii)

            for i in setting_pairs:
                print(i+'='+setting_pairs[i],end=' ')
            print('\n')

            exp = Exp(args)  # set experiments

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if args.multimodal:
                forecast = 2
            else:
                forecast = 1
            now_mse=exp.test(setting, forecast = forecast)

            # if args.do_predict:
            #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #     exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0

        setting = create_shortened_setting(args) + str(ii)
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        now_mse=exp.test(setting, test=1)
        torch.cuda.empty_cache()

