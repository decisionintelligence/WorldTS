import os
import torch
from models import (
    DLinear,
    GPT4MTS,
    GPT4TS,
    PatchTST,
    PatchTST_clip,
    TimesNet,
    iTransformer,
    iTransformer_clip,
)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'iTransformer': iTransformer,
            'PatchTST': PatchTST,
            'DLinear':DLinear,
            'GPT4MTS': GPT4MTS,
            'GPT4TS': GPT4TS,
            'PatchTST_clip': PatchTST_clip,
            'TimesNet': TimesNet,
            'iTransformer': iTransformer,
            'iTransformer_clip': iTransformer_clip,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #     self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            # device = torch.device('cuda:{}'.format(self.args.gpu))
            # print('Use GPU: cuda:{}'.format(self.args.gpu))
            device = torch.device(f"cuda:{self.args.gpu}")
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
