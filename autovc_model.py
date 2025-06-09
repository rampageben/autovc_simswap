import torch 
from autovc_network import *
from collections import OrderedDict


class AutovcModel(torch.nn.Module):
    def __init__(self, device):
        super(AutovcModel, self).__init__()

        self.device = device

        # 載入 speaker encoder
        self.speaker_encoder = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
        ckpt = torch.load('/content/drive/MyDrive/autovc/3000000-BL.ckpt')
        new_state_dict = OrderedDict()
        for key, val in ckpt['model_b'].items():
            new_state_dict[key[7:]] = val
        self.speaker_encoder.load_state_dict(new_state_dict)
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False
        self.speaker_encoder.to(self.device)
        self.speaker_encoder.eval()
        
        self.G = Generator(dim_neck= 16, dim_emb = 256, dim_pre = 512, freq = 16).to(self.device)

        self.D_1 = Discriminator(input_nc=1).to(self.device)
        self.D_2 = Discriminator(input_nc=1).to(self.device)

        
       




