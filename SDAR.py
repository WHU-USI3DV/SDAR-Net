import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Style_loss_module(nn.Module):
    def __init__(self,):
        super(Style_loss_module, self).__init__()

    def gram_matrix(self,tensor):
        '''
        Gram矩阵表示图像的风格特征，在保证内容的情况下，进行风格传输
        '''
        # 计算gram矩阵
        # gram = einsum('b i n, b n j  -> b i j', tensor,tensor.transpose(1,2))
        gram=torch.bmm(tensor,tensor.transpose(1,2))
        return gram
    
    def forward(self, x_bot, gt_bot,return_gram=False):
        B=x_bot.size(0)

        gram_x=self.gram_matrix(x_bot.view(B,x_bot.size(1),-1))
        gram_gt=self.gram_matrix(gt_bot.view(B,gt_bot.size(1),-1))
        style_loss = torch.mean((gram_gt - gram_x) ** 2)/(x_bot.size(1)*x_bot.size(2)*x_bot.size(3))
        if return_gram:
            return style_loss,gram_x,gram_gt
        else:
            return style_loss
    
    def orth_reg(self, x_bot):
        B,C,H,W=x_bot.size()

        gram_x=self.gram_matrix(x_bot.view(B,C,-1))/(H*W)
        identity=torch.eye(C,device=x_bot.device).unsqueeze(0)
        loss=torch.mean((gram_x-identity)**2)

        return loss,gram_x

from attention import NonLocalSparseAttention
from modules import AFF, RCB, AtrousBlock, DownSample, GetGradientNopadding, GetGradientNopadding_1cGray, UpSample

class GramGlobalWeightNet(nn.Module):
    def __init__(self, channels, num_weights=3,):
        """
        从完整 Gram 矩阵中提取信息，生成加权 logits。
        
        Args:
            channels: 输入特征通道数 C
            num_weights: 输出权重数量（如 3 个分支）
            proj_dim: 投影维度 K（控制参数量和表达能力）
        """
        super().__init__()
        self.channels = channels
        self.num_b=num_weights
        self.proj_a = nn.Linear(channels,int(channels//4))
        self.proj_b = nn.Linear(channels,int(channels//4))

        self.mlp = nn.Sequential(
                nn.Linear(int((channels//4)*(channels//4))+channels, channels),#+channels
                nn.ReLU(inplace=True),
                nn.Linear(channels, 1)
            )
        # self.mlp = nn.Linear(int((channels//4)*(channels//4)), 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.posemb=torch.linspace(0,1,channels).view(1,channels,1).cuda()

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def gram_matrix(self,tensor):
        '''
        Gram矩阵表示图像的风格特征，在保证内容的情况下，进行风格传输
        '''
        # 计算gram矩阵
        # gram = einsum('b i n, b n j  -> b i j', tensor,tensor.transpose(1,2))
        gram=torch.bmm(tensor,tensor.transpose(1,2))#/(tensor.size(1)*tensor.size(2))
        return gram

    def forward(self, xs):
        """
        Args:
            xs: list of [B, C, H, W] 
        
        Returns:
            logits: [B, num_weights]
        """
        Logs=[]
        i=0
        for x in xs:
            B, C, _,_ = x.shape
            gram_matrix=self.gram_matrix(x.view(B,C,-1))
            a = self.proj_a(gram_matrix)
            b = self.proj_b(gram_matrix.transpose(1,2))

            Gb = torch.bmm(a.transpose(-1, -2), b) # [B, C, K]
            Gb = Gb.view(B, -1)

            z = self.pool(x).view(B, C)
            ggz=torch.cat([z,Gb],dim=1)
            # ggz=Gb
            logit = self.mlp(ggz)  # [B, num_weights]
            i+=1
            Logs.append(logit)
        logits=torch.cat(Logs,dim=1)
        return logits

class SDAR_Net(nn.Module):
    def __init__(self, num_branch=3, n_feat=32, height=256, width=256, n_RCB=2, chan_factor=2, bias=True):
        super(SDAR_Net, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width
        self.act = nn.LeakyReLU(0.1, True)
        atrous = [1, 2, 3, 4]

        rcb_top = [RCB(int(n_feat * chan_factor ** 0), self.act, bias=bias) for _ in range(n_RCB)]
        self.dau_top = nn.Sequential(*rcb_top)
        rcb_mid = [RCB(int(n_feat * chan_factor ** 1), self.act, bias=bias) for _ in range(n_RCB)]
        self.dau_mid = nn.Sequential(*rcb_mid)
        rcb_bot = [RCB(int(n_feat * chan_factor ** 2), self.act, bias=bias) for _ in range(n_RCB)]
        self.dau_bot = nn.Sequential(*rcb_bot)
        self.nl_mid = NonLocalSparseAttention(channels=int(n_feat * chan_factor ** 1))
        self.nl_bot = NonLocalSparseAttention(channels=int(n_feat * chan_factor ** 2))
        self.atb_top = AtrousBlock(int(n_feat * chan_factor ** 0), 3, 1, self.act, atrous)
        self.atb_mid = AtrousBlock(int(n_feat * chan_factor ** 1), 3, 1, self.act, atrous)
        self.atb_bot = AtrousBlock(int(n_feat * chan_factor ** 2), 3, 1, self.act, atrous)

        self.down2 = nn.Sequential(
            DownSample(int((chan_factor ** 0) * n_feat), 2, chan_factor),
        )
        self.down4 = nn.Sequential(
            DownSample(int((chan_factor ** 0) * n_feat), 2, chan_factor),
            DownSample(int((chan_factor ** 1) * n_feat), 2, chan_factor),
        )

        self.up21_1 = UpSample(int((chan_factor ** 1) * n_feat), 2, chan_factor)
        self.up21_2 = UpSample(int((chan_factor ** 1) * n_feat), 2, chan_factor)
        self.up32_1 = UpSample(int((chan_factor ** 2) * n_feat), 2, chan_factor)
        self.up32_2 = UpSample(int((chan_factor ** 2) * n_feat), 2, chan_factor)

        self.conv_in = nn.Conv2d(4, n_feat, kernel_size=3, padding=1, bias=bias)
        self.conv_mid = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=bias)
        self.conv_out = nn.Conv2d(n_feat, 3, kernel_size=3, padding=1, bias=bias)

        # only two inputs for AFF
        self.aff_top = AFF(int(n_feat * chan_factor ** 0), self.act)
        self.aff_mid = AFF(int(n_feat * chan_factor ** 1), self.act)
        self.aff_final = AFF(n_feat, self.act)

        self.get_gradient = GetGradientNopadding_1cGray()
        self.b_concat_1 = nn.Conv2d(2 * n_feat, n_feat, kernel_size=3, padding=1, bias=bias)
        self.b_block_1 = RCB(2 * n_feat, self.act, bias=bias)

        self.b_concat_2 = nn.Conv2d(2 * n_feat, n_feat, kernel_size=3, padding=1, bias=bias)
        self.b_block_2 = RCB(2 * n_feat, self.act, bias=bias)

        self.style_loss=Style_loss_module()

        self.num_branch=num_branch
        self.adaptive_route=GramGlobalWeightNet(channels=n_feat * chan_factor ** 2,num_weights=num_branch)

    def encode(self,x):
        x_top = x.clone()
        x_grad = self.get_gradient(x)
        x_str = torch.cat([x_top, x_grad], dim=1)
        x_str = self.conv_in(x_str)
        x_style = self.down4(x_str)
        return x_str,x_style
    
    def transfer(self, x_str,x_style):
        x_mid = self.down2(x_str)
        x_str1 = self.dau_top(self.atb_top(x_str))
        x_mid1 = self.dau_mid(self.atb_mid(x_mid))
        x_style1 = self.dau_bot(self.atb_bot(x_style))

        x_mid1 = self.aff_mid(x_mid1, self.up32_1(x_style1))
        x_str1 = self.aff_top(x_str1, self.up21_1(x_mid1))

        x_str2 = self.dau_top(self.atb_top(x_str1))
        x_mid2 = self.dau_mid(self.nl_mid(x_mid1))
        x_style2 = self.dau_bot(self.nl_bot(x_style1))

        x_mid2 = self.aff_mid(x_mid2, self.up32_2(x_style2))
        x_str2 = self.aff_top(x_str2, self.up21_2(x_mid2))

        mid_out = self.conv_mid(x_str2)
        mid_out = mid_out + x_str2

        x_cat_1 = torch.cat([x_str,x_str1], dim=1)
        x_cat_1 = self.b_block_1(x_cat_1)
        x_cat_1 = self.b_concat_1(x_cat_1)

        x_cat_2 = torch.cat([x_cat_1, x_str2], dim=1)
        x_cat_2 = self.b_block_2(x_cat_2)
        x_cat_2 = self.b_concat_2(x_cat_2)

        out_f = self.aff_final(mid_out, x_cat_2)
        return out_f,x_style2
    
    def decode(self,out_f):
        
        result = self.conv_out(out_f)
        return result
    
    def forward(self, input):
        #print(x.shape)
        x_str,x_style=self.encode(input)
        x_str,x_style=self.transfer(x_str,x_style)
        output=self.decode(x_str)
        return output
    
    def forward_recon(self, input):
        #print(x.shape)
        x_str,x_style=self.encode(input)
        output=self.decode(x_str)
        return output
      
    def forward_style_loss(self, input, gt):
		#print(x.shape)
        x_str,x_style=self.encode(input)
        x_str,x_style=self.transfer(x_str,x_style)
        _,gt_style=self.encode(gt)
        style_loss=self.style_loss(x_style,gt_style)
        output=self.decode(x_str)
        return output,style_loss
    
    def forward_route(self, input,return_logits=False,return_proc_outs=False):
		#print(x.shape)
        x_str,x_style=self.encode(input)
        x_style_w=torch.zeros_like(x_style)
        x_str_w=torch.zeros_like(x_str)
        x_style_branch_outs=[x_style]
        x_str_branch_outs=[x_str]
        x_outs=[self.decode(x_str)]
        # weight_logits = self.adaptive_route(x_style) 
        for i in range(self.num_branch-1):
            x_str,x_style=self.transfer(x_str,x_style)
            x_style_branch_outs.append(x_style)
            x_str_branch_outs.append(x_str)
            if return_proc_outs:
                x_out=self.decode(x_str)
                x_outs.append(x_out)

        weight_logits = self.adaptive_route(x_style_branch_outs) 
        weights = F.softmax(weight_logits, dim=1)  # [B, 3]
        
        for i in range(self.num_branch):
            x_style_w+=weights[:, i].view(-1, 1, 1, 1) * x_style_branch_outs[i]
            x_str_w+=weights[:, i].view(-1, 1, 1, 1) * x_str_branch_outs[i]
        output=self.decode(x_str_w)
        if return_logits:
            if return_proc_outs:
                return output,weight_logits,x_outs
            else:
                return output,weight_logits
        else:
            return output
        
    def forward_route_style_loss(self, input,gt,return_logits=False,return_proc_outs=False):
		#print(x.shape)
        x_str,x_style=self.encode(input)
        x_style_w=torch.zeros_like(x_style)
        x_str_w=torch.zeros_like(x_str)
        x_style_branch_outs=[x_style]
        x_str_branch_outs=[x_str]
        x_outs=[input]
        # weight_logits = self.adaptive_route(x_style) 
        for i in range(self.num_branch-1):
            x_str,x_style=self.transfer(x_str,x_style)
            x_style_branch_outs.append(x_style)
            x_str_branch_outs.append(x_str)
            if return_proc_outs:
                x_out=self.decode(x_str)
                x_outs.append(x_out)
        weight_logits = self.adaptive_route(x_style_branch_outs) 
        weights = F.softmax(weight_logits, dim=1)  # [B, 3]
        
        for i in range(self.num_branch):
            x_style_w+=weights[:, i].view(-1, 1, 1, 1) * x_style_branch_outs[i]
            x_str_w+=weights[:, i].view(-1, 1, 1, 1) * x_str_branch_outs[i]
        output=self.decode(x_str_w)
        _,gt_style=self.encode(gt)
        style_loss=self.style_loss(x_style_w,gt_style)
        if return_logits:
            if return_proc_outs:
                return output,weight_logits,style_loss,x_outs
            else:
                return output,weight_logits,style_loss
        else:
            return output,style_loss