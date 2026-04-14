# --- Imports --- #
from module_utils import *
from attention import NonLocalSparseAttention
## modules from Semi-UIR
class SFT_layer(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(SFT_layer, self).__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )

    def forward(self, x, inter):
        '''
        :param x: degradation representation: B * C
        :param inter: degradation intermediate representation map: B * C * H * W
        '''
        gamma = self.conv_gamma(inter)
        beta = self.conv_beta(inter)

        return x * gamma + beta

class GetGradientNopadding(nn.Module):
    def __init__(self):
        super(GetGradientNopadding, self).__init__()
        kernel_v = [[-0, -1, -0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[-0, 0, 0],
                    [-1, 0, 1],
                    [-0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

        weights = torch.tensor([0.2989, 0.5870, 0.1140])
        # self.weights =nn.Parameter(data=weights.view(1, 3, 1, 1), requires_grad=True)
        self.weights = weights.view(1, 3, 1, 1)
    def rgb_to_grayscale(self, x):
        """Convert RGB image to grayscale using standard weights"""
        if x.size(1) == 3:
            return (x * self.weights.to(x.device)).sum(dim=1, keepdim=True).repeat(1,3,1,1)
        return x

    def forward(self, inp_feat,gray=False):
        if gray:
            inp_feat=self.rgb_to_grayscale(inp_feat)
        x_list = []
        for i in range(inp_feat.shape[1]):
            x_i = inp_feat[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        res = torch.cat(x_list, dim=1)

        return res

class GetGradientNopadding_1cGray(nn.Module):
    def __init__(self):
        super(GetGradientNopadding_1cGray, self).__init__()
        kernel_v = [[-0, -1, -0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[-0, 0, 0],
                    [-1, 0, 1],
                    [-0, 0, 0]]
        
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

        weights = torch.tensor([0.2989, 0.5870, 0.1140])
        self.weights = weights.view(1, 3, 1, 1)
    def rgb_to_grayscale(self, x):
        """Convert RGB image to grayscale using standard weights"""
        if x.size(1) == 3:
            return (x * self.weights.to(x.device)).sum(dim=1, keepdim=False).unsqueeze(1)
        return x

    def forward(self, inp_feat):
        inp_feat=self.rgb_to_grayscale(inp_feat)
        x_list = []
        for i in range(inp_feat.shape[1]):
            x_i = inp_feat[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)
        res = torch.cat(x_list, dim=1)
        return res

class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels * chan_factor), 1, stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        return self.bot(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels // chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        )

    def forward(self, x):
        return self.bot(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class ContextBlock(nn.Module):

    def __init__(self, n_feat, activation, bias=True):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            activation,
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        input_x = input_x.view(batch, channel, height * width)
        input_x = input_x.unsqueeze(1)
        context_mask = self.conv_mask(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax(context_mask)
        context_mask = context_mask.unsqueeze(3)
        context = torch.matmul(input_x, context_mask)
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        context = self.modeling(x)
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term
        return x


# Residual Context Block (RCB)
class RCB(nn.Module):
    def __init__(self, n_feat, act, bias=True,k=3):
        super(RCB, self).__init__()

        self.act = act
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=k, stride=1, padding=int((k-1)//2), bias=bias),
            self.act,
            nn.Conv2d(n_feat, n_feat, kernel_size=k, stride=1, padding=int((k-1)//2), bias=bias)
        )
        self.gcnet = ContextBlock(n_feat, self.act, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.act(self.gcnet(res))
        res = x + res
        return res


# Attention Feature Fusion (AFF)
class AFF(nn.Module):
    def __init__(self, channels, activation, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            activation,
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            activation,
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = torch.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class AFF_global(nn.Module):
    def __init__(self, channels, activation, r=4):
        super(AFF_global, self).__init__()
        inter_channels = int(channels // r)

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            activation,
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, residual):
        xa = x + residual
        xg = self.global_att(xa)
        wei = torch.sigmoid(xg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class AtrousBlock(nn.Module):
    def __init__(self, mid_channels, kernel_size, stride, activation, atrous=[1, 2, 3, 4]):
        super(AtrousBlock, self).__init__()
        self.atrous_layers = []
        for i in range(4):
            self.atrous_layers.append(
                nn.Conv2d(mid_channels, mid_channels // 2, kernel_size, stride, dilation=atrous[i],
                          padding=atrous[i]))
        self.atrous_layers = nn.Sequential(*self.atrous_layers)
        self.conv = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0)
        self.act = activation
        self.att = AFF(mid_channels, self.act)

    def forward(self, data):
        x1 = self.act(self.atrous_layers[0](data))
        x2 = self.act(self.atrous_layers[1](data))
        x3 = self.act(self.atrous_layers[2](data))
        x4 = self.act(self.atrous_layers[3](data))

        x_total = self.act(self.conv(torch.cat((x1, x2, x3, x4), 1)))
        output = self.att(data, x_total)
        return output

