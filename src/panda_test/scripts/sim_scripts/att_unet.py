import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, dropout_rate=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x.float())
        x = self.relu(x)
        # if self.dropout:
        #     x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x.float())
        x = self.relu(x)
        # if self.dropout:
        #     x = self.dropout(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, dropout_rate=None):
        super().__init__()
        self.conv = conv_block(in_c, out_c, dropout_rate)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.attention = AttentionBlock(F_g=out_c, F_l=out_c, F_int=out_c // 2)
        self.conv = conv_block(2 * out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        skip = self.attention(g=x, x=skip)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x
    
class GaussianNoiseLayer(nn.Module):
    def __init__(self, mean=0.0, std=0.1):
        super(GaussianNoiseLayer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std + self.mean
            return x + noise
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = encoder_block(3, 8)
        self.e2 = encoder_block(8, 16)
        self.e3 = encoder_block(16, 32, dropout_rate=0.1)
        self.e4 = encoder_block(32, 64, dropout_rate=0.1)
        self.e5 = encoder_block(64, 128, dropout_rate=0.1)
        self.e6 = encoder_block(128, 256, dropout_rate=0.1)
        self.e7 = encoder_block(256, 512, dropout_rate=0.1)
        self.e8 = encoder_block(512, 1024, dropout_rate=0.1)
        self.b = conv_block(1024, 2048)
        self.d1 = decoder_block(2048, 1024)
        self.d2 = decoder_block(1024, 512)
        self.d3 = decoder_block(512, 256)
        self.d4 = decoder_block(256, 128)
        self.d5 = decoder_block(128, 64)
        self.d6 = decoder_block(64, 32)
        self.d7 = decoder_block(32, 16)
        self.d8 = decoder_block(16, 8)
        self.outputs = nn.Conv2d(8, 3, kernel_size=1, padding=0)
        self.noise = GaussianNoiseLayer(mean=0, std=0.1)

    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
        s6, p6 = self.e6(p5)
        s7, p7 = self.e7(p6)
        s8, p8 = self.e8(p7)
        b = self.b(p8)
        # print(b.shape, s7.shape)
        d1 = self.d1(b, s8)
        d2 = self.d2(d1, s7)
        d3 = self.d3(d2, s6)
        d4 = self.d4(d3, s5)
        d5 = self.d5(d4, s4)
        d6 = self.d6(d5, s3)
        d7 = self.d7(d6, s2)
        d8 = self.d8(d7, s1)
        d9 = self.noise(d8)
        outputs = self.outputs(d9)
        return outputs
