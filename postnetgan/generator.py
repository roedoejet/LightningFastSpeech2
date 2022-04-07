from doctest import OutputChecker
import torch
import torch.nn as nn

class MelGenerator(nn.Module):

    def __init__(self, mel_channels: int = 80, window_width = 20, conditional_projection = 1000) -> None:
        super(MelGenerator, self).__init__()
        self.mel_channels = mel_channels
        self.window_width = window_width

        self.cond_proj = nn.Linear(mel_channels * window_width, conditional_projection)

        self.main = nn.Sequential(
            nn.Linear(100 + conditional_projection, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(1024, mel_channels * window_width),
            nn.Tanh()
        )

        # Initializing all neural network weights.
        self._initialize_weights()

    def forward(self, inputs: torch.Tensor, cond: list = None) -> torch.Tensor:
        conditional_inputs = torch.cat([inputs, self.cond_proj(torch.flatten(cond, 1))], dim=-1)
        out = self.main(conditional_inputs)
        out = out.reshape(out.size(0), self.window_width, self.mel_channels)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvMelGenerator(nn.Module):
    def __init__(self, ngf = 512, nc = 80, postnet_kernel_size=5):
        super(ConvMelGenerator, self).__init__()
        postnet_kernel_size = 5
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(
                nc+1,
                ngf,
                kernel_size=postnet_kernel_size,
                stride=1,
                padding=int((postnet_kernel_size - 1) / 2),
                dilation=1,
                bias=False
            ),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose1d(
                ngf,
                ngf,
                kernel_size=postnet_kernel_size,
                stride=1,
                padding=int((postnet_kernel_size - 1) / 2),
                dilation=1,
                bias=False
            ),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose1d(
                ngf,
                ngf,
                kernel_size=postnet_kernel_size,
                stride=1,
                padding=int((postnet_kernel_size - 1) / 2),
                dilation=1,
                bias=False
            ),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose1d(
                ngf,
                ngf,
                kernel_size=postnet_kernel_size,
                stride=1,
                padding=int((postnet_kernel_size - 1) / 2),
                dilation=1,
                bias=False
            ),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose1d(
                ngf,
                nc,
                kernel_size=postnet_kernel_size,
                stride=1,
                padding=int((postnet_kernel_size - 1) / 2),
                dilation=1,
                bias=False
            ),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        # Initializing all neural network weights.
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs, conditional):
        #print(inputs.shape, conditional.shape)
        x = torch.cat([
            inputs.squeeze(),
            conditional.squeeze(),
        ], dim=-1)
        #print(x.shape)
        x = x.contiguous().transpose(1, 2)
        out = self.main(x)
        out = out.contiguous().transpose(1, 2)
        return out

class StartGenBlock(nn.Module):
    def __init__(self, in_c, out_c, shape, nc=1):
        super(StartGenBlock, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_c,out_c,5,1,0,bias=True),
            nn.LayerNorm((out_c, shape, shape)),
            nn.ReLU(True),
        )

        self.second = nn.Sequential(
            nn.Conv2d(out_c + 1,out_c,3,1,1,bias=True),
            nn.LayerNorm((out_c, shape, shape)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(out_c+1,nc,1,1,0,bias=True),
            nn.Tanh(),
        )

    def forward(self, inputs, conditional):
        activations = self.main(inputs)
        x = torch.cat([
            conditional,
            activations,
        ], dim=1)
        out_act = self.second(x)
        out_img = self.out_conv(torch.cat([
            out_act,
            conditional
        ], dim=1))
        return out_act, out_img

class GenBlock(nn.Module):
    def __init__(self, in_c, out_c, shape, nc=1, scale=2):
        super(GenBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear')

        self.main = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,1,1,bias=True),
            nn.LayerNorm((out_c, shape, shape)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c,out_c,3,1,1,bias=True),
            nn.LayerNorm((out_c, shape, shape)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_c+1,nc,1,1,0,bias=True),
            nn.Tanh(),
        )
    
    def forward(self, activations, conditional):
        activations = self.upsample(activations)

        x = torch.cat([
            conditional,
            activations,
        ], dim=1)

        out_act = self.main(x)

        out_img = self.out_conv(torch.cat([
            out_act,
            conditional
        ], dim=1))
        return out_act, out_img

class Conv2dMelGenerator(nn.Module):
    def __init__(self, ngf=32, noise=100, nc=1):
        super(Conv2dMelGenerator, self).__init__()
        
        self.blocks = nn.ModuleList([
            StartGenBlock(noise, ngf * 8, 5), # -> ndf * 8 x 5 x 5
            GenBlock(ngf * 8 + nc, ngf * 8, 10), # -> ndf * 4 x 10 x 10
            GenBlock(ngf * 8 + nc, ngf * 4, 20), # -> ndf *2 x 20 x 20
            GenBlock(ngf * 4 + nc, ngf * 4, 40), # -> ndf x 40 x 40
            GenBlock(ngf * 4 + nc, 1, 80), # -> 1 x 80 x 80
        ])

        # Initializing all neural network weights.
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, latent, conditionals):
        imgs = []
        for i, layer in enumerate(self.blocks):
            if i == 0:
                out, out_img = layer(latent, conditionals[::-1][i])
            else:
                out, out_img = layer(out, conditionals[::-1][i])
            imgs.append(out_img)
        imgs.reverse()
        # for i in imgs:
        #     print('img', i.shape)
        return imgs