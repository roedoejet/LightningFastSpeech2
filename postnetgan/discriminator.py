from turtle import forward
import torch
import torch.nn as nn

class MelDiscriminator(nn.Module):

    def __init__(self, mel_channels: int = 80, window_width = 20, conditional_projection = 1000) -> None:
        super(MelDiscriminator, self).__init__()

        self.cond_proj = nn.Linear(mel_channels * window_width, conditional_projection)

        self.main = nn.Sequential(
            nn.Linear(mel_channels * window_width + conditional_projection, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Initializing all neural network weights.
        self._initialize_weights()

    def forward(self, inputs: torch.Tensor, cond: list = None) -> torch.Tensor:
        conditional = torch.flatten(cond, 1)
        conditional_inputs = torch.cat([
            torch.flatten(inputs, 1),
            self.cond_proj(conditional)
        ], dim=-1)
        out = self.main(conditional_inputs)
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

class ConvMelDiscriminator(nn.Module):
    def __init__(self, ngf=256, nc=80, postnet_kernel_size=5):
        super(ConvMelDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                nc*2,
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
            nn.Conv1d(
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
            nn.Conv1d(
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
            nn.Conv1d(
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
            nn.Conv1d(
                ngf,
                1,
                kernel_size=20,
                stride=1,
                padding=0,
                dilation=1,
                bias=False
            ),
            nn.Sigmoid()
        )

        # Initializing all neural network weights.
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
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
        out = self.main(x).squeeze(-1)
        return out

class StartDisBlock(nn.Module):
    def __init__(self, in_c, out_c, shape):
        super(StartDisBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,1,1,bias=True),
            nn.LayerNorm((out_c, shape, shape)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c,out_c,3,1,1,bias=True),
            nn.LayerNorm((out_c, shape, shape)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2,2),
        )

    def forward(self, inputs, conditional):
        x = torch.cat([
            inputs,
            conditional,
        ], dim=1)
        out = self.main(x)
        return out

class DisBlock(nn.Module):
    def __init__(self, in_c, out_c, shape):
        super(DisBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,1,1,bias=True),
            nn.LayerNorm((out_c, shape, shape)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c,out_c,3,1,1,bias=True),
            nn.LayerNorm((out_c, shape, shape)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2,2),
        )
    
    def forward(self, activations, images, conditional):
        x = torch.cat([
            images,
            conditional,
            activations,
        ], dim=1)
        out = self.main(x)
        return out

class EndDisBlock(nn.Module):
    def __init__(self, in_c, out_c, shape):
        super(EndDisBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,1,1,bias=True),
            nn.LayerNorm((out_c, shape, shape)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c,out_c,4,2,0,bias=True),
            nn.LayerNorm((out_c, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_c, 1),
            nn.Sigmoid()
        )
    
    def forward(self, activations, images, conditional):
        x = torch.cat([
            images,
            conditional,
            activations,
        ], dim=1)
        out = self.main(x)
        out = self.fc(out.squeeze())
        return out

class Conv2dMelDiscriminator(nn.Module):
    def __init__(self, ndf=32, nc=1):
        super(Conv2dMelDiscriminator, self).__init__()

        self.blocks = nn.ModuleList([
            StartDisBlock(nc * 2, ndf * 4, 80), # -> ndf x 40 x 40
            DisBlock(ndf * 4 + nc * 2, ndf * 4, 40), # -> ndf*2 x 20 x 20
            DisBlock(ndf * 4 + nc * 2, ndf * 8, 20), # -> ndf*4 x 10 x 10
            DisBlock(ndf * 8 + nc * 2, ndf * 8, 10), # -> ndf*8 x 5 x 5
            EndDisBlock(ndf * 8 + nc * 2, ndf * 8, 5), # -> 1 x 1 x 1
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

    def forward(self, images, conditionals):
        for i, layer in enumerate(self.blocks):
            # print('ic', images[i].shape, conditionals[i].shape)
            if i == 0:
                out = layer(images[i], conditionals[i])
            else:
                out = layer(out, images[i], conditionals[i])
        return out
