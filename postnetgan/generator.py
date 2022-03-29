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
        out = self.main(x)
        out = out.contiguous().transpose(1, 2)
        return out

class Conv2dMelGenerator(nn.Module):
    def __init__(self, ngf = 64, nc = 1):
        super(Conv2dMelGenerator, self).__init__()
        self.main = nn.Sequential(
            # 2 x 80 x 80
            nn.Conv2d(
                nc+1,
                ngf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 64 x 40 x 40
            nn.Conv2d(
                ngf,
                ngf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                dilation=1,
                bias=False
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 128 x 20 x 20
            nn.Conv2d(
                ngf * 2,
                ngf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                dilation=1,
                bias=False
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 256 x 10 x 10
            nn.ConvTranspose2d(
                ngf * 4,
                ngf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                dilation=1,
                bias=False
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 128 x 20 x 20
            nn.ConvTranspose2d(
                ngf * 2,
                ngf,
                kernel_size=4,
                stride=2,
                padding=1,
                dilation=1,
                bias=False
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 64 x 40 x 40
            nn.ConvTranspose2d(
                ngf,
                nc,
                kernel_size=4,
                stride=2,
                padding=1,
                dilation=1,
                bias=False
            ),
            nn.Tanh()
            # 1 x 80 x 80
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
        #print('gen')
        #print(inputs.shape, conditional.shape)
        x = torch.cat([
            inputs,
            conditional,
        ], dim=1)
        #print(x.shape)
        out = self.main(x)
        #print(out.shape)
        return out