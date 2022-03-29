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

class Conv2dMelDiscriminator(nn.Module):
    def __init__(self, ngf=64, nc=1):
        super(Conv2dMelDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # 1 x 80 x 80
            nn.Conv2d(
                nc + 1,
                ngf,
                kernel_size=5,
                stride=2,
                padding=2,
                dilation=1,
                bias=False
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 64 x 40 x 40
            nn.Conv2d(
                ngf,
                ngf * 2,
                kernel_size=5,
                stride=2,
                padding=2,
                dilation=1,
                bias=False
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 128 x 20 x 20
            nn.Conv2d(
                ngf * 2,
                ngf * 4,
                kernel_size=5,
                stride=2,
                padding=2,
                dilation=1,
                bias=False
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 256 x 10 x 10
            nn.Conv2d(
                ngf * 4,
                ngf * 6,
                kernel_size=5,
                stride=2,
                padding=2,
                dilation=1,
                bias=False
            ),
            nn.BatchNorm2d(ngf * 6),
            nn.ReLU(True),
            # 384 x 5 x 5
            nn.Conv2d(
                ngf * 6,
                nc,
                kernel_size=5,
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
        #print('disc')
        #print(inputs.shape, conditional.shape)
        x = torch.cat([
            inputs,
            conditional,
        ], dim=1)
        #print(x.shape)
        out = self.main(x)
        #print(out.shape)
        return out
