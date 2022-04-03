import pytorch_lightning as pl
from torch.nn.modules.transformer import TransformerEncoder
from model import ConformerEncoderLayer, PositionalEncoding, VarianceAdaptor
from torch import nn
from torch.utils.data import DataLoader
import torch
import configparser
import multiprocessing
import wandb
import torchvision.transforms as VT
from torchvision.utils import make_grid
from synthesiser import Synthesiser
from postnet import PostNet
from postnetgan import MelDiscriminator, MelGenerator, Conv2dMelDiscriminator, Conv2dMelGenerator
from sklearn.metrics import accuracy_score
import numpy as np

from dataset import ProcessedDataset, UnprocessedDataset

cpus = multiprocessing.cpu_count()

config = configparser.ConfigParser()
config.read("config.ini")

# TODO:
# allow to replace with "real" conformer
# allow to replace with linear FFN with same number of params
# preprocess on frame level and allow phoneme level
# add option for postnet


class FastSpeech2Loss(nn.Module):
    def __init__(self, postnet, postnet_type, blur):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        self.postnet = postnet
        self.postnet_type = postnet_type
        self.blur = blur

    @staticmethod
    def get_loss(pred, truth, loss, mask, unsqueeze=False):
        truth.requires_grad = False
        if unsqueeze:
            mask = mask.unsqueeze(-1)
        pred = pred.masked_select(mask)
        truth = truth.masked_select(mask)
        return loss(pred, truth)

    def forward(self, pred, truth, src_mask, tgt_mask, tgt_max_length=None):
        # TODO: do this via keys
        mel_pred, pitch_pred, energy_pred, duration_pred, _, postnet_pred, _ = pred

        if not self.blur:
            mel_tgt, pitch_tgt, energy_tgt, duration_tgt = truth
        else:
            mel_tgt, pitch_tgt, energy_tgt, duration_tgt, mel_blur = truth

        duration_tgt = torch.log(duration_tgt.float() + 1)

        src_mask = ~src_mask
        tgt_mask = ~tgt_mask

        if tgt_max_length is not None:
            mel_tgt = mel_tgt[:, :tgt_max_length, :]
            if self.blur:
                mel_blur = mel_blur[:, :tgt_max_length, :]
            #tgt_mask = tgt_mask[:, :tgt_max_length]
            if config["dataset"].get("variance_level") == "frame":
                pitch_tgt = pitch_tgt[:, :tgt_max_length]
                energy_tgt = energy_tgt[:, :tgt_max_length]

        if not self.blur:
            mel_loss = FastSpeech2Loss.get_loss(
                mel_pred, mel_tgt, self.l1_loss, tgt_mask, unsqueeze=True
            )
        else:
            mel_loss = FastSpeech2Loss.get_loss(
                mel_pred, mel_blur, self.l1_loss, tgt_mask, unsqueeze=True
            )

        if self.postnet:
            if self.postnet_type == 'conv':
                postnet_loss = FastSpeech2Loss.get_loss(
                    postnet_pred, mel_tgt-mel_blur, self.l1_loss, tgt_mask, unsqueeze=True
                )
            elif self.postnet_type == 'gan':
                real_output = torch.cat([p['real_output'] for p in postnet_pred], dim=0)
                real_label =  torch.cat([p['real_label'] for p in postnet_pred], dim=0)
                fake_output_d = torch.cat([p['fake_output_d'] for p in postnet_pred], dim=0)
                fake_label =  torch.cat([p['fake_label'] for p in postnet_pred], dim=0)
                fake_output_g = torch.cat([p['fake_output_g'] for p in postnet_pred], dim=0)
                #print(real_output.shape, real_label.shape)
                d_loss_real = self.bce_loss(real_output, real_label)
                #print(fake_output_d.shape, fake_label.shape)
                d_loss_fake = self.bce_loss(fake_output_d, fake_label)
                d_loss = (d_loss_real + d_loss_fake)
                #print(fake_output_g.shape, real_label.shape)
                g_loss = self.bce_loss(fake_output_g, real_label)

                
        if config["dataset"].get("variance_level") == "frame":
            pitch_energy_mask = tgt_mask
        elif config["dataset"].get("variance_level") == "phoneme":
            pitch_energy_mask = src_mask

        pitch_loss = FastSpeech2Loss.get_loss(
            pitch_pred, pitch_tgt, self.mse_loss, pitch_energy_mask
        )
        energy_loss = FastSpeech2Loss.get_loss(
            energy_pred, energy_tgt, self.mse_loss, pitch_energy_mask
        )

        duration_loss = FastSpeech2Loss.get_loss(
            duration_pred, duration_tgt, self.mse_loss, src_mask
        )

        total_loss = mel_loss + pitch_loss + energy_loss + duration_loss

        if self.postnet:
            if self.postnet_type == 'conv':
                total_loss = total_loss + postnet_loss
            elif self.postnet_type == 'gan':
                total_loss = total_loss + d_loss + g_loss

        result = [
            total_loss,
            mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        ]

        if self.postnet:
            if self.postnet_type == 'conv':
                result.append(postnet_loss) 
            elif self.postnet_type == 'gan':
                result.append(d_loss_real) 
                result.append(d_loss_fake) 
                result.append(g_loss)
        
        return result


class FastSpeech2(pl.LightningModule):
    def __init__(self, learning_rate=config["train"].getfloat("lr")):
        super().__init__()

        self.lr = learning_rate

        # data
        self.batch_size = config["train"].getint("batch_size")
        self.epochs = config["train"].getint("epochs")
        self.has_dvector = config["model"].getboolean("dvector")
        self.has_blur = config["model"].getboolean("blur")
        train_path = config["train"].get("train_path")
        valid_path = config["train"].get("valid_path")
        train_ud = UnprocessedDataset(train_path, dvector=self.has_dvector, blur=self.has_blur)
        valid_ud = UnprocessedDataset(valid_path, dvector=self.has_dvector, blur=self.has_blur)
        self.train_ds = ProcessedDataset(
            unprocessed_ds=train_ud,
            split="train",
            phone_vec=False
        )
        self.valid_ds = ProcessedDataset(
            unprocessed_ds=valid_ud,
            split="val",
            phone_vec=False,
            phone2id=self.train_ds.phone2id,
            stats=self.train_ds.stats
        )

        self.synth = Synthesiser(22050)

        vocab_n = self.train_ds.vocab_n
        speaker_n = self.train_ds.speaker_n
        stats = self.train_ds.stats

        # config
        encoder_hidden = config["model"].getint("encoder_hidden")
        encoder_head = config["model"].getint("encoder_head")
        encoder_layers = config["model"].getint("encoder_layers")
        encoder_dropout = config["model"].getfloat("encoder_dropout")
        decoder_hidden = config["model"].getint("decoder_hidden")
        decoder_head = config["model"].getint("decoder_head")
        decoder_layers = config["model"].getint("decoder_layers")
        decoder_dropout = config["model"].getfloat("decoder_dropout")
        conv_filter_size = config["model"].getint("conv_filter_size")
        kernel = (
            config["model"].getint("conv_kernel_1"),
            config["model"].getint("conv_kernel_2"),
        )
        self.tgt_max_length = config["model"].getint("tgt_max_length")
        self.max_lr = config["train"].getfloat("max_lr")
        mel_channels = config["model"].getint("mel_channels")
        self.has_postnet = config["model"].getboolean("postnet")
        self.postnet_type = config["model"].get("postnet_type").lower()

        # modules

        self.phone_embedding = nn.Embedding(vocab_n, encoder_hidden, padding_idx=0)

        if not self.has_dvector:
            self.speaker_embedding = nn.Embedding(speaker_n, encoder_hidden,)

        self.encoder = TransformerEncoder(
            ConformerEncoderLayer(
                encoder_hidden,
                encoder_head,
                conv_in=encoder_hidden,
                conv_filter_size=conv_filter_size,
                conv_kernel=kernel,
                batch_first=True,
                dropout=encoder_dropout,
            ),
            encoder_layers,
        )
        self.positional_encoding = PositionalEncoding(
            encoder_hidden, dropout=encoder_dropout
        )
        self.variance_adaptor = VarianceAdaptor(stats)
        self.decoder = TransformerEncoder(
            ConformerEncoderLayer(
                decoder_hidden,
                decoder_head,
                conv_in=decoder_hidden,
                conv_filter_size=conv_filter_size,
                conv_kernel=kernel,
                batch_first=True,
                dropout=decoder_dropout,
            ),
            decoder_layers,
        )

        self.linear = nn.Linear(decoder_hidden, mel_channels,)

        if self.has_postnet:
            if self.postnet_type == "gan":
                assert self.has_blur
                self.postnet_disc = Conv2dMelDiscriminator()
                self.postnet_gen = Conv2dMelGenerator()
            elif self.postnet_type == "conv":
                self.postnet = PostNet()

        self.loss = FastSpeech2Loss(postnet=self.has_postnet, postnet_type=self.postnet_type, blur=self.has_blur)


        for model in [
            self.linear,
            self.decoder,
            self.variance_adaptor,
            self.encoder,
            self.phone_embedding,
        ]:
            for param in model.parameters():
                param.requires_grad = False


        self.is_wandb_init = False

        self.sizes = [80, 40, 20, 10, 5]
        # add noise?
        size_transforms = [
            VT.Compose([
                VT.Resize((s, s)),
                # VT.CenterCrop((s, s)),
                # VT.Normalize(0.5, 0.5),
            ])
            for s in self.sizes
        ]
        self.size_list = lambda x: [s(x) for s in size_transforms]

    def forward(self, phones, speakers, pitch=None, energy=None, duration=None, mel=None):
        phones = phones.to(self.device)
        speakers = speakers.to(self.device)
        if pitch is not None:
            pitch = pitch.to(self.device)
        if energy is not None:
            energy = energy.to(self.device)
        if duration is not None:
            duration = duration.to(self.device)
        src_mask = phones.eq(0)
        output = self.phone_embedding(phones)
        output = self.positional_encoding(output)
        output = self.encoder(output, src_key_padding_mask=src_mask)

        if not self.has_dvector:
            speaker_out = (
                self.speaker_embedding(speakers)
                .reshape(-1, 1, output.shape[-1])
                .repeat_interleave(phones.shape[1], dim=1)
            )
        else:
            speaker_out = (
                speakers
                .reshape(-1, 1, output.shape[-1])
                .repeat_interleave(phones.shape[1], dim=1)
            )

        output = output + speaker_out
        variance_out = self.variance_adaptor(
            output, src_mask, pitch, energy, duration, self.tgt_max_length
        )
        output = variance_out["x"]
        output = self.positional_encoding(output)
        output = self.decoder(output, src_key_padding_mask=variance_out["tgt_mask"])
        output = self.linear(output)

        if self.has_postnet:
            if self.postnet_type == 'conv':
                postnet_output = self.postnet(output)
                final_output = postnet_output + output
            elif self.postnet_type == 'gan':
                batch_size = output.shape[0]
                postnet_output = []
                final_output = output.detach().clone()
                width = 80
                for i in range(batch_size):
                    conditional_item = output[i].detach().clone()

                    out_len = conditional_item.shape[0]//width
                    out_mod = conditional_item.shape[0]%width

                    if out_mod != 0:
                        batch_size = out_len + 1
                    else:
                        batch_size = out_len

                    # 0.9 -> label smoothing gan trick
                    real_label = torch.full((batch_size, 1), .9, dtype=output.dtype).to(self.device)
                    fake_label = torch.full((batch_size, 1), 0, dtype=output.dtype).to(self.device)

                    noise = torch.randn([batch_size, 100, 1, 1]).to(self.device)

                    targets = []
                    conditionals = []
                    for j in range(out_len):
                        if mel is not None:
                            patch = mel[i][width*j:width*(j+1)].unsqueeze(0)
                            targets.append(self.size_list(patch))
                        patch = conditional_item[width*j:width*(j+1)].unsqueeze(0)
                        conditionals.append(self.size_list(patch))
                    if out_mod != 0:
                        if mel is not None:
                            patch = mel[i][-width:].unsqueeze(0)
                            patchs = self.size_list(patch)
                            targets.append(patchs)
                        patch = conditional_item[-width:].unsqueeze(0)
                        conditionals.append(self.size_list(patch))

                    mel_mean = self.train_ds.stats['mel_mean']
                    mel_std = self.train_ds.stats['mel_std']

                    conditionals = [torch.stack([c[i] for c in conditionals]).to(self.device) for i in range(len(self.sizes))]
                    conditionals = [(c - mel_mean) / mel_std for c in conditionals]

                    if mel is not None:
                        targets = [torch.stack([t[i] for t in targets]).to(self.device) for i in range(len(self.sizes))]
                        targets = [(t - mel_mean) / mel_std for t in targets]
                        real_output = self.postnet_disc(targets, conditionals)
                    else:
                        real_output = None

                    # Train with fake.
                    fake = [f for c, f in zip(conditionals, self.postnet_gen(noise, conditionals))] # conditionals + 

                    fake_output_d = self.postnet_disc([f.detach() for f in fake], conditionals)
                    # update G network
                    fake_output_g = self.postnet_disc(fake, conditionals)

                    if mel is None:
                        for j in range(out_len):
                            final_output[i][width*j:width*(j+1)] = (fake[0].detach().clone().squeeze()[j]) * mel_std + mel_mean
                        if out_mod != 0:
                            final_output[i][-out_mod:] = (fake[0].detach().clone().squeeze(1)[-1][-out_mod:]) * mel_std + mel_mean
                    else:
                        final_output = None

                    if real_output is not None:
                        real_output_clone = real_output.clone().squeeze().unsqueeze(-1)
                    else:
                        real_output_clone = None

                    result = {
                        'real_label': real_label.clone(),
                        'fake_label': fake_label.clone(),
                        'real_output': real_output_clone,
                        'fake_output_d': fake_output_d.clone().squeeze().unsqueeze(-1),
                        'fake_output_g': fake_output_g.clone().squeeze().unsqueeze(-1),
                    }

                    if real_output is not None:
                        result['fake'] = fake
                        result['real'] = targets
                        if not self.training:
                            result['accuracy'] = accuracy_score(
                                [1]*len(real_output)+[0]*len(fake_output_d),
                                [int(x) for x in list((real_output.cpu()*(1/0.9)).round())]+[int(x) for x in list((fake_output_d.cpu()*(1/0.9)).round())]
                            )

                    postnet_output.append(result)
        else:
            postnet_output = None
            final_output = output

        return (
            (
                output,
                variance_out["pitch"],
                variance_out["energy"],
                variance_out["log_duration"],
                variance_out["duration_rounded"],
                postnet_output,
                final_output,
            ),
            src_mask,
            variance_out["tgt_mask"],
        )

    def training_step(self, batch):
        logits, src_mask, tgt_mask = self(
            batch["phones"],
            batch["speaker"],
            batch["pitch"],
            batch["energy"],
            batch["duration"],
            batch["mel"],
        )
        if not self.has_blur:
            truth = (
                batch["mel"],
                batch["pitch"],
                batch["energy"],
                batch["duration"],
            )
        else:
            truth = (
                batch["mel"],
                batch["pitch"],
                batch["energy"],
                batch["duration"],
                batch["mel_blur"],
            )
        loss = self.loss(logits, truth, src_mask, tgt_mask, self.tgt_max_length)
        log_dict = {
            "train/total_loss": loss[0].item(),
            "train/mel_loss": loss[1].item(),
            "train/pitch_loss": loss[2].item(),
            "train/energy_loss": loss[3].item(),
            "train/duration_loss": loss[4].item(),
        }
        if self.has_postnet:
            if self.postnet_type == 'conv':
                log_dict['train/postnet_loss'] = loss[5].item()
            elif self.postnet_type == 'gan':
                log_dict['train/postnet_dis_real'] = loss[5].item()
                log_dict['train/postnet_dis_fake'] = loss[6].item()
                log_dict['train/postnet_gen'] = loss[7].item()
        self.log_dict(
            log_dict,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        return loss[0]

    def validation_step(self, batch, batch_idx):
        preds, src_mask, tgt_mask = self(
            batch["phones"],
            batch["speaker"],
            batch["pitch"],
            batch["energy"],
            batch["duration"],
            batch["mel"],
        )
        if not self.has_blur:
            truth = (
                batch["mel"],
                batch["pitch"],
                batch["energy"],
                batch["duration"],
            )
        else:
            truth = (
                batch["mel"],
                batch["pitch"],
                batch["energy"],
                batch["duration"],
                batch["mel_blur"],
            )
        loss = self.loss(preds, truth, src_mask, tgt_mask, self.tgt_max_length)
        log_dict = {
            "eval/total_loss": loss[0].item(),
            "eval/mel_loss": loss[1].item(),
            "eval/pitch_loss": loss[2].item(),
            "eval/energy_loss": loss[3].item(),
            "eval/duration_loss": loss[4].item(),
        }
        if self.has_postnet:
            if self.postnet_type == 'conv':
                log_dict['eval/postnet_loss'] = loss[5].item()
            elif self.postnet_type == 'gan':
                log_dict['eval/postnet_dis_real'] = loss[5].item()
                log_dict['eval/postnet_dis_fake'] = loss[6].item()
                log_dict['eval/postnet_gen'] = loss[7].item()
                log_dict['eval/accuracy'] = np.mean([x['accuracy'] for x in preds[-2]])
        self.log_dict(
            log_dict,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        if batch_idx == 0 and self.trainer.is_global_zero:
            if not self.is_wandb_init:
                wandb.init(project='LightningFastSpeech', group='DDP')
                self.is_wandb_init = True
            old_src_mask = src_mask
            old_tgt_mask = tgt_mask
            preds, src_mask, tgt_mask = self(batch["phones"], batch["speaker"])
            mels, pitchs, energys, _, durations, postnet_output, final_mels = preds
            log_data = []
            for i in range(len(mels)):
                if i == 0:
                    all_preds, _, _ = self(
                        batch["phones"],
                        batch["speaker"],
                        batch["pitch"],
                        batch["energy"],
                        batch["duration"],
                        batch["mel"],
                    )
                    postnet_output = all_preds[-2]
                    for j in range(len(postnet_output[0]['fake'])):
                        for k in range(len(self.sizes)):
                            real_img = postnet_output[0]['real'][k][j]
                            fake_img = postnet_output[0]['fake'][k][j]
                            # print(f'gan_examples_{self.sizes[k]}x{self.sizes[k]}')
                            # print('real', real_img.mean(), real_img.std(), real_img.min(), real_img.max())
                            # print('fake', fake_img.mean(), fake_img.std(), fake_img.min(), fake_img.max())
                            wandb.log({f'gan_examples_{self.sizes[k]}x{self.sizes[k]}': wandb.Image(make_grid([real_img, fake_img]))})
                if i >= 10:
                    break
                mel = final_mels[i][~tgt_mask[i]].cpu()
                true_mel = batch["mel"][i][~old_tgt_mask[i]].cpu()
                if len(mel) == 0:
                    print('predicted 0 length output, this is normal at the beginning of training')
                    continue
                pred_fig = self.valid_ds.plot(
                    {
                        "mel": mel,
                        "pitch": pitchs[i].cpu(),
                        "energy": energys[i].cpu(),
                        "duration": durations[i][~src_mask[i]].cpu(),
                        "phones": batch["phones"][i]
                    }
                )
                true_fig = self.valid_ds.plot(
                    {
                        "mel": true_mel.cpu(),
                        "pitch": batch["pitch"][i].cpu(),
                        "energy": batch["energy"][i].cpu(),
                        "duration": batch["duration"][i].cpu()[~old_src_mask[i]],
                        "phones": batch["phones"][i],
                    }
                )
                pred_audio = self.synth(mel.to("cuda:0"))[0]
                true_audio = self.synth(true_mel.to("cuda:0"))[0]
                log_data.append(
                    [
                        batch["text"][i],
                        wandb.Image(pred_fig),
                        wandb.Image(true_fig),
                        wandb.Audio(pred_audio, sample_rate=22050),
                        wandb.Audio(true_audio, sample_rate=22050),
                    ]
                )
            table = wandb.Table(
                data=log_data,
                columns=[
                    "text",
                    "predicted_mel",
                    "true_mel",
                    "predicted_audio",
                    "true_audio",
                ],
            )
            wandb.log({"examples": table})
        
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr, betas=(0.5, 0.999))

        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer,
        #     max_lr=self.lr,
        #     steps_per_epoch=len(self.train_ds) // self.batch_size,
        #     epochs=self.epochs,
        # )

        # sched = {
        #     "scheduler": self.scheduler,
        #     "interval": "step",
        # }

        return self.optimizer
        #return [self.optimizer], [sched]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            collate_fn=self.train_ds.collate_fn,
            num_workers=cpus,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            collate_fn=self.valid_ds.collate_fn,
            num_workers=cpus,
        )
