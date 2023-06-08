from transformers import AutoModel, AutoConfig
from models.poolers import get_pooling_layer_cv
from models.criterion import get_criterion
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


def get_timm_backbone(config, pretrained):
    backbone = timm.create_model(
        config.model.backbone_type,
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
        in_chans=1,
    )
    return backbone


def get_model(config, pretrained=True):
    if config.model.architecture == 'psi_cnn':
        print('Using psi_cnn model')
        return PsiModel(config, pretrained=pretrained)
    elif config.model.architecture == 'tattaka_sed':
        print('Using tattaka_sed model')
        return TattakaModel(config, pretrained=pretrained)
    elif config.model.architecture == 'kaeruru_sed':
        print('Using kaeruru_sed model')
        return KaeruruModel(config, pretrained=pretrained)
    
    print('Using custom model')
    return CustomModel(config, pretrained=pretrained)


def load_state(model, config, filepaths):
    pretrained_dir = f'{config.experiment_name}_pretrain'
    # fn = f'fold_{-1}_chkp.pth'
    # checkpoint_path = filepaths.models_dir / 'exp107_pretrain' / 'chkp' / fn
    fn = f'fold_{0}_chkp.pth'
    checkpoint_path = filepaths.models_dir / 'exp091_pretrain' / 'chkp' / fn

    print('Loading state from checkpoint: ', checkpoint_path)

    state = torch.load(checkpoint_path)

    if config.model.architecture == 'psi_cnn':
        drop_keys = ["bn2.num_batches_tracked"]
        state = {
            key.replace('model.', ''): value for key, value in state['model'].items()
            if ('clf.' not in key) and ('pool.p' not in key) and (key.replace('model.', '') not in drop_keys)
        }
        model.model.load_state_dict(state)
    else:
        drop_keys = ["conv_head.weight", "bn2.weight", "bn2.bias", "bn2.running_mean", "bn2.running_var",
                    "bn2.num_batches_tracked"]
        state = {
            key.replace('model.', ''): value for key, value in state['model'].items()
            if ('clf.' not in key) and ('pool.p' not in key) and (key.replace('model.', '') not in drop_keys)
        }
        model.model.load_state_dict(state)
    return model


class CustomModel(nn.Module):
    def __init__(self, config, pretrained=True):
        super(CustomModel, self).__init__()

        self.model = get_timm_backbone(config, pretrained=pretrained)
        self.pool = get_pooling_layer_cv(config)

        self.clf = nn.Linear(self.model.num_features, len(config.dataset.labels))
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

        self.config = config

    def forward(self, inputs):
        feature = self.model(inputs['spec'])
        feature = self.pool(feature)
        feature = feature[:, :, 0, 0]
        logits = self.clf(feature)

        loss = None
        if 'labels' in inputs.keys():
            weight = inputs['weight']
            loss = self.criterion(logits, inputs['labels'])
            
            loss = (loss.mean(dim=1) * weight) / weight.sum()
            loss = loss.sum()

        return logits, loss


class PsiModel(nn.Module):
    def __init__(self, config, pretrained=True):
        super(PsiModel, self).__init__()

        self.model = get_timm_backbone(config, pretrained=pretrained)
        self.pool = get_pooling_layer_cv(config)

        self.clf = nn.Linear(self.model.num_features, len(config.dataset.labels))
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.config = config

    def forward(self, inputs):
        spec = inputs['spec']
        b, p, c, f, t = spec.shape
        spec = spec.reshape(b*p, c, f, t)  # (batch_size * parts, channels, freq, time)
        spec = spec.permute(0, 1, 3, 2)    # (batch_size * parts, channels, time, freq)

        feature = self.model(spec)         # (batch_size * parts, feats, time, freq)
        out_f, t, f = feature.shape[1], feature.shape[2], feature.shape[3]

        feature = feature.reshape((b, p, out_f, t, f))  # (batch_size, parts, feats, time, freq)
        feature = feature.permute(0, 2, 1, 3, 4)        # (batch_size, feats, parts, time, freq)
        feature = feature.reshape(b, out_f, p*t, f)     # (batch_size, feats, parts*time, freq)

        feature = self.pool(feature)
        feature = feature[:, :, 0, 0]  # (batch_size, feats)
        logits = self.clf(feature)     # (batch_size, num_classes)

        loss = None
        if 'labels' in inputs.keys():
            weight = inputs['weight']
            loss = self.criterion(logits, inputs['labels'])

            loss = (loss.mean(dim=1) * weight) / weight.sum()
            loss = loss.sum()

        return logits, loss


def gem_freq(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)


class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem_freq(x, p=self.p, eps=self.eps)


class AttHead(nn.Module):
    def __init__(
            self, in_chans, p=0.5, num_class=264, train_period=15.0, infer_period=5.0
    ):
        super().__init__()
        self.train_period = train_period
        self.infer_period = infer_period
        self.pooling = GeMFreq()

        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, 512),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(
            in_channels=512,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.fix_scale = nn.Conv1d(
            in_channels=512,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)  # (bs, time, ch)

        feat = self.dense_layers(feat).permute(0, 2, 1)  # (bs, 512, time)
        # print(feat.shape)

        time_att = torch.tanh(self.attention(feat))

        assert self.train_period >= self.infer_period

        if self.training or self.train_period == self.infer_period:  # or True 
            # print('train')
            clipwise_pred = torch.sum(
                torch.sigmoid(self.fix_scale(feat)) * torch.softmax(time_att, dim=-1),
                dim=-1,
            )  # sum((bs, 24, time), -1) -> (bs, 24)
            logits = torch.sum(
                self.fix_scale(feat) * torch.softmax(time_att, dim=-1),
                dim=-1,
            )
        else:
            # print('eval')
            clipwise_pred_long = torch.sum(
                torch.sigmoid(self.fix_scale(feat)) * torch.softmax(time_att, dim=-1),
                dim=-1,
            )  # sum((bs, 24, time), -1) -> (bs, 24)

            feat_time = feat.size(-1)
            start = feat_time / 2 - feat_time * (self.infer_period / self.train_period) / 2
            end = start + feat_time * (self.infer_period / self.train_period)

            start = int(start)
            end = int(end)

            feat = feat[:, :, start:end]
            att = torch.softmax(time_att[:, :, start:end], dim=-1)

            # print(feat.shape)

            clipwise_pred = torch.sum(
                torch.sigmoid(self.fix_scale(feat)) * att,
                dim=-1,
            )
            logits = torch.sum(
                self.fix_scale(feat) * att,
                dim=-1,
            )
            time_att = time_att[:, :, start:end]
        return (
            logits,
            clipwise_pred,
            self.fix_scale(feat).permute(0, 2, 1),
            time_att.permute(0, 2, 1),
        )


def get_timm_backbone(config, pretrained):
    backbone = timm.create_model(
        config.model.backbone_type,
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
        in_chans=1,
    )
    return backbone


class TattakaModel(nn.Module):
    def __init__(
            self,
            config,
            pretrained,
    ):
        super().__init__()

        # self.model = get_timm_backbone(config, pretrained)
        self.model = timm.create_model(
            config.model.backbone_type, features_only=True, pretrained=False, in_chans=1
        )
        encoder_channels = self.model.feature_info.channels()
        dense_input = encoder_channels[-1]
        self.head = AttHead(
            dense_input,
            p=config.model.dropout,
            num_class=len(config.dataset.labels),
            train_period=config.dataset.train_duration,
            infer_period=config.dataset.valid_duration,
        )
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs):
        spec = inputs['spec']

        feats = self.model(spec)
        logits, output_clip, output_frame, output_attention = self.head(feats[-1])

        loss = None
        if 'labels' in inputs.keys():
            weight = inputs['weight']

            loss_clip = self.criterion(torch.logit(output_clip), inputs['labels'])
            # loss_frame = self.criterion(output_frame.max(1)[0], inputs['labels'])
            # loss = loss_clip + loss_frame * 0.5
            loss = loss_clip
            loss = (loss.mean(dim=1) * weight) / weight.sum()
            loss = loss.sum()

        return output_clip, loss


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output


class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class KaeruruModel(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()

        self.bn0 = nn.BatchNorm2d(config.dataset.img_size[0])

        base_model = timm.create_model(config.model.backbone_type, pretrained=pretrained, in_chans=1)
        layers = list(base_model.children())[:-2]
        self.model = nn.Sequential(*layers)

        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        else:
            in_features = base_model.classifier.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, len(config.dataset.labels), activation="sigmoid")

        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, inputs):
        x = inputs['spec']  # (batch_size, 3, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = x.transpose(2, 3)

        x = self.model(x)

        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)

        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        #         output_dict = {
        #             'framewise_output': framewise_output,
        #             'clipwise_output': clipwise_output,
        #             'logit': logit,
        #             'framewise_logit': framewise_logit,
        #         }

        loss = None
        if 'labels' in inputs.keys():
            weight = inputs['weight']

            loss_clip = self.criterion(logit, inputs['labels'])
            # loss_frame =  self.criterion(framewise_logit, inputs['labels'])
            # loss = loss_clip * 0.5 + loss_frame * 0.5
            loss = loss_clip

            loss = (loss.mean(dim=1) * weight) / weight.sum()
            loss = loss.sum()

        return logit, loss