import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.Encoder import Encoder


class InfoCTM(nn.Module):
    def __init__(self, args, trans_e2c):
        super().__init__()

        self.args = args

        self.trans_e2c = torch.as_tensor(trans_e2c).float()
        self.trans_e2c = nn.Parameter(self.trans_e2c, requires_grad=False)
        self.trans_c2e = self.trans_e2c.T

        self.num_topic = args.num_topic
        self.temperature = args.temperature

        self.encoder_en = Encoder(args.vocab_size_en, args.num_topic, args.en1_units, args.dropout)
        self.encoder_cn = Encoder(args.vocab_size_cn, args.num_topic, args.en1_units, args.dropout)

        self.a = 1 * np.ones((1, int(args.num_topic))).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T), requires_grad=False)
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / args.num_topic))).T + (1.0 / (args.num_topic * args.num_topic)) * np.sum(1.0 / self.a, 1)).T), requires_grad=False)

        self.decoder_bn_en = nn.BatchNorm1d(args.vocab_size_en, affine=True)
        self.decoder_bn_en.weight.requires_grad = False
        self.decoder_bn_cn = nn.BatchNorm1d(args.vocab_size_cn, affine=True)
        self.decoder_bn_cn.weight.requires_grad = False

        self.phi_en = nn.Parameter(nn.init.xavier_uniform_(torch.empty((args.num_topic, args.vocab_size_en))))
        self.phi_cn = nn.Parameter(nn.init.xavier_uniform_(torch.empty((args.num_topic, args.vocab_size_cn))))

        self.compute_pos_neg()

    def pos_neg_mono_mask(self, embeddings, _type):
        norm_embed = F.normalize(embeddings)
        cos_sim = torch.matmul(norm_embed, norm_embed.T)

        if _type == 'pos':
            pos_mask = (cos_sim >= self.args.pos_threshold).float()
            return pos_mask

    def translation_mask(self, mask, trans_dict_matrix):
        # V1 x V2
        trans_mask = torch.matmul(mask, trans_dict_matrix)
        return trans_mask

    def compute_pos_neg(self):
        # Ve x Ve
        pos_mono_mask_en = self.pos_neg_mono_mask(torch.as_tensor(self.args.pretrain_word_embeddings_en), _type='pos')
        # Vc x Vc
        pos_mono_mask_cn = self.pos_neg_mono_mask(torch.as_tensor(self.args.pretrain_word_embeddings_cn), _type='pos')

        # Ve x Vc
        pos_trans_mask_en = self.translation_mask(pos_mono_mask_en, self.trans_e2c)
        pos_trans_mask_cn = self.translation_mask(pos_mono_mask_cn, self.trans_c2e)

        neg_trans_mask_en = (pos_trans_mask_en <= 0).float()
        neg_trans_mask_cn = (pos_trans_mask_cn <= 0).float()

        self.pos_trans_mask_en = nn.Parameter(pos_trans_mask_en, requires_grad=False)
        self.pos_trans_mask_cn = nn.Parameter(pos_trans_mask_cn, requires_grad=False)
        self.neg_trans_mask_en = nn.Parameter(neg_trans_mask_en, requires_grad=False)
        self.neg_trans_mask_cn = nn.Parameter(neg_trans_mask_cn, requires_grad=False)

    def MutualInfo(self, anchor_feature, contrast_feature, mask, neg_mask, temperature):

        anchor_dot_contrast = torch.div(
            torch.matmul(F.normalize(anchor_feature, dim=1), F.normalize(contrast_feature, dim=1).T),
            temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * neg_mask
        sum_exp_logits = exp_logits.sum(1, keepdim=True)

        log_prob = logits - torch.log(sum_exp_logits + torch.exp(logits) + 1e-10)
        mean_log_prob = -(mask * log_prob).sum()
        return mean_log_prob

    def get_beta(self):
        beta_en = self.phi_en
        beta_cn = self.phi_cn
        return beta_en, beta_cn

    def get_theta(self, x, lang):
        theta, mu, logvar = getattr(self, f'encoder_{lang}')(x)

        if self.training:
            return theta, mu, logvar
        else:
            return mu

    def decode(self, theta, beta, lang):
        bn = getattr(self, f'decoder_bn_{lang}')
        d1 = F.softmax(bn(torch.matmul(theta, beta)), dim=1)
        return d1

    def forward(self, x_en, x_cn):
        theta_en, mu_en, logvar_en = self.get_theta(x_en, lang='en')
        theta_cn, mu_cn, logvar_cn = self.get_theta(x_cn, lang='cn')

        beta_en, beta_cn = self.get_beta()

        loss = 0.
        tmp_rst_dict = dict()

        x_recon_en = self.decode(theta_en, beta_en, lang='en')
        x_recon_cn = self.decode(theta_cn, beta_cn, lang='cn')
        loss_en = self.loss_function(x_recon_en, x_en, mu_en, logvar_en)
        loss_cn = self.loss_function(x_recon_cn, x_cn, mu_cn, logvar_cn)

        loss = loss_en + loss_cn
        tmp_rst_dict['loss_en'] = loss_en
        tmp_rst_dict['loss_cn'] = loss_cn

        fea_en = beta_en.T
        fea_cn = beta_cn.T

        loss_TAMI = self.MutualInfo(fea_en, fea_cn, self.pos_trans_mask_en, self.neg_trans_mask_en, temperature=self.temperature)
        loss_TAMI += self.MutualInfo(fea_cn, fea_en, self.pos_trans_mask_cn, self.neg_trans_mask_cn, temperature=self.temperature)

        loss_TAMI = loss_TAMI / (self.pos_trans_mask_en.sum() + self.pos_trans_mask_cn.sum())

        loss_TAMI = self.args.weight_MI * loss_TAMI
        loss += loss_TAMI
        tmp_rst_dict['loss_TAMI'] = loss_TAMI

        rst_dict = {
            'loss': loss,
        }

        rst_dict.update(tmp_rst_dict)

        return rst_dict

    def loss_function(self, recon_x, x, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.num_topic)

        RECON = -(x * (recon_x + 1e-10).log()).sum(1)

        LOSS = (RECON + KLD).mean()
        return LOSS
