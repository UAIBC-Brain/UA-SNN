import  torch
import numpy as np
from torch import nn

class Lossess(nn.Module):
 def __init__(self,):  # Random seed for init. int
        super().__init__()

 def forward(self, evi_alp_,out_fr, label,fisher_c,kl_c,label_onehot,epochss, labels_=None, return_output='alpha',loss ='IEDL', compute_loss=False, ):
    # assert not (label is None and compute_loss)

    self.loss_mse_ = torch.tensor(0.0)
    self.loss_var_ = torch.tensor(0.0)
    self.loss_kl_ = torch.tensor(0.0)
    self.loss_fisher_ = torch.tensor(0.0)
    self.loss = loss
    # Calculate loss
    if compute_loss:
        if self.loss == 'IEDL':
            # IEDL -> fisher_mse
            self.loss_mse_, self.loss_var_, self.loss_fisher_ = self.compute_fisher_mse(label_onehot, evi_alp_)
            # print('loss_mse_',self.loss_mse_)
            # print('loss_var_',self.loss_var_)
            # print('loss_fisher_',self.loss_fisher_)
        elif self.loss == 'EDL':
            # EDL -> mse
            self.loss_mse_, self.loss_var_ = self.compute_mse(label_onehot, evi_alp_)
        elif self.loss == 'DEDL':
            self.loss_mse_, self.loss_var_ = self.compute_mse(label_onehot, evi_alp_)
            _, _, self.loss_fisher_ = self.compute_fisher_mse(label_onehot, evi_alp_)
        else:
            raise NotImplementedError

        self.fisher_c= fisher_c
        self.kl_c = kl_c
        self.target_con = 1.0
        labels_ = label

        self.evi_alp_ = (evi_alp_ - self.target_con) * (1 - label_onehot) + self.target_con
        self.loss_kl_ = self.compute_kl_loss(self.evi_alp_, labels_, self.target_con)
        # print('loss_kl_',self.loss_kl_)
        # print('fisher_c',self.fisher_c)
        # print('kl_c',self.kl_c)
        if self.kl_c == -1:
            regr = np.minimum(1.0, epochss / 10.)
            self.grad_loss = self.loss_mse_ + self.loss_var_ + self.fisher_c * self.loss_fisher_ + regr * self.loss_kl_
        else:
            self.grad_loss = self.loss_mse_ + self.loss_var_ + self.fisher_c * self.loss_fisher_ + self.kl_c * self.loss_kl_
        # print('grad_loss', self.grad_loss)
    if return_output == 'hard':
        # return max(out_fr)
        return self.predict(out_fr)
    elif return_output == 'soft':
        # return softmax(out_fr)
        return self.softmax(out_fr)
    elif return_output == 'alpha':
        return self.evi_alp_
    else:
        raise AssertionError


 def compute_mse(self, label_onehot, evi_alp_):
    evi_alp0_ = torch.sum(evi_alp_, dim=-1, keepdim=True)

    loss_mse_ = (label_onehot - evi_alp_ / evi_alp0_).pow(2).sum(-1).mean()
    loss_var_ = (evi_alp_ * (evi_alp0_ - evi_alp_) / (evi_alp0_ * evi_alp0_ * (evi_alp0_ + 1))).sum(
        -1).mean()

    return loss_mse_, loss_var_


 def compute_fisher_mse(self, label_onehot, evi_alp_):
    evi_alp0_ = torch.sum(evi_alp_, dim=-1, keepdim=True)

    gamma1_alp = torch.polygamma(1, evi_alp_)
    gamma1_alp0 = torch.polygamma(1, evi_alp0_)

    gap = label_onehot - evi_alp_ / evi_alp0_

    loss_mse_ = (gap.pow(2) * gamma1_alp).sum(-1).mean()

    loss_var_ = (evi_alp_ * (evi_alp0_ - evi_alp_) * gamma1_alp / (evi_alp0_ * evi_alp0_ * (evi_alp0_ + 1))).sum(
        -1).mean()

    loss_det_fisher_ = - (torch.log(gamma1_alp).sum(-1) + torch.log(1.0 - (gamma1_alp0 / gamma1_alp).sum(-1))).mean()

    return loss_mse_, loss_var_, loss_det_fisher_


 def compute_kl_loss(self, alphas, labels, target_concentration, concentration=1.0, epsilon=1e-8):
    # TODO: Need to make sure this actually works right...
    # todo: so that concentration is either fixed, or on a per-example setup

    # Create array of target (desired) concentration parameters
    if target_concentration < 1.0:
        concentration = target_concentration

    target_alphas = torch.ones_like(alphas) * concentration
    target_alphas += torch.zeros_like(alphas).scatter_(-1, labels.unsqueeze(-1), target_concentration - 1)#散播的操作，将类别赋给target_alphas

    alp0 = torch.sum(alphas, dim=-1, keepdim=True)
    target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)#实际上target_alp0是总类别数k，将类别target_alphas求和

    alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
    alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
    assert torch.all(torch.isfinite(alp0_term)).item()

    alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                            + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                          torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
    alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
    assert torch.all(torch.isfinite(alphas_term)).item()

    loss = torch.squeeze(alp0_term + alphas_term).mean()

    return loss

 # def step(self):
 #        self.optimizer.zero_grad()
 #        self.grad_loss.backward()
 #        self.optimizer.step()