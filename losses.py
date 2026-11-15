import torch
import torch.nn.functional as F
from torch import nn


class Lossess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        evidence_alpha,
        logits,
        labels,
        lambda_1,
        lambda_2,
        label_onehot,
        epochss=None,
        return_output="alpha",
        loss="IEDL",
        compute_loss=False,
    ):
       
        self.loss_mse_ = evidence_alpha.new_tensor(0.0)
        self.loss_var_ = evidence_alpha.new_tensor(0.0)
        self.loss_kl_ = evidence_alpha.new_tensor(0.0)
        self.loss_fisher_ = evidence_alpha.new_tensor(0.0)

        if lambda_1 < 0 or lambda_2 < 0:
            raise ValueError(
                "lambda_1 and lambda_2 must be non-negative. "
                "Use --kl_c 1 for the main UA-SNN setting or --kl_c 0 "
                "for the without-KL ablation."
            )

        if compute_loss:
            if loss == "IEDL":
                (
                    self.loss_mse_,
                    self.loss_var_,
                    self.loss_fisher_,
                ) = self.compute_fisher_mse(label_onehot, evidence_alpha)
            elif loss == "EDL":
                self.loss_mse_, self.loss_var_ = self.compute_mse(
                    label_onehot, evidence_alpha
                )
            elif loss == "DEDL":
                self.loss_mse_, self.loss_var_ = self.compute_mse(
                    label_onehot, evidence_alpha
                )
                _, _, self.loss_fisher_ = self.compute_fisher_mse(
                    label_onehot, evidence_alpha
                )
            else:
                raise NotImplementedError(f"Unknown evidential loss: {loss}")

            target_concentration = 1.0
            incorrect_class_alpha = (
                (evidence_alpha - target_concentration) * (1 - label_onehot)
                + target_concentration
            )
            self.loss_kl_ = self.compute_kl_loss(
                incorrect_class_alpha, labels, target_concentration
            )

            self.grad_loss = (
                self.loss_mse_
                + self.loss_var_
                + lambda_1 * self.loss_fisher_
                + lambda_2 * self.loss_kl_
            )

        if return_output == "loss":
            if not compute_loss:
                raise ValueError("return_output='loss' requires compute_loss=True")
            return self.grad_loss
        if return_output == "hard":
            return logits.argmax(dim=-1)
        if return_output == "soft":
            return F.softmax(logits, dim=-1)
        if return_output == "alpha":
            return evidence_alpha
        raise ValueError(f"Unknown return_output: {return_output}")

    @staticmethod
    def compute_mse(label_onehot, evidence_alpha):
        alpha_0 = torch.sum(evidence_alpha, dim=-1, keepdim=True)
        loss_mse = (label_onehot - evidence_alpha / alpha_0).pow(2).sum(-1).mean()
        loss_var = (
            evidence_alpha
            * (alpha_0 - evidence_alpha)
            / (alpha_0 * alpha_0 * (alpha_0 + 1))
        ).sum(-1).mean()
        return loss_mse, loss_var

    @staticmethod
    def compute_fisher_mse(label_onehot, evidence_alpha):
        alpha_0 = torch.sum(evidence_alpha, dim=-1, keepdim=True)
        trigamma_alpha = torch.polygamma(1, evidence_alpha)
        trigamma_alpha_0 = torch.polygamma(1, alpha_0)
        gap = label_onehot - evidence_alpha / alpha_0

        loss_mse = (gap.pow(2) * trigamma_alpha).sum(-1).mean()
        loss_var = (
            evidence_alpha
            * (alpha_0 - evidence_alpha)
            * trigamma_alpha
            / (alpha_0 * alpha_0 * (alpha_0 + 1))
        ).sum(-1).mean()
        negative_log_det_fisher = -(
            torch.log(trigamma_alpha).sum(-1)
            + torch.log(1.0 - (trigamma_alpha_0 / trigamma_alpha).sum(-1))
        ).mean()
        return loss_mse, loss_var, negative_log_det_fisher

    @staticmethod
    def compute_kl_loss(
        alphas,
        labels,
        target_concentration,
        concentration=1.0,
        epsilon=1e-8,
    ):
        if target_concentration < 1.0:
            concentration = target_concentration

        target_alphas = torch.ones_like(alphas) * concentration
        target_alphas += torch.zeros_like(alphas).scatter_(
            -1, labels.unsqueeze(-1), target_concentration - 1
        )

        alpha_0 = torch.sum(alphas, dim=-1, keepdim=True)
        target_alpha_0 = torch.sum(target_alphas, dim=-1, keepdim=True)
        alpha_0_term = torch.lgamma(alpha_0 + epsilon) - torch.lgamma(
            target_alpha_0 + epsilon
        )
        alpha_0_term = torch.where(
            torch.isfinite(alpha_0_term), alpha_0_term, torch.zeros_like(alpha_0_term)
        )

        alphas_term = torch.sum(
            torch.lgamma(target_alphas + epsilon)
            - torch.lgamma(alphas + epsilon)
            + (alphas - target_alphas)
            * (torch.digamma(alphas + epsilon) - torch.digamma(alpha_0 + epsilon)),
            dim=-1,
            keepdim=True,
        )
        alphas_term = torch.where(
            torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term)
        )
        return torch.squeeze(alpha_0_term + alphas_term).mean()
