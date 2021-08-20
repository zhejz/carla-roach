import torch as th
from torch.nn import functional as F
from torch.distributions import Beta, Normal

EPS = 1e-5


class BranchedLoss():
    def __init__(self, branch_weights, action_weights, speed_weight, value_weight, features_weight, l1_loss,
                 action_kl, action_agg, action_mll):
        if l1_loss:
            self.loss = F.l1_loss
        else:
            self.loss = F.mse_loss

        self.branch_weights = branch_weights
        self.action_weights = action_weights
        self.speed_weight = speed_weight
        self.value_weight = value_weight
        self.features_weight = features_weight
        self.action_kl = action_kl
        self.action_mll = action_mll
        self.action_agg = action_agg

        self.n_branches = len(branch_weights)
        self.n_actions = len(action_weights)

    def forward(self, outputs, supervisions, commands):
        commands.clamp_(0, self.n_branches-1)
        # action loss
        branch_masks = self._get_branch_masks(commands, self.n_branches, self.n_actions)
        action_loss = 0.0
        if 'action_branches' in outputs:
            # deterministc output
            for i in range(self.n_branches):
                masked_action = supervisions['action']*branch_masks[i]
                masked_action_pred = outputs['action_branches'][i]*branch_masks[i]
                for j in range(self.n_actions):
                    loss_ij = self.loss(masked_action_pred[:, j], masked_action[:, j])
                    action_loss += loss_ij * self.branch_weights[i] * self.action_weights[j]
        else:
            if self.action_kl:
                # probability output, kl loss
                kl_loss = 0.0
                for i in range(self.n_branches):
                    dist_sup = Beta(supervisions['action_mu'], supervisions['action_sigma'])
                    dist_pred = Beta(outputs['mu_branches'][i], outputs['sigma_branches'][i])
                    kl_div = th.distributions.kl_divergence(dist_sup, dist_pred)*branch_masks[i]

                    for j in range(self.n_actions):
                        loss_ij = th.mean(kl_div[:, j])
                        kl_loss += loss_ij * self.branch_weights[i] * self.action_weights[j]
                action_loss += kl_loss

            if self.action_agg is not None:
                # aggrevated
                agg_loss = 0.0
                for i in range(self.n_branches):
                    mu = th.clamp(outputs['mu_branches'][i], min=EPS)
                    sigma = th.clamp(outputs['sigma_branches'][i], min=EPS)
                    dist_pred = Beta(mu, sigma)
                    # dist_pred = Beta(outputs['mu_branches'][i], outputs['sigma_branches'][i])
                    scaled_action = (supervisions['action']+1)/2.0
                    scaled_action = th.clamp(scaled_action, EPS, 1-EPS)
                    log_prob = dist_pred.log_prob(scaled_action)
                    aggrevated_loss = -1.0*th.exp(log_prob - log_prob.detach())*supervisions['advantage'].unsqueeze(1)

                    for j in range(self.n_actions):
                        loss_ij = th.mean(aggrevated_loss[:, j])
                        agg_loss += loss_ij * self.branch_weights[i] * self.action_weights[j]
                action_loss += self.action_agg*agg_loss

            if self.action_mll is not None:
                # maximum log likelihood
                mll_loss = 0.0
                for i in range(self.n_branches):
                    mu = th.clamp(outputs['mu_branches'][i], min=EPS)
                    sigma = th.clamp(outputs['sigma_branches'][i], min=EPS)
                    dist_pred = Beta(mu, sigma)
                    scaled_action = (supervisions['action']+1)/2.0
                    scaled_action = th.clamp(scaled_action, EPS, 1-EPS)
                    log_prob = -1.0*dist_pred.log_prob(scaled_action)

                    for j in range(self.n_actions):
                        loss_ij = th.mean(log_prob[:, j])
                        mll_loss += loss_ij * self.branch_weights[i] * self.action_weights[j]
                action_loss += self.action_mll*mll_loss

        # speed_loss
        speed_loss = th.zeros_like(action_loss)
        if 'pred_speed' in outputs and 'speed' in supervisions:
            speed_loss = self.loss(outputs['pred_speed'], supervisions['speed']) * self.speed_weight

        # value_loss
        value_loss = th.zeros_like(action_loss)
        if 'pred_value' in outputs and 'value' in supervisions:
            value_loss = F.mse_loss(outputs['pred_value'], supervisions['value']) * self.value_weight

        # features_loss
        features_loss = th.zeros_like(action_loss)
        if 'pred_features' in outputs and 'features' in supervisions:
            features_loss = F.mse_loss(outputs['pred_features'], supervisions['features']) * self.features_weight

        return action_loss, speed_loss, value_loss, features_loss

    @staticmethod
    def _get_branch_masks(commands, n_branches, n_actions):
        controls_masks = []

        for i in range(n_branches):
            mask = (commands == i).float()
            mask = th.cat([mask] * n_actions, 1)
            controls_masks.append(mask)

        return controls_masks
