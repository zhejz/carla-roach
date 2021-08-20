import logging
import torch.nn as nn
import torch as th
import copy
from collections import OrderedDict

from .networks.branching import Branching
from .networks.fc import FC
from .networks.join import Join
from .networks import resnet

log = logging.getLogger(__name__)


class CoILICRA(nn.Module):

    def __init__(self, im_shape, input_states, acc_as_action,
                 value_as_supervision, action_distribution, dim_features_supervision, rl_ckpt=None,
                 freeze_value_head=False, freeze_action_head=False,
                 resnet_pretrain=True,
                 perception_output_neurons=512,
                 measurements_neurons=[128, 128],
                 measurements_dropouts=[0.0, 0.0],
                 join_neurons=[512],
                 join_dropouts=[0.0],
                 speed_branch_neurons=[256, 256],
                 speed_branch_dropouts=[0.0, 0.5],
                 value_branch_neurons=[256, 256],
                 value_branch_dropouts=[0.0, 0.5],
                 number_of_branches=6,
                 branches_neurons=[256, 256],
                 branches_dropouts=[0.0, 0.5],
                 squash_outputs=True,
                 perception_net='resnet34',
                 ):
        super(CoILICRA, self).__init__()

        self._init_kwargs = copy.deepcopy(locals())
        del self._init_kwargs['self']
        del self._init_kwargs['__class__']

        if th.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.number_of_branches = number_of_branches

        rl_state_dict = None
        if rl_ckpt is not None:
            try:
                rl_state_dict = th.load(rl_ckpt, map_location='cpu')['policy_state_dict']
                log.info(f'Load rl_ckpt: {rl_ckpt}')
            except:
                log.info(f'Unable to load rl_ckpt: {rl_ckpt}')

        # preception block
        self.dim_features_supervision = dim_features_supervision
        if dim_features_supervision > 0:
            join_neurons[-1] = dim_features_supervision

        self._video_input = 'video' in perception_net
        self.perception = resnet.get_model(perception_net, im_shape,
                                           num_classes=perception_output_neurons, pretrained=resnet_pretrain)

        # measurement block
        input_states_len = 0
        if 'speed' in input_states:
            input_states_len += 1
        if 'vec' in input_states:
            input_states_len += 2
        if 'cmd' in input_states:
            input_states_len += 6
        self.measurements = FC(params={'neurons': [input_states_len] + measurements_neurons,
                                       'dropouts': measurements_dropouts,
                                       'end_layer': nn.ReLU})

        # concat/join block
        self.join = Join(params={'after_process':
                                 FC(params={'neurons':
                                            [measurements_neurons[-1] + perception_output_neurons] + join_neurons,
                                            'dropouts': join_dropouts,
                                            'end_layer': nn.ReLU}),
                                 'mode': 'cat'})

        if squash_outputs:
            end_layer_speed = nn.Sigmoid
            end_layer_action = nn.Tanh
        else:
            end_layer_speed = None
            end_layer_action = None

        # speed branch
        self.speed_branch = FC(params={'neurons': [perception_output_neurons] + speed_branch_neurons + [1],
                                       'dropouts': speed_branch_dropouts + [0.0],
                                       'end_layer': end_layer_speed})

        # value branch
        self.value_as_supervision = value_as_supervision
        if value_as_supervision:
            self.value_branch = FC(params={'neurons': [join_neurons[-1]] + value_branch_neurons + [1],
                                           'dropouts': value_branch_dropouts + [0.0],
                                           'end_layer': None})
            if rl_state_dict is not None:
                self._load_state_dict(self.value_branch, rl_state_dict, 'value_head')
                log.info(f'Load rl_ckpt state dict for value_head.')
            if freeze_value_head:
                for param in self.value_branch.parameters():
                    param.requires_grad = False
                log.info('Freeze value head weights.')

        # action branches
        self.action_distribution = action_distribution
        assert action_distribution in ['beta', 'beta_shared', None]
        if action_distribution == 'beta':
            dim_out = 2
            mu_branch_vector = []
            sigma_branch_vector = []
            for i in range(number_of_branches):
                mu_branch_vector.append(FC(params={'neurons': [join_neurons[-1]] + branches_neurons + [dim_out],
                                                   'dropouts': branches_dropouts + [0.0],
                                                   'end_layer': nn.Softplus}))
                sigma_branch_vector.append(FC(params={'neurons': [join_neurons[-1]] + branches_neurons + [dim_out],
                                                      'dropouts': branches_dropouts + [0.0],
                                                      'end_layer': nn.Softplus}))
            self.mu_branches = Branching(mu_branch_vector)
            self.sigma_branches = Branching(sigma_branch_vector)
        elif action_distribution == 'beta_shared':
            # shared branches_neurons
            dim_out = 2
            mu_branch_vector = []
            sigma_branch_vector = []

            for i in range(number_of_branches):
                policy_head = FC(params={'neurons': [join_neurons[-1]] + branches_neurons,
                                         'dropouts': branches_dropouts,
                                         'end_layer': nn.ReLU})
                dist_mu = nn.Sequential(nn.Linear(branches_neurons[-1], dim_out), nn.Softplus())
                dist_sigma = nn.Sequential(nn.Linear(branches_neurons[-1], dim_out), nn.Softplus())

                if rl_state_dict is not None:
                    self._load_state_dict(policy_head, rl_state_dict, 'policy_head')
                    self._load_state_dict(dist_mu, rl_state_dict, 'dist_mu')
                    self._load_state_dict(dist_sigma, rl_state_dict, 'dist_sigma')
                    log.info(f'Load rl_ckpt state dict for policy_head, dist_mu, dist_sigma.')

                mu_branch_vector.append(nn.Sequential(policy_head, dist_mu))
                sigma_branch_vector.append(nn.Sequential(policy_head, dist_sigma))

            self.mu_branches = Branching(mu_branch_vector)
            self.sigma_branches = Branching(sigma_branch_vector)
            if freeze_action_head:
                for param in self.mu_branches.parameters():
                    param.requires_grad = False
                for param in self.sigma_branches.parameters():
                    param.requires_grad = False
                log.info('Freeze action head weights.')

        else:
            if acc_as_action:
                dim_out = 2
            else:
                dim_out = 3
            branch_fc_vector = []
            for i in range(number_of_branches):
                branch_fc_vector.append(FC(params={'neurons': [join_neurons[-1]] + branches_neurons + [dim_out],
                                                   'dropouts': branches_dropouts + [0.0],
                                                   'end_layer': end_layer_action}))
            self.branches = Branching(branch_fc_vector)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, im, state):
        '''
        im: (b, c, t, h, w) np.uint8
        state: (n,)
        '''
        if not self._video_input:
            b, c, t, h, w = im.shape
            im = im.view(b, c*t, h, w)

        """ ###### APPLY THE PERCEPTION MODULE """
        x = self.perception(im)

        """ ###### APPLY THE MEASUREMENT MODULE """
        m = self.measurements(state)

        """ Join measurements and perception"""
        j = self.join(x, m)

        # build outputs dict
        outputs = {'pred_speed': self.speed_branch(x)}

        if self.value_as_supervision:
            outputs['pred_value'] = self.value_branch(j)

        if self.action_distribution is None:
            outputs['action_branches'] = self.branches(j)
        else:
            outputs['mu_branches'] = self.mu_branches(j)
            outputs['sigma_branches'] = self.sigma_branches(j)

        if self.dim_features_supervision > 0:
            outputs['pred_features'] = j

        return outputs

    def forward_branch(self, command, im, state):
        with th.no_grad():
            im_tensor = im.unsqueeze(0).to(self.device)
            state_tensor = state.unsqueeze(0).to(self.device)
            command_tensor = command.to(self.device)
            command_tensor.clamp_(0, self.number_of_branches-1)
            outputs = self.forward(im_tensor, state_tensor)
            if self.action_distribution == 'beta' or self.action_distribution=='beta_shared':
                action_branches = self._get_action_beta(outputs['mu_branches'], outputs['sigma_branches'])
                action = self.extract_branch(action_branches, command_tensor)
            else:
                action = self.extract_branch(outputs['action_branches'], command_tensor)
        return action[0].cpu().numpy(), outputs['pred_speed'].item()

    @staticmethod
    def extract_branch(action_branches, branch_number):
        '''
        action_branches: list, len=num_branches, (batch_size, action_dim)
        '''
        output_vec = th.stack(action_branches)
        # output_vec: (num_branches, batch_size, action_dim)

        if len(branch_number) > 1:
            branch_number = th.squeeze(branch_number.type(th.LongTensor))
        else:
            branch_number = branch_number.type(th.LongTensor)
        # branch_number: (batch_size,)

        branch_number = th.stack([branch_number, th.LongTensor(range(0, len(branch_number)))])

        return output_vec[branch_number[0], branch_number[1], :]

    @property
    def init_kwargs(self):
        return self._init_kwargs

    @classmethod
    def load(cls, path):
        if th.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        saved_variables = th.load(path, map_location=device)
        # Create policy object
        saved_variables['policy_init_kwargs']['resnet_pretrain'] = False
        saved_variables['policy_init_kwargs']['rl_ckpt'] = None
        model = cls(**saved_variables['policy_init_kwargs'])
        # Load weights
        log.info(f'load state dict : {path}')
        model.load_state_dict(saved_variables['policy_state_dict'])
        model.to(device)
        return model

    @staticmethod
    def _get_action_beta(alpha_branches, beta_branches):
        action_branches = []
        for alpha, beta in zip(alpha_branches, beta_branches):
            x = th.zeros_like(alpha)
            x[:, 1] += 0.5
            mask1 = (alpha > 1) & (beta > 1)
            x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)

            mask2 = (alpha <= 1) & (beta > 1)
            x[mask2] = 0.0

            mask3 = (alpha > 1) & (beta <= 1)
            x[mask3] = 1.0

            # mean
            mask4 = (alpha <= 1) & (beta <= 1)
            x[mask4] = alpha[mask4]/(alpha[mask4]+beta[mask4])

            x = x * 2 - 1

            action_branches.append(x)

        return action_branches

    @staticmethod
    def _load_state_dict(il_net, rl_state_dict, key_word):
        rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
        il_keys = il_net.state_dict().keys()
        assert len(rl_keys) == len(il_net.state_dict().keys()), f'mismatch number of layers loading {key_word}'
        new_state_dict = OrderedDict()
        for k_il, k_rl in zip(il_keys, rl_keys):
            new_state_dict[k_il] = rl_state_dict[k_rl]
        il_net.load_state_dict(new_state_dict)