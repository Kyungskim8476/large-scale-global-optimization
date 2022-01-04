import math
import torch
from torch.optim import Optimizer

class HyAdamC_Basic(Optimizer):
    device = None
    gamma = 1.0

    def __init__(self, params, lr=1e-3, k_dof=1.0, betas=(0.9, 0.999), eps=1e-8, gamma=1.0):
        self.gamma = gamma
        defaults = dict(lr=lr, k_dof=k_dof, betas=betas, eps=eps, weight_decay=weight_decay)
        super(HyAdamC_Basic, self).__init__(params, defaults)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.gamma == 0:
            print("Fatal error: gamma is zero!")
            exit()

    def __setstate__(self, state):
        super(HyAdamC_Basic, self).__setstate__(state)

    # closure (callable, optional): A closure that reevaluates the model and returns the loss.
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Do not support sparse gradients.')

                state = self.state[p]
                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['rho_inf'] = 2. / (1. - beta2) - 1.  # For the init-term velocity function
                    state['first_moment'] = torch.zeros_like(p.data)  # For the previous momenums and gradients
                    state['second_moment'] = torch.zeros_like(p.data)
                    state['previous_grad'] = torch.zeros_like(p.data)

                    # Process the outlier gradients (Initialization)
                    state['W_t'] = ( torch.tensor(0.) + beta1 / (1.0 - beta1) ).to(self.device)
                    state['dim'] = p.data.numel()
                    if not group["k_dof"] == math.inf:
                        state['dof'] = ( torch.tensor(0.) + group["k_dof"] * state['dim'] ).to(self.device)

                # Read the previous momentums and gradients
                first_moment, second_moment, previous_grad = state['first_moment'], state['second_moment'], state['previous_grad']
                state['step'] += 1

                # Process the outlier gradients (update the coefficient values to minimize the outlier gradients)
                Wt = state['W_t']
                beta1, beta2 = group['betas']
                if group["k_dof"] == math.inf:
                    betaw = beta1
                else:
                    wt = grad.sub(first_moment).pow_(2).div_(second_moment.add(group['eps'])).sum()
                    wt.add_(state['dof']).pow_(-1).mul_(state['dim'] + state['dof'])
                    betaw = Wt.div(Wt.add(wt))
                    Wt.mul_(2.0 - 1.0/beta1).add_(wt)

                # Compute the first momentum
                first_moment.mul_(betaw).add_(grad.mul(1 - betaw))

                # Long-term velocity control function
                grad_residual = grad - first_moment
                second_moment.mul_(beta2).addcmul_(1 - beta2, grad_residual, grad_residual)

                # Compute the bias-corrections
                bias_correction_first = 1 - beta1 ** state['step']
                bias_correction_second = 1 - beta2 ** state['step']

                # Short-term velocity control function (Basic version)
                diff = abs(previous_grad - grad)
                dfc = 1. / (1. + torch.exp( -self.gamma*diff ))
                state['previous_grad'] = grad.clone()

                # Init-term velocity control function for the warm-up strategy
                state['rho_t'] = state['rho_inf'] - 2.*state['step']*(beta2 ** state['step']) / bias_correction_second
                if state['rho_t'] > 4:
                    lt = math.sqrt(bias_correction_second) / ( torch.sqrt(second_moment) + group['eps'] )
                    rt = math.sqrt(
                        ( (state['rho_t']-4.)*(state['rho_t']-2.)*state['rho_inf'] )  /
                        ( (state['rho_inf']-4.)*(state['rho_inf']-2.)*state['rho_t']  )
                    )
                    # Combine all velocity control functions and coefficients to determine the next search direction.
                    p.data = p.data -group['lr']*rt*(first_moment * dfc / bias_correction_first)*lt
                else:
                    # Combine all velocity control functions and coefficients to determine the next search direction.
                    p.data = p.data -group['lr']*(first_moment * dfc / bias_correction_first)

        return loss
