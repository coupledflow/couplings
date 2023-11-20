# USES TORCHDIFFEQ https://github.com/rtqichen/torchdiffeq

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint #odeint_adjoint as odeint
import math


class PFlowRHS(nn.Module):
    def __init__(self, config, b, y, cond):
        super(PFlowRHS, self).__init__()
        self.config = config
        self.b = b
        # y and cond are stored here because when integrating the ode,
        # they do not change. The ode integrator has an interface
        # where any inputs must be incremented in the ode integration.
        # this is a hack around that. An alternative is to include
        # y and cond in the state, and set their time derivatives to zero.
        self.y = y
        self.cond = cond
        
    def forward(self, t, states):
        (zt,) = states
        t_arr = torch.ones(zt.shape[0]).type_as(zt) * t
        dzt = self.b(zt = zt, t = t_arr, y = self.y, cond = self.cond)
        return (dzt,)
             
class PFlowIntegrator:
        
    def __init__(self, config):
        
        self.config = config

    def __call__(self, b, z0, y, cond, T_min, T_max, steps, method='dopri5', return_last = True):

        c = self.config
        
        rhs = PFlowRHS(config = c, 
            b = b,
            y = y, 
            cond = cond,
        )

        t = torch.linspace(
            T_min, T_max, steps
        ).type_as(z0)

        int_args = {
            'method': method, 
            'atol': c.integration_atol, 
            'rtol': c.integration_rtol,
        }

        (z,) = odeint(rhs, (z0,), t, **int_args)
        if return_last:
            return z[-1]
        else:
            return z

