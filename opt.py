import jax
# jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])


import numpy as np
from protes import protes


def opt(func, d, n, m, k=100, k_top=10, k_gd=1, lr=5.E-2, r=5, is_max=True):
    info = {}
    i, y = protes(func, d, n, m, k, k_top, k_gd, lr, r, info=info,
        is_max=is_max, log=True, with_info_i_opt_list=False)
    return i, y, info['m_opt_list'], info['y_opt_list']
