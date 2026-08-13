"""Minimal extract of comfy/ldm/modules/diffusionmodules/model.py.

Only the helpers used by the vendored Lightricks modules are kept.
"""

import torch


def torch_cat_if_needed(xl, dim):
    xl = [x for x in xl if x is not None and x.shape[dim] > 0]
    if len(xl) > 1:
        return torch.cat(xl, dim)
    elif len(xl) == 1:
        return xl[0]
    else:
        return None
