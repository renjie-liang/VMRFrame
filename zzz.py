import torch
from models.ActionFormerlib.meta_archs import ConvTransformerBackbone
backbone = ConvTransformerBackbone(
                **{
                    'n_in' : 512,
                    'n_embd' : 256,
                    'n_head': 4,
                    'n_embd_ks': 3,
                    'max_len': 192,
                    'arch' : [2, 2, 5],
                    'mha_win_size': [7, 7, 7, 7, 7, -1],
                    'scale_factor' : 2,
                    'with_ln' : True,
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : 0.0,
                    'path_pdrop' : 0.1,
                    'use_abs_pe' : True,
                    'use_rel_pe' : False
                }
            )

batched_inputs = torch.rand([2, 512, 192])
batched_masks = torch.ones([2, 1, 192], dtype=bool)
res = backbone(batched_inputs, batched_masks)