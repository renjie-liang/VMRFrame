import torch
from models.ActionFormerlib.meta_archs import ConvTransformerBackbone
L = 64
backbone = ConvTransformerBackbone(
                **{
                    'n_in' : 128,
                    'n_embd' : 128,
                    'n_head': 4,
                    'n_embd_ks': 3,
                    'max_len': 64,
                    'arch' : [2, 2, 3],
                    'mha_win_size': [5, 5, 5, -1],
                    'scale_factor' : 2,
                    'with_ln' : True,
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : 0.0,
                    'path_pdrop' : 0.1,
                    'use_abs_pe' : True,
                    'use_rel_pe' : False
                }
            )

batched_inputs = torch.rand([2, 128, L])
batched_masks = torch.ones([2, 1, L], dtype=bool)
res = backbone(batched_inputs, batched_masks)

for i in res:
    for j in i:
        print(j.shape)









# import torch
# from models.ActionFormerlib.meta_archs import ConvTransformerBackbone
# L = 64
# backbone = ConvTransformerBackbone(
#                 **{
#                     'n_in' : 1024,
#                     'n_embd' : 256,
#                     'n_head': 4,
#                     'n_embd_ks': 3,
#                     'max_len': L,
#                     'arch' : [2, 2, 3],
#                     'mha_win_size': [5, 5, 5, -1],
#                     'scale_factor' : 2,
#                     'with_ln' : True,
#                     'attn_pdrop' : 0.0,
#                     'proj_pdrop' : 0.0,
#                     'path_pdrop' : 0.1,
#                     'use_abs_pe' : True,
#                     'use_rel_pe' : False
#                 }
#             )

# batched_inputs = torch.rand([2, 1024, L])
# batched_masks = torch.ones([2, 1, L], dtype=bool)
# res = backbone(batched_inputs, batched_masks)

# for i in res:
#     for j in i:
#         print(j.shape)