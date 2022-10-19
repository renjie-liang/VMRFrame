CUDA_VISIBLE_DEVICES=2,3 python main.py --config ./config/charades/main_c3dFT_cpl.json
# CUDA_VISIBLE_DEVICES=2 python main_vsl.py --config ./config/charades/main_c3dFT_vsl.json --eval --checkpoint /storage/rjliang/3_ActiveLearn/VSLNet/ckpt_t7/vslnet_charades_new_128_rnn/model/vslnet_6402.t7


# 
CUDA_VISIBLE_DEVICES=0,1 python main.py --config ./config/charades/main_c3dFT_cpl.json --checkpoint ./ckpt/tmp/weakly_best.pkl


CUDA_VISIBLE_DEVICES=2,3 python main.py --config ./config/charades/main_c3dFT_cpl_half.json --suffix half --checkpoint ./ckpt/tmp/weakly_best.pkl


CUDA_VISIBLE_DEVICES=2,3 python main.py --config ./config/tacos/main_i3d.json


CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/charades/main_i3d_re1_old.json \
            --checkpoint ckpt/charades_re1_old/best.pkl --suffix re1_old

CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/charades/main_i3d_re1.json --suffix re1 
            \ --eval --checkpoint ckpt/charades_re1/best.pkl -

CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/anet/main_i3d_re0.json --suffix re0

CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/anet/main_i3d_re0.json --suffix re0


CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/tacos/main_i3d_cpl.json
