CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/anet/main_i3d_seqpan.json
CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/charades/main_i3d_seqpan.json
CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/tacos/main_i3d_seqpan.json


CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/anet/BAN.json --eval
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/charades/BAN.json --eval


CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/anet/BaseFast.json --eval
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/charades/BaseFast.json --eval



CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/anet/SeqPAN.json
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/charades/SeqPAN.json