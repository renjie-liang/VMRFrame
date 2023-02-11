CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/charades/SeqPAN.json --debug
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/anet/SeqPAN.json

CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/anet/ActionFormer.yaml --debug