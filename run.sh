CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/charades/SeqPAN.json --debug


CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/anet/ActionFormer.yaml --debug


video_list[0]["feats"].shape
torch.Size([512, 188])