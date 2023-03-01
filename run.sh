anet
AUTH:TEST | R1I3: 61.65 R1I5: 45.50 R1I7: 28.37 mIoU: 45.11
QINY:TEST | R1I3: 58.11 R1I5: 41.64 R1I7: 26.75 mIoU: 42.95
RENJ:TEST | R1I3: 64.55 R1I5: 46.91 R1I7: 28.79 mIoU: 46.64

charades
INFO:TEST | R1I3: 73.84 R1I5: 60.86 R1I7: 41.34 mIoU: 53.92
INFO:TEST | R1I3: 69.76 R1I5: 55.19 R1I7: 35.99 mIoU: 50.12 
INFO:TEST | R1I3: 72.39 R1I5: 57.28 R1I7: 36.24 mIoU: 51.36


CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/charades/SeqPAN.yaml --debug
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/anet/SeqPAN.yal
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/anet/ActionFormer.yaml --debug

configs.model.vlen


CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/anet/SeqPAN.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/anet/BaseFast.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/anet/OneTeacher.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/anet/OneTeacher_SoftLabel.yaml

CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/anet/SeqPAN.yaml --checkpoint ./ckpt/anet_/best_SeqPAN_2832.pkl --eval



CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/anet/BAN.yaml --debug --eval

CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/anet/SeqPAN.yaml --debug
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/anet/OneTeacher_c3d.yaml
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/anet/BaseFast_c3d.yaml --debug
CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/anet/BaseFast_CCA_PreTrain_c3d.yaml --debug


CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/charades/BackBone.yaml --debug
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/charades/SeqPAN.yaml --debug
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/charades/OneTeacher.yaml --debug
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/charades/MultiTeacher.yaml --debug
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/charades/ActionFormer.yaml --debug


CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/charades/SeqPAN_SimilarSentence.yaml --debug
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/charades/BackBoneBertSentence.yaml --debug

