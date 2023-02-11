# SeqPAN by PyTroch


 A SeqPAN model was reproduced by PyTorch for the paper "Parallel Attention Network with Sequence Matching for Video Grounding".

The original TensorFLow implementation is in https://github.com/IsaacChanghau/SeqPAN. 

<br>

## Performance
| version             | R1@.03 | R1@0.5 | R1@0.7 | mIoU  |
| -------------       | ------ | ------ | ------ | ----- |
| TensorFlow(paper)   | 61.65  | 45.50  | 28.37  | 45.11 |
| PyTorch(ours)       | 64.55  | 46.91  | 28.79  | 46.64 |

<br>

## Prepareation
To download the video feature, please refer to https://github.com/IsaacChanghau/SeqPAN. It is possible to save the video feature in any preferred location.

Adjust "paths" value within the ./config/anet/SeqPAN.yaml.

<br>

## Quick Start
```python

# train
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/anet/SeqPAN.yaml
# test
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/anet/SeqPAN.yaml --eval
# debug
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/anet/SeqPAN.yaml --debug

```