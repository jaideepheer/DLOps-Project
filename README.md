# DL-Ops Assignment 4

> Jaideep Singh Heer (M20CS056)

## Code

All project code can be found at: https://github.com/jaideepheer/DLOps-Assignment_4

## Wandb Data

The following report link shows these results on wandb:
https://wandb.ai/dlops-course/task_tuberculosis/reports/Assignment-4-Wandb-Report--VmlldzoxNzkzNTE2?accessToken=asuj8fqtningkpdta4nfu8t9tuzdfnw3rtn7enpk72ej70zy5igyyied94dz0yoh

The following link grants access to the entire wandb project data:
https://wandb.ai/dlops-course/task_tuberculosis

## Model Training

| S.No. | Model | GPU -  Train Time / Accuracy | GPU + Optimization-  Train Time / Accuracy |
|---|---|---|---|
| 1 | Resnet50 | 208s / 0.9896 | 180s / 0.9926 |
| 2 | Resnet18 | 201s / 0.9985 | 89s / 0.9970 |

## Model Inferencing

| S.No. | Model, Batchsize | PyTorch Throughput | ONNX Throughput | TRT FP32 Throughput | TRT FP16 Throughput | TRT INT8 Throughput |
|---|---|---|---|---|---|---|
| 1. | Resnet50 | 467.094 | 99.896 | 870.515 | 1390.212 | 1933.696 |
| 2. | Resnet18 | 1543.057 | 788.615 | 1474.283 | 2040.892 | 2987.476 |

<br/>

| S.No. | Model, Batchsize | PyTorch  Latency | ONNX Latency | TRT FP32 Latency | TRT FP16 Latency | TRT INT8 Latency |
|---|---|---|---|---|---|---|
| 1. | Resnet50 | 0.06525 | 0.3051 | 0.03501 | 0.02192 | 0.01576 |
| 2. | Resnet18 | 0.01975 | 0.03865 | 0.02067 | 0.01493 | 0.0102 |

## Memory space

| S.No. | Model | PyTorch  Model Size | ONNX  Model Size | TRT FP32 Model Size | TRT FP16  Model Size | TRT INT8  Model Size |
|---|---|---|---|---|---|---|
| 1. | Resnet50 | 95MB | 94MB | 95MB | 48MB | 29MB |
| 2. | Resnet18 | 44MB | 44MB | 45MB | 23MB | 12MB |