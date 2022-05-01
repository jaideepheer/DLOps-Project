# DL-Ops Project

> Jaideep Singh Heer (M20CS056)
> Chiranjeev (P21CS007)
> Himanshu Kumar Anand (M21MA011)

## Wandb Logs

https://wandb.ai/dlops-course/task_SOCOFing/table?workspace=user-jaideep

## Prepare data

First, prepare the SOCOFing data by following the steps at [./data/Readme.md](./data)

## Configure launch scripts

Before running anything, please ensure that the images used in `bash/launch-job.sh`, `bash/launch-triton.sh` and `bash/launch-client.sh` use compatable images form nvidia image repositories.

Use the [nvidia frameworks support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) to select the image suitable for your system and CUDA version.

Before train/predict/convert, please ensure `bash/launch-job.sh` has correct configs. For eg. you will probably want to change the `rootpath` to something suitable for your system.

## Train

To train on a DGX-2 server using the default `resnet50` model,
```
cd bash
source ./launch_train.sh
```

Alternatively use `source ./launch-job.sh` with `mode` as `train` instead, to use custom models and settings.

## Convert

To convert models in `saved_models/SOCOFing` using `tensorrt` and `onnx`,
```
cd bash
source ./launch_convert.sh
```

Alternatively use `source ./launch-job.sh` with `mode` as `convert` instead, to use custom models and settings.

## Deployment

### Deploy tritonrt server

...

### Deploy webapp client

...
