from conf import *
from model import *
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import os
import nvidia.dali as dali
import nvidia.dali.types as types
import os

NUM_CLASSES = args.n_classes
INPUT_SHAPE = (3, args.image_size, args.image_size)
OPT_BATCH = 8
MAX_BATCH = 128
MIN_BATCH = 1


def get_pbtxt_onnx():
    return f"""
name: "{args.model_name}_onnx"
platform: "onnxruntime_onnx"
dynamic_batching {{
preferred_batch_size: [8,16,32]
max_queue_delay_microseconds: 100
}}

max_batch_size: {MAX_BATCH}
input {{
  name: "input"
  data_type: TYPE_FP32
  dims: [ {', '.join(map(str, INPUT_SHAPE))} ]
}}
output {{
  name: "output"
  data_type: TYPE_FP32
  dims: [{NUM_CLASSES}]
}}


instance_group [
    {{
        count: 1
        kind: KIND_GPU
        gpus: [0]
    }}
]

default_model_filename: "model.onnx"
"""


def get_pbtxt_torch():
    return f"""
name: "{args.model_name}_torch"
platform: "pytorch_libtorch"
dynamic_batching {{
preferred_batch_size: [8,16,32]
max_queue_delay_microseconds: 100
}}

max_batch_size: {MAX_BATCH}
input {{
  name: "input__0"
  data_type: TYPE_FP32
  dims: [ {', '.join(map(str, INPUT_SHAPE))} ]
}}
output {{
  name: "output__0"
  data_type: TYPE_FP32
  dims: [{NUM_CLASSES}]
}}


instance_group [
    {{
        count: 1
        kind: KIND_GPU
        gpus: [0]
    }}
]

default_model_filename: "model.pt"
"""


def get_pbtxt_fp(bits: int):
    post = f"trt_fp{bits}" if bits != 8 else "int8"
    return f"""
name: "{args.model_name}_{post}"
platform: "tensorrt_plan"
dynamic_batching {{
preferred_batch_size: [8,16,32]
max_queue_delay_microseconds: 100
}}

max_batch_size: {MAX_BATCH}
input {{
  name: "input"
  data_type: TYPE_FP32
  dims: [ {', '.join(map(str, INPUT_SHAPE))} ]
}}
output {{
  name: "output"
  data_type: TYPE_FP32
  dims: {NUM_CLASSES}
}}


instance_group [
    {{
        count: 1
        kind: KIND_GPU
        gpus: [0]
    }}
]

default_model_filename: "model.plan"
"""

def get_pbtxt_dali():
    return f"""
name: "dali_{args.model_name}"
backend: "dali"
max_batch_size: {MAX_BATCH}
input [
{{
    name: "DALI_INPUT_0"
    data_type: TYPE_UINT8
    dims: [ -1 ]
}}
]
 
output [
{{
    name: "DALI_OUTPUT_0"
    data_type: TYPE_FP32
    dims: [ {', '.join(map(str, INPUT_SHAPE))} ]
}}
]

"""


def create_dali_pipeline():
    @dali.pipeline_def(batch_size=256, num_threads=4, device_id=0)
    def pipe():
        images = dali.fn.external_source(device="cpu", name="DALI_INPUT_0")
        images = dali.fn.decoders.image(images, device="mixed", output_type=types.RGB)
        images = dali.fn.resize(images, resize_x=INPUT_SHAPE[-2], resize_y=INPUT_SHAPE[-1])
        images = dali.fn.crop_mirror_normalize(images,
                                            dtype=types.FLOAT,
                                            output_layout="CHW",
                                            crop=INPUT_SHAPE[1:],
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        return images
    
    os.makedirs(f"{args.triton_dir}{args.model_name}_dali/1")
    pipe().serialize(filename=f"{args.triton_dir}{args.model_name}_dali/1/model.dali")

    # write pbtxt
    with open(f"{args.triton_dir}{args.model_name}_dali/config.pbtxt", "w") as f:
        f.write(get_pbtxt_dali())

def main():
    for d in ('torch', 'onnx', 'trt_fp32', 'trt_fp16', 'trt_int8', 'dali'):
        os.system(f'rm -rf {args.triton_dir}{args.model_name}_{d}/')

    # create dali model
    create_dali_pipeline()

    os.makedirs(f"{args.triton_dir}{args.model_name}_torch/1")
    os.makedirs(f"{args.triton_dir}{args.model_name}_onnx/1")
    os.makedirs(f"{args.triton_dir}{args.model_name}_trt_fp32/1")
    os.makedirs(f"{args.triton_dir}{args.model_name}_trt_fp16/1")
    os.makedirs(f"{args.triton_dir}{args.model_name}_trt_int8/1")

    # save config.pbtxt
    with open(f"{args.triton_dir}{args.model_name}_torch/config.pbtxt", "w") as f:
        f.write(get_pbtxt_torch())
    with open(f"{args.triton_dir}{args.model_name}_onnx/config.pbtxt", "w") as f:
        f.write(get_pbtxt_onnx())
    with open(f"{args.triton_dir}{args.model_name}_trt_fp32/config.pbtxt", "w") as f:
        f.write(get_pbtxt_fp(32))
    with open(f"{args.triton_dir}{args.model_name}_trt_fp16/config.pbtxt", "w") as f:
        f.write(get_pbtxt_fp(16))
    with open(f"{args.triton_dir}{args.model_name}_trt_int8/config.pbtxt", "w") as f:
        f.write(get_pbtxt_fp(8))
    
    JIT_MODEL_PATH = f'{args.triton_dir}{args.model_name}_torch/1/model.pt'
    ONNX_MODEL_PATH = f'{args.triton_dir}{args.model_name}_onnx/1/model.onnx'
    TRT_MODEL_PATH = f'{args.triton_dir}{args.model_name}_trt_fp32/1/model.plan'
    TRT_MODEL_PATH_FP16 = f'{args.triton_dir}{args.model_name}_trt_fp16/1/model.plan'
    TRT_MODEL_PATH_INT8 = f'{args.triton_dir}{args.model_name}_trt_int8/1/model.plan'
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU_ID

    model = build_model()
    print(f"Loading weights for model {args.model_name} from: {args.out_weight}")
    model.load_state_dict(torch.load(args.out_weight))
    model.eval()

    if args.channels_last:
        example = torch.randn((args.batch_size, *INPUT_SHAPE), dtype=torch.float32, device=args.device).to(memory_format=torch.channels_last)
    else:
        example = torch.randn((args.batch_size, *INPUT_SHAPE), dtype=torch.float32, device=args.device)

    script = torch.jit.trace(model, example)
    script.save(JIT_MODEL_PATH)


    if args.channels_last:
        x = torch.randn((1, *INPUT_SHAPE), dtype=torch.float32, device=args.device).to(memory_format=torch.channels_last)
    else:
        x = torch.randn((1, *INPUT_SHAPE), dtype=torch.float32, device=args.device)

    torch.onnx.export(model,                       # model being run
                      x,                           # model input (or a tuple for multiple inputs)
                      ONNX_MODEL_PATH,             # Path to saved onnx model
                      export_params=True,          # store the trained parameter weights inside the model file
                      opset_version=13,            # the ONNX version to export the model to
                      input_names = ['input'],     # the model's input names
                      output_names = ['output'],   # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
    basic_shape = 'x'.join(map(str, INPUT_SHAPE))
    
    os.system(f'trtexec --onnx={ONNX_MODEL_PATH} --explicitBatch --workspace=40000 --optShapes=input:{OPT_BATCH}x{basic_shape} --maxShapes=input:{MAX_BATCH}x{basic_shape} --minShapes=input:{MIN_BATCH}x{basic_shape} --saveEngine={TRT_MODEL_PATH}')

    os.system(f'trtexec --onnx={ONNX_MODEL_PATH} --explicitBatch --workspace=40000 --optShapes=input:{OPT_BATCH}x{basic_shape} --maxShapes=input:{MAX_BATCH}x{basic_shape} --minShapes=input:{MIN_BATCH}x{basic_shape} --saveEngine={TRT_MODEL_PATH_FP16} --fp16')
    
    os.system(f'trtexec --onnx={ONNX_MODEL_PATH} --explicitBatch --workspace=40000 --optShapes=input:{OPT_BATCH}x{basic_shape} --maxShapes=input:{MAX_BATCH}x{basic_shape} --minShapes=input:{MIN_BATCH}x{basic_shape} --saveEngine={TRT_MODEL_PATH_INT8} --int8')
if __name__ == '__main__':
    main()
