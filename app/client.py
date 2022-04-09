import os
from PIL import Image
import json
import numpy as np
import pandas as pd
import time
import streamlit as st
import torchvision

import tritonclient.http as http_client
import tritonclient.grpc as grpc_client

imagenet_mean = np.array([[[0.485]], [[0.456]], [[0.406]]])
imagenet_std = np.array([[[0.229]], [[0.224]], [[0.225]]])

model_configuration = {
    "http_url": "172.25.0.42:30369",
    "grpc_url": "172.25.0.42:30522",
    "model_version": "1",
    "verbose": False,
    "dtype": "FP32",
    "resnet50_torch": {
        "name": "resnet50_torch",
        "input": "input__0",
        "output": "output__0"
    },
    "resnet50_onnx": {
        "name": "resnet50_onnx",
        "input": "input",
        "output": "output"
    },
    "resnet50_trt_fp32": {
        "name": "resnet50_trt_fp32",
        "input": "input",
        "output": "output"
    },
    "resnet50_trt_fp16": {
        "name": "resnet50_trt_fp16",
        "input": "input",
        "output": "output"
    },
    "resnet50_trt_int8": {
        "name": "resnet50_trt_int8",
        "input": "input",
        "output": "output"
    }
}

with open('/workspace/Pytorch_CV_Lab/app/labels.json') as file:
    labels = json.load(file)
    
with open('/workspace/Pytorch_CV_Lab/app/transforms.json') as file:
    augmentations = json.load(file)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def load_image(fname,mean=imagenet_mean,std=imagenet_std,size=(224,224),use_centercrop=False, transform=None):
    """Load image given path, required size, and transforms"""
    img = Image.open(fname).convert("RGB").resize(size)
    if transform is not None:
        img = transform(img)
    img = np.array(img,dtype=np.float32).transpose(2,1,0) / 255. #NHC --> C H W
    if use_centercrop:
        img = centercrop(img,(size[-2],size[-1]))
    return np.array((img - mean) / std,dtype=np.float32)


def top_k_predictions(file, model, transform):
    """
    Returns predictions obtained by running the specified 
    file using the specified model and transforms
    """

    if transform is not None:
        data = np.array([load_image(file, transform=transform)])
    else:
        data = np.array([load_image(file)])

    triton_http_client = http_client.InferenceServerClient(url=model_configuration["http_url"],
                                                       verbose=model_configuration["verbose"])
    input0 = http_client.InferInput(model_configuration[model]["input"],data.shape,
                               model_configuration["dtype"])
    input0.set_data_from_numpy(data,binary_data=True)

    output = http_client.InferRequestedOutput(model_configuration[model]["output"], binary_data=True)

    start_time = time.time()
    response = triton_http_client.infer(model_configuration[model]["name"], model_version=
                                        model_configuration["model_version"],inputs=[input0], outputs=[output])
    result = response.get_response()
    end_time = time.time()

    total_inference_time = end_time - start_time
    predictions = softmax(response.as_numpy(model_configuration[model]["output"]))[0]
    pred = pd.DataFrame(np.array(predictions), index=labels, columns=["Image"] )

    return pred, total_inference_time
        

def select_transformations():
    """Creates drop-down menu for model selection for inferencing"""
    return st.sidebar.selectbox(
            "Select a transformation:", sorted(list(augmentations.keys()))
        )

def get_transormations_params(transform_name):
    """Returns default augmentation parameters for specified transform"""
    param_values = augmentations[transform_name]
    return getattr(torchvision.transforms, transform_name)(**param_values)
