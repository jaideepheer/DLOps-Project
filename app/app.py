from src.conf import args
import streamlit as st
import os
from PIL import Image
import pandas as pd
import numpy as np
import base64
from itertools import product
import time
import json

import tritonclient.http as http_client
import tritonclient.grpc as grpc_client

model_names = [
    f"{model}_{mode}"
    for model, mode in product(
        ["resnet50"], ["torch", "onnx", "trt_fp32", "trt_fp16", "trt_int8"]
    )
]

model_configuration = {
    "http_url": os.environ["TRITON_HTTP_URL"],
    "grpc_url": os.environ["TRITON_GRPC_URL"],
    "model_version": "1",
    "verbose": False,
    "dtype": "FP32",
}

print(model_names)


imagenet_mean = np.array([[[0.485]], [[0.456]], [[0.406]]])
imagenet_std = np.array([[[0.229]], [[0.224]], [[0.225]]])


with open("/workspace/app/labels.json") as file:
    labels = json.load(file)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def load_image(img_path: str):
    """
    Loads an encoded image as an array of bytes.

    This is a typical approach you'd like to use in DALI backend.
    DALI performs image decoding, therefore this way the processing
    can be fully offloaded to the GPU.
    """
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32).transpose(
        2, 1, 0
    )
    return np.array((img - imagenet_mean) / imagenet_std, dtype=np.float32)


def top_k_predictions(data, model, transform):
    """
    Returns predictions obtained by running the specified
    file using the specified model and transforms
    """
    data = np.array([load_image(data)])
    triton_http_client = http_client.InferenceServerClient(
        url=model_configuration["http_url"], verbose=model_configuration["verbose"]
    )
    input0 = http_client.InferInput(
        "input__0", data.shape, model_configuration["dtype"]
    )
    input0.set_data_from_numpy(data)

    output = http_client.InferRequestedOutput("output__0")

    start_time = time.time()
    response = triton_http_client.infer(
        model,
        model_version="1",
        inputs=[input0],
        outputs=[output],
    )
    # result = response.get_response()
    end_time = time.time()

    total_inference_time = end_time - start_time
    resp = response.as_numpy("output__0")[0]
    predictions = softmax(resp)
    pred = pd.DataFrame(np.array(predictions), index=labels, columns=["Image"])

    return pred, total_inference_time


def model_inference():
    """
    Page for model inference. Has functionality to upload an image and run inference on
    multiple models, with the facility to apply any data augmentation of choice
    """

    file_up = st.file_uploader("Upload an image", type="bmp")

    st.write(f"Connection URL: {model_configuration['http_url']}")
    print(f"Connection URL: {model_configuration['http_url']}")

    model = st.sidebar.selectbox(
        "Choose Model for Inferencing from Triton Inference Server ",
        model_names,
    )

    if file_up is not None:
        image = Image.open(file_up).convert("RGB")
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    is_run = st.sidebar.button("Run Inference")

    if is_run:
        if file_up is not None:
            labels, total_inference_time = top_k_predictions(file_up, model, None)
            st.write(
                "Total Inference Time is {:.5f} seconds".format(total_inference_time)
            )
            st.write("Predicted Class and Scores")
            st.bar_chart(labels)
        else:
            st.error("First upload an image")


def model_performance():
    """
    Page for model performance. Functionality to choose model, batch size, concurrency
    range, measurement interval as well as protocol
    """

    latency_comps = [
        "Client Send",
        "Network+Server Send/Recv",
        "Server Queue",
        "Server Compute Input",
        "Server Compute Infer",
        "Server Compute Output",
        "Client Recv",
    ]
    latency_p = ["p50 latency", "p90 latency", "p95 latency", "p99 latency"]

    model = st.sidebar.selectbox(
        "Choose Model for evaluating Model Performance", model_names
    )

    batch_size = st.sidebar.slider(
        "Choose Batch Size", min_value=1, max_value=128, step=1
    )

    conc_start = st.sidebar.number_input(
        "Choose concurrency range start point", min_value=1, step=1
    )
    conc_end = st.sidebar.number_input(
        "Choose concurrency range end point", min_value=conc_start, step=1
    )
    conc_step = st.sidebar.number_input(
        "Choose concurrency range step", min_value=1, step=1
    )

    protocol = st.sidebar.radio("Choose the protocal", ["HTTP", "gRPC"])

    measurement_int = st.sidebar.number_input(
        "Choose measurement interval (in msec.)", value=5000, step=1
    )

    url = (
        os.environ["TRITON_HTTP_URL"]
        if protocol == "gRPC"
        else os.environ["TRITON_GRPC_URL"]
    )

    st.write(f"Connection URL: {url}")
    print(f"Connection URL: {url}")

    # if model.endswith("torch") or model.endswith("onnx"):
    #     cmd = f"perf_analyzer -u {url} -m {model} --concurrency-range {str(conc_start)}:{str(conc_end)}:{str(conc_step)} -i {protocol} -p {str(measurement_int)} --shape INPUT:{batch_size},3,512,512 -f /workspace/app/test.csv"
    # else:
    cmd = f"perf_analyzer -u {url} -m {model} --concurrency-range {str(conc_start)}:{str(conc_end)}:{str(conc_step)} -i {protocol} -p {str(measurement_int)} -b {batch_size} -f /workspace/app/test.csv"

    st.markdown(f"Command used: `{cmd}`")

    if st.sidebar.button("Evaluate Model Performance"):

        gif_runner = st.image("/workspace/app/loading.gif")
        stream = os.popen(cmd)
        out = stream.read()
        print(out)
        gif_runner.empty()

        data = pd.read_csv("/workspace/app/test.csv", index_col=0)
        data = data.sort_index()

        st.markdown("## Model Performance")
        st.write(data)

        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        linko = f'<a href="data:file/csv;base64,{b64}" download="perf_analysis.csv">Download data</a>'
        st.markdown(linko, unsafe_allow_html=True)

        st.markdown("## Components of Latency")
        st.bar_chart(data[latency_comps])

        st.markdown("## Latency vs. Throughput")
        data["average"] = data[latency_p].mean(axis=1)
        throughput = pd.DataFrame(
            {"Average Latency": np.array(data["average"])},
            index=np.array(data["Inferences/Second"]),
        )
        st.line_chart(throughput)


if __name__ == "__main__":

    st.title("NVIDIA Image Classification for SOCOFing Data")

    page = st.sidebar.selectbox(
        "Choose your page", ["Model Inference", "Model Performance"]
    )

    if page == "Model Inference":
        model_inference()
    elif page == "Model Performance":
        model_performance()
