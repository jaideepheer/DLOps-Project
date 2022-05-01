import streamlit as st
import os
from PIL import Image
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
from itertools import product


model_names = [
    f"ensemble_dali_{model}_{mode}"
    for model, mode in product(
        ["resnet18", "resnet50"], ["torch", "onnx", "trt_fp32", "trt_fp16", "trt_int8"]
    )
]

print(model_names)


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

    if model.endswith("torch") or model.endswith("onnx"):
        cmd = f"perf_analyzer -u {url} -m {model} --concurrency-range {str(conc_start)}:{str(conc_end)}:{str(conc_step)} -i {protocol} -p {str(measurement_int)} --shape input__0:{batch_size},3,512,512 -f /workspace/app/test.csv"
    else:
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

    st.title("NVIDIA Image Classification Application")
    model_performance()
