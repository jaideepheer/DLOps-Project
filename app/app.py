import streamlit as st
import os
from client import top_k_predictions, load_image, select_transformations, get_transormations_params
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64


def model_inference():
    """
    Page for model inference. Has functionality to upload an image and run inference on 
    multiple models, with the facility to apply any data augmentation of choice
    """
    
    file_up = st.file_uploader("Upload an image", type="jpg")

    model = st.sidebar.selectbox("Choose Model for Inferencing from Triton Inference Server ", 
                                    ['resnet50_torch', 'resnet50_onnx', 'resnet50_trt_fp32', 'resnet50_trt_fp16', 'resnet50_trt_int8'])

    is_augment = st.sidebar.radio("Do you want to apply data augmentations?", ["Yes", "No"])

    if file_up is not None:
        image = Image.open(file_up).convert("RGB")
        st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    if is_augment == "Yes": 
        if file_up is not None:
            transform_name = select_transformations()
            transform = get_transormations_params(transform_name)
            if st.sidebar.checkbox("Show Augmentations"):
                image = Image.open(file_up).convert("RGB")
                transformed_img = transform(image)
                st.image(transformed_img, caption='Augmented Image.', use_column_width=True)
                
    is_run = st.sidebar.button('Run Inference')
                
    if is_run:
        if file_up is not None:
            if is_augment == "No":
                labels, total_inference_time = top_k_predictions(file_up, model, None)
            else:
                labels, total_inference_time = top_k_predictions(file_up, model, transform)
            st.write("Total Inference Time is {:.5f} seconds".format(total_inference_time))
            st.write("Predicted Class and Scores")
            st.bar_chart(labels)
        else:
            st.error("First upload an image")
            

def model_performance():
    """
    Page for model performance. Functionality to choose model, batch size, concurrency
    range, measurement interval as well as protocol
    """

    latency_comps = ["Client Send", "Network+Server Send/Recv", "Server Queue", "Server Compute Input", "Server Compute Infer", "Server Compute Output", "Client Recv"]
    latency_p = ["p50 latency", "p90 latency", "p95 latency", "p99 latency"]
        
    model = st.sidebar.selectbox("Choose Model for evaluating Model Performance", 
                                 ['resnet50_torch', 'resnet50_onnx', 'resnet50_trt_fp32', 'resnet50_trt_fp16', 'resnet50_trt_int8'])
    
    batch_size = st.sidebar.slider("Choose Batch Size", min_value=1, max_value=128, step=1)
    
    conc_start = st.sidebar.number_input("Choose concurrency range start point", min_value=1, step=1)
    conc_end = st.sidebar.number_input("Choose concurrency range end point", min_value=conc_start, step=1)
    conc_step = st.sidebar.number_input("Choose concurrency range step", min_value=1, step=1)
    
    protocol = st.sidebar.radio("Choose the protocal", ["HTTP", "gRPC"])
    
    measurement_int = st.sidebar.number_input("Choose measurement interval (in msec.)", value=5000, step=1)
    
    url = "172.25.0.42:30369"
    if protocol == "gRPC":
        url = "172.25.0.42:30522"
    
    if model == "resnet50_torch" :
        cmd = f'perf_analyzer -u {url} -m {model} --concurrency-range {str(conc_start)}:{str(conc_end)}:{str(conc_step)} -i {protocol} -p {str(measurement_int)} --shape input__0:{batch_size},3,224,224 -f test.csv'
    elif model == "resnet50_onnx" :
        cmd = f'perf_analyzer -u {url} -m {model} --concurrency-range {str(conc_start)}:{str(conc_end)}:{str(conc_step)} -i {protocol} -p {str(measurement_int)} --shape input:{batch_size},3,224,224 -f test.csv'
    else:
        cmd = f'perf_analyzer -u {url} -m {model} -b {batch_size} --concurrency-range {str(conc_start)}:{str(conc_end)}:{str(conc_step)} -i {protocol} -p {str(measurement_int)} -f test.csv'

    st.markdown(f"Command used: `{cmd}`")
    
    if st.sidebar.button("Evaluate Model Performance"):
        
        gif_runner = st.image("loading.gif")
        stream = os.popen(cmd)
        out = stream.read()
        gif_runner.empty()
        
        data = pd.read_csv('test.csv', index_col=0)
        data = data.sort_index()
        
        st.markdown("## Model Performance")
        st.write(data)

        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode() 
        linko= f'<a href="data:file/csv;base64,{b64}" download="perf_analysis.csv">Download data</a>'
        st.markdown(linko, unsafe_allow_html=True)
        
        st.markdown("## Components of Latency")
        st.bar_chart(data[latency_comps])
        
        st.markdown("## Latency vs. Throughput")
        data['average'] = data[latency_p].mean(axis=1)
        throughput = pd.DataFrame({
            "Average Latency" : np.array(data["average"])
        }, index=np.array(data["Inferences/Second"]))
        st.line_chart(throughput)
        

if __name__ == '__main__':

    st.title("NVIDIA Image Classification Application")
    
    page = st.sidebar.selectbox("Choose your page", ["Model Inference", "Model Performance"]) 
    
    if page == "Model Inference":
        model_inference()
    elif page == "Model Performance":
        model_performance()
