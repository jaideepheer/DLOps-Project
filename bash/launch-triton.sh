#!/bin/bash

# For the DGX-2 server, we have CUDA 11.2.1
# See: https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
image="nvcr.io/nvidia/tritonserver:21.03-py3"

rootpath="$(realpath $(dirname $(realpath "${BASH_SOURCE[-1]}"))/../)"
# DGX-2 server specific, comment this out
rootpath="/DATA1/$username/dlops_project"

repopath="${rootpath}/triton_model_repository/SOCOFing/"

username=$(whoami)

deployment=$username

nodeport=30040

gpu=3
gpu_limit=0
rootdir='/models'

cat <<EOF >$HOME/${username}-dep.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $deployment
  labels:
    app: $deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: $deployment
  template:
    metadata:
      labels:
        app: $deployment
    spec:
      containers:
      - name: $deployment
        image: $image
        resources:
          limits:
            nvidia.com/gpu: $gpu_limit
        env:
        - name: CUDA_VISIBLE_DEVICES 
          value: "$gpu"
        ports:
        - containerPort: 8888
        command: ["/bin/bash"]
        args: ["-c","tritonserver --model-repository=${rootdir} --strict-model-config=true"]
        volumeMounts:
        - name: raid
          mountPath: ${rootdir}
      volumes:
      - name: raid
        hostPath:
          path: $repopath

EOF

pushd $HOME

kubectl create -f ${username}-dep.yaml


cat <<EOF >$HOME/${username}-svc.yaml

kind: Service
apiVersion: v1
metadata:
  name: $deployment
spec:
  type: NodePort
  selector:
    app: $deployment
  ports:
  - name: http-triton
    nodePort:
    port: 8000
    targetPort: 8000
  - name: grpc-triton
    nodePort:
    port: 8001
    targetPort: 8001
  - name: metrics-triton
    nodePort:
    port: 8002
    targetPort: 8002
EOF


kubectl create -f ${username}-svc.yaml
rm ${username}-dep.yaml
rm ${username}-svc.yaml
sleep 10
host_ip="$(hostname -I | cut -f1 -d' ')"

podname=$( kubectl get pod |grep -w "$deployment"|awk '{print $1}')

svc_nodeport="$(kubectl describe svc $deployment |awk '{if ($1=="NodePort:") print $3}'|awk '{print substr($0,1,5)}')"

host_ip_link="$(echo "$svc_nodeport")"

echo "Welcome to NVIDIA DGX A100 Cluster. Ports for HTTP, GRPC and Promethous are:"

echo "$host_ip_link"

echo $host_ip_link > triton-login-$deployment.txt

popd
