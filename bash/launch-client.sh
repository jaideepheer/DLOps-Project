#!/bin/bash

# For the DGX-2 server, we have CUDA 11.2.1
# See: https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
image="nvcr.io/nvidia/pytorch:21.03-py3"

rootpath="$(realpath $(dirname $(realpath "${BASH_SOURCE[-1]}"))/../)"
# DGX-2 server specific, comment this out
rootpath="/DATA1/$username/dlops_project"

host_ip="$(hostname -I | cut -f1 -d' ')"
# host_ip="10.100.79.117"

echo "Enter Pod Name"

read deployment

echo "Enter triton http port"

read httpport
# httpport=30769

echo "Enter triton grpc port"

read grpcport
# grpcport=30970

username=$(whoami)
rootdir='/workspace'

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
        env:
        - name: HOME
          value: $HOME
        ports:
        - containerPort: 8501
        command: [ "python3" ]
        args: [ "${rootdir}/app/setup.py", "run", "${rootdir}/app/app.py" ] 
        env:
          - name: TRITON_HTTP_URL
            value: ${host_ip}:${httpport}
          - name: TRITON_GRPC_URL
            value: ${host_ip}:${grpcport}
          - name: PYTHONPATH
            value: "${rootdir}"
        resources:
          limits:
            nvidia.com/gpu: 0
        volumeMounts:
        - name: raid
          mountPath: "${rootdir}"
      volumes:
      - name: raid
        hostPath:
          path: $rootpath
          type: Directory
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
  - protocol: TCP
    nodePort:
    port: 8501
    targetPort: 8501
EOF


kubectl create -f ${username}-svc.yaml
rm ${username}-dep.yaml
rm ${username}-svc.yaml
sleep 10

podname=$( kubectl get pod |grep -w "$deployment"|awk '{print $1}')


jup_token="$(kubectl logs $podname |grep token|awk '{print $4}'|awk '{print substr($0,29,77)}')"


svc_nodeport="$(kubectl describe svc $deployment |awk '{if ($1=="NodePort:") print $3}'|awk '{print substr($0,1,5)}')"

host_ip_link="$(echo "http://$host_ip:$svc_nodeport")"

echo "Welcome to NVIDIA DGX A100 Cluster. Kindly paste the given URL onto your browser $host_ip_link"

echo $host_ip_link > client-login-$deployment.txt

popd