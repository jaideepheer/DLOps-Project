#!/bin/bash

username=$(whoami)

# For the DGX-2 server, we have CUDA 11.2.1
# See: https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
image="nvcr.io/nvidia/pytorch:21.03-py3"

rootpath="$(realpath $(dirname $(realpath "${BASH_SOURCE[-1]}"))/../)"

# DGX-2 server specific, comment this out
rootpath="/DATA1/$username/$(basename $rootpath)"

echo "Mounting code from: $rootpath"

echo "Enter Mode"

read mode

echo "Enter config name"

read config

echo "Echo enter model name"

read model

echo "Enter jobname"

read job

echo "Use wandb [True/False]?"

read wandb

gpu=3
gpu_limmit=0
rootdir='/workspace'

echo "Will use key $WANDB_API_KEY"

cat <<EOF >$HOME/${job}-dep.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: $job
spec:
  template:
    spec:
      containers:
      - name: $job
        image: $image
        # command: ["/bin/bash"]
        # args: ["-c", "while true; do date; sleep 500; done"]
        command: ["python3"]
        args: ["${rootdir}/src/main.py", "--mode", "${mode}", "-C", "${config}", "--model", "${model}", "--wandb", "${wandb}", "--rootdir", "${rootdir}"]
        env:
          - name: CUDA_VISIBLE_DEVICES 
            value: "$gpu"
          - name: WANDB_API_KEY
            value: "${WANDB_API_KEY}"
          - name: PYTHONPATH
            value: "${rootdir}"
        resources:
          limits:
            nvidia.com/gpu: $gpu_limmit
        volumeMounts:
        - name: raid
          mountPath: "${rootdir}"
        # Shared memory hack
        # https://stackoverflow.com/a/46434614/10027894
        - mountPath: /dev/shm
          name: dshm
      volumes:
      - name: raid
        hostPath:
          path: ${rootpath}
          type: Directory
      # Shared memory hack
      # https://stackoverflow.com/a/46434614/10027894
      # https://github.com/kubernetes/kubernetes/pull/63641
      - name: dshm
        emptyDir:
          sizeLimit: "350Mi"
          medium: Memory
      restartPolicy: Never
  backoffLimit: 1
EOF
pushd $HOME

kubectl create -f ${job}-dep.yaml
rm -rf ${job}-dep.yaml

popd
