#!/bin/bash

username=$(whoami)

# echo "Enter Container Image"

# read image

image="dlops/labs:nvidia01"

# echo "Is train/convert/eval task [y/n]?"

# read yn
# if [ "$yn" = "y" ]; then
#   filepath="/workspace/Assignment_1/scripts/src/main.py"
# else
#   echo "wht??"
#   return
# fi

# echo "Enter Filepath"
# read filepath

filepath="/workspace/Assignment_1/src/main.py"

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

gpu=1

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
        command: ["python3"]
        args: ["${filepath}", "--mode", "${mode}", "-C", "${config}", "--model", "${model}", "--wandb", "${wandb}"]
        env:
          - name: WANDB_API_KEY
            value: $WANDB_API_KEY
          - name: PYTHONPATH
            value: /workspace/Assignment_1
        resources:
          limits:
            nvidia.com/gpu: $gpu
        volumeMounts:
        - name: raid
          mountPath: /workspace
        # Shared memory hack
        # https://stackoverflow.com/a/46434614/10027894
        - mountPath: /dev/shm
          name: dshm
      volumes:
      - name: raid
        hostPath:
          path: /raid/dlops-course/${username}
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
