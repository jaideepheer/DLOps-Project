cat <<EOF >$HOME/chown-dep.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: chown
spec:
  template:
    spec:
      containers:
      - name: chown
        image: dlops/labs:nvidia01
        command: ["/bin/chown","-R","$(id -u):$(id -g)", "/workspace/Assignment_1"]
        volumeMounts:
        - name: raid
          mountPath: /workspace
      volumes:
      - name: raid
        hostPath:
          path: /raid/dlops-course/$(whoami)
      restartPolicy: Never
  backoffLimit: 1
EOF
pushd $HOME

kubectl create -f chown-dep.yaml
rm -rf chown-dep.yaml

popd