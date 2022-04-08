for mode in pytorch onnx tensorrt32 tensorrt16 tensorrt8
do
    source ./launch-job.sh < <(echo "$mode
    tuberculosis
    resnet18
    infer18-$mode
    ")

    source ./launch-job.sh < <(echo "$mode
    tuberculosis
    resnet50
    infer50-$mode
    ")
done