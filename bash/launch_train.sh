source ./launch-job.sh < <(echo "train
tuberculosis
resnet50
non50
")

source ./launch-job.sh < <(echo "train
tuberculosis_optimized
resnet50
opti50
")

source ./launch-job.sh < <(echo "train
tuberculosis
resnet18
non18
")

source ./launch-job < <(echo "train
tuberculosis_optimized
resnet18
opti18
")