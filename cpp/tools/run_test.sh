device=$1
k=$2
n=$3
d=$4
method=$5
./Nice_test --gtest_filter=KDACTest/0.${device}${k}_${n}_${d}_${method} 2>&1 | tee ${device}.out
