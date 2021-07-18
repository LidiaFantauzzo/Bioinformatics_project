#!/usr/bin/env bash

path=$(pwd)

num_epochs=700
lr=0.05
model="bisenetv2"
batch_size=16
name="brats20"
dropout="True"

###################### Functions ###################################
function run_fedavg() {
	pushd "${path}"/src/ > /dev/null 2>&1 || exit
	python3 main.py \
    --model ${model} \
    --num-epochs "${num_epochs}" \
    --lr ${lr} \
    --batch-size ${batch_size}\
	--dropout ${dropout}
	popd > /dev/null 2>&1 || exit
}

##################### Script #################################
pushd ../ > /dev/null 2>&1 # hide output: > /dev/null 2>&1

# Run FedAvg experiments
echo "Start..."
run_fedavg 

popd ../ > /dev/null 2>&1
