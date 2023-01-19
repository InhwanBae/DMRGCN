#!/bin/bash
echo "Start training task queues"

# Hyperparameters
name_array=("eth" "hotel" "univ" "zara1" "zara2")
prefix="social-dmrgcn-"
suffix="-experiment"
lr="0.0001"
lr_sh_rate=32
n_stgcn=1
n_tpcnn=4
num_epochs=128
num_batches=128

device_id_array=(0 1 2 3 4)
PID_array=()

if [ $# -eq 5 ]; then
 echo "CUDA_VISIBLE_DEVICES set to: $1 $2 $3 $4 $5"
 device_id_array=($1 $2 $3 $4 $5)
fi

# Signal handler
sighdl ()
{
  echo "Kill training processes"
  for (( i=0; i<${#name_array[@]}; i++ ))
  do
    kill ${PID_array[$i]}
  done
  echo "Done."
  exit 0
}

trap sighdl SIGINT SIGTERM

# Start training tasks
for (( i=0; i<${#name_array[@]}; i++ ))
do
  printf "Training ${name_array[$i]}"
  CUDA_VISIBLE_DEVICES=${device_id_array[$i]} python3 train_test.py \
  --dataset "${name_array[$i]}" --tag "${prefix}""${name_array[$i]}""${suffix}" \
  --n_stgcn $n_stgcn --n_tpcnn $n_tpcnn \
  --batch_size $num_batches --num_epochs $num_epochs \
  --lr "$lr" --use_lrschd --lr_sh_rate $lr_sh_rate &
  PID_array[$i]=$!
  printf " job ${#PID_array[@]} pid ${PID_array[$i]}\n"
done

for (( i=0; i<${#name_array[@]}; i++ ))
do
  wait ${PID_array[$i]}
done

# Start testing tasks
for (( i=0; i<${#name_array[@]}; i++ ))
do
  printf "Testing ${name_array[$i]}"
  CUDA_VISIBLE_DEVICES=${device_id_array[$i]} python3 test.py --tag "${prefix}""${name_array[$i]}""${suffix}" &
  PID_array[$i]=$!
  printf " job ${#PID_array[@]} pid ${PID_array[$i]}\n"
done

for (( i=0; i<${#name_array[@]}; i++ ))
do
  wait ${PID_array[$i]}
done

echo "Done."
