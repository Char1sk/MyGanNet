# debug train.py in Local or Colab
set -ex
if [ -z "$1" ]; then name="exp_debug"; else name=$1; fi;
python ./train.py                                                                   \
    --log_name ${name}                                                              \
    --delta 1 --lamda 1 --gamma 1                                                   \
    --epochs 10 --batch_size 2                                                      \
    --test_start 5 --test_period 5                                                  \
