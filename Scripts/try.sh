# try train.py in Colab to see results
set -ex
if [ -z "$1" ]; then name="try"; else name=$1; fi;
python ./train.py                                                                   \
    --log_name ${name}                                                              \
    --delta 1 --lamda 1 --gamma 1                                                   \
    --epochs 200 --batch_size 16                                                    \
    --test_start 20 --test_period 20                                                \
