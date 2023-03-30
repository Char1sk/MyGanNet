# part 2
set -ex
if [ -z "$1" ]; then name="train_first_08"; else name=$1; fi;
python /kaggle/input/my-net/MyNet/train.py                                          \
    --log_name ${name}                                                              \
    --num_parts 2                                                                   \
    --delta 1 --lamda 10 --gamma 5                                                  \
    --epochs 400 --batch_size 12                                                    \
    --test_start 50 --test_period 50                                                \
    --vgg_model /kaggle/input/modelscagan/vgg.model                                 \
    --inception_model /kaggle/input/modelscagan/pt_inception.pth                    \
    --data_folder /kaggle/input/my-cufs-new/My-CUFS-New                             \
