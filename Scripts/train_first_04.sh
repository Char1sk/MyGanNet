# param, long epoch
set -ex
if [ -z "$1" ]; then name="train_first_04"; else name=$1; fi;
python /kaggle/input/my-net/MyNet/train.py                                          \
    --log_name ${name}                                                              \
    --delta 1 --lamda 10 --gamma 5                                                  \
    --epochs 700 --batch_size 12                                                    \
    --test_start 100 --test_period 100                                              \
    --vgg_model /kaggle/input/modelscagan/vgg.model                                 \
    --inception_model /kaggle/input/modelscagan/pt_inception.pth                    \
    --data_folder /kaggle/input/my-cufs-new/My-CUFS-New                             \
