# fixed loss pred_global! added test loss, do eval
set -ex
if [ -z "$1" ]; then name="train_first_02"; else name=$1; fi;
python /kaggle/input/my-net/MyNet/train.py                                          \
    --log_name ${name}                                                              \
    --pad                                                                           \
    --delta 1 --lamda 10 --gamma 10                                                 \
    --epochs 400 --batch_size 12                                                    \
    --test_start 50 --test_period 50                                                \
    --vgg_model /kaggle/input/modelscagan/vgg.model                                 \
    --inception_model /kaggle/input/modelscagan/pt_inception.pth                    \
    --data_folder /kaggle/input/my-cufs-new/My-CUFS-New                             \
