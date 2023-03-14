# try train.py in Colab to see results
set -ex
if [ -z "$1" ]; then name="train_first"; else name=$1; fi;
python /kaggle/input/mygannet/MyNet/train.py                                        \
    --log_name ${name}                                                              \
    --delta 1 --lamda 1 --gamma 1                                                   \
    --epochs 200 --batch_size 16                                                    \
    --test_start 40 --test_period 40                                                \
    --vgg_model /kaggle/input/modelscagan/vgg.model                                 \
    --inception_model /kaggle/input/modelscagan/pt_inception.pth                    \
    --data_folder /kaggle/input/my-cufs-new/My-CUFS-New                             \
