sbatch -n 1 --mem-per-cpu 32G -t 24:00:00 -p gpu --gres=gpu:3 \
./run.sh predictSeizureConvExp.py with batch_size=64 \
standardized_ensemble \
n_process=1 \
lr=0.002 \
num_temporal_filter=1 \
num_spatial_filter=40 \
use_early_stopping=True \
num_conv_spatial_layers=5 \
patience=75 \
dropout=0.7 \
num_epochs=45 \
use_inception_like=True \
num_gpus=4 \
use_vp=False \
use_self_train=True
