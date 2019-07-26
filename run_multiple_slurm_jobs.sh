#!/usr/bin/env bash
#
# for num_conv_spatial_layers in 1 3 5
# do
#     for num_conv_temporal_layers in 1 3 5
#     do
#       for conv_spatial_filter in "conv_spatial_filter_2_2" "conv_spatial_filter_3_3"
#       do
#           for conv_temporal_filter in "conv_temporal_filter_1_7" "conv_temporal_filter_2_3"
#             do
#               for num_spatial_filter in 100 200 300
#               do
#                 sbatch -n 1 --mem-per-cpu 16G -t 4:00:00 -p gpu --gres=gpu:1 run.sh model_name=$(hexdump -n 16 -e '4/4 "%08X" 1 "\n"' /dev/urandom).h5 n_process=1 batch_size=32 lr=0.00005 use_vp=False num_conv_spatial_layers=$num_conv_spatial_layers num_conv_temporal_layers=$num_conv_temporal_layers $conv_spatial_filter num_spatial_filter=$num_spatial_filter $conv_temporal_filter
#               done
#         done
#       done
#     done
# done
#
# #
# for num_conv_spatial_layers in 7
# do
#     for num_conv_temporal_layers in 7
#     do
#       for conv_spatial_filter in "conv_spatial_filter_2_2" "conv_spatial_filter_3_3"
#       do
#           for conv_temporal_filter in "conv_temporal_filter_1_7" "conv_temporal_filter_2_3"
#             do
#               for num_spatial_filter in 200
#               do
#                 sbatch -n 1 --mem-per-cpu 16G -t 4:00:00 -p gpu --gres=gpu:1 run.sh model_name=$(hexdump -n 16 -e '4/4 "%08X" 1 "\n"' /dev/urandom).h5  batch_size=32 n_process=1 lr=0.00005 use_vp=False num_conv_spatial_layers=$num_conv_spatial_layers patience=5 num_epochs=50 num_conv_temporal_layers=$num_conv_temporal_layers $conv_spatial_filter num_spatial_filter=$num_spatial_filter  $conv_temporal_filter
#               done
#         done
#       done
#     done
# done

# for num_conv_spatial_layers in 0 1
# do
#   for num_conv_temporal_layers in 0 1
#   do
#     for num_spatial_filter in 1 2 3 4 5 7 10
#     do
#       sbatch -n 1 --mem-per-cpu 16G -t 4:00:00 -p gpu --gres=gpu:1 run.sh standardized_combined_simple_ensemble run_on_training_loss batch_size=32 n_process=1 lr=0.0002 use_vp=False num_conv_spatial_layers=$num_conv_spatial_layers patience=5 num_conv_temporal_layers=$num_conv_temporal_layers $conv_spatial_filter  num_temporal_filter=1 num_temporal_filter=1 num_epochs=50 num_spatial_filter=$num_spatial_filter
#     done
#   done
# done


# for num_conv_spatial_layers in 0 1
# do
#   for num_conv_temporal_layers in 0 1
#   do
#     for num_spatial_filter in 1 2 3 4 5 7 10
#     do
#       for num_steps_per_epoch in 1 3 5 10 20 30 50 100
#       do
#       sbatch -n 1 --mem-per-cpu 16G -t 1:00:00 -p gpu --gres=gpu:1 run.sh steps_per_epoch=$num_steps_per_epoch standardized_combined_simple_ensemble batch_size=32 n_process=1 lr=0.0002 use_vp=False num_conv_spatial_layers=$num_conv_spatial_layers num_conv_temporal_layers=$num_conv_temporal_layers $conv_spatial_filter num_temporal_filter=1 num_spatial_filter=$num_spatial_filter
#     done
#   done
#   done
# done

# for num_conv_spatial_layers in 1 2 3
# do
#   for num_conv_temporal_layers in 1
#   do
#     for num_spatial_filter in 20 40 60 100
#     do
#       for num_steps_per_epoch in 30 50 100
#       do
#         for dropout in 0.25 0.4 0.5 0.6
#         do
#       sbatch -n 1 --mem-per-cpu 32G -t 6:00:00 -p gpu --gres=gpu:1 run.sh stop_on_training_loss steps_per_epoch=$num_steps_per_epoch standardized_combined_simple_ensemble n_process=1 lr=0.002 use_vp=False num_conv_spatial_layers=$num_conv_spatial_layers num_conv_temporal_layers=$num_conv_temporal_layers num_temporal_filter=1 num_epochs=1000 shuffle_generator=False num_spatial_filter=$num_spatial_filter dropout=$dropout
#     done
#     done
#   done
#   done
# done


# for num_conv_spatial_layers in 1 2 3
# do
#   for num_conv_temporal_layers in 1
#   do
#     for num_spatial_filter in 20 40 60 100
#     do
#       for num_steps_per_epoch in 30 50 100
#       do
#         for dropout in 0.25 0.4 0.5 0.6
#         do
#       sbatch -n 1 --mem-per-cpu 32G -t 6:00:00 -p gpu --gres=gpu:1 run.sh patience=50 stop_on_training_loss steps_per_epoch=$num_steps_per_epoch standardized_combined_simple_ensemble n_process=1 lr=0.002 use_vp=False num_conv_spatial_layers=$num_conv_spatial_layers num_conv_temporal_layers=$num_conv_temporal_layers num_temporal_filter=1 num_epochs=1000 shuffle_generator=False num_spatial_filter=$num_spatial_filter dropout=$dropout
#     done
#     done
#   done
#   done
# done

for num_conv_spatial_layers in 3 4 5
do
  for num_conv_temporal_layers in 4 5
  do
    for num_spatial_filter in 60 100
    do
      for num_steps_per_epoch in 50 100 200
      do
        for conv_temporal_filter in "conv_temporal_filter_1_7" "conv_temporal_filter_2_3" "conv_temporal_filter_1_3"
          do
        for dropout in  0.5 0.75
        do
      sbatch -n 1 --mem-per-cpu 24G -t 6:00:00 -p gpu --gres=gpu:1 run.sh  conv_temporal_filter steps_per_epoch=$num_steps_per_epoch standardized_combined_simple_ensemble n_process=1 lr=0.002 use_vp=False num_conv_spatial_layers=$num_conv_spatial_layers num_conv_temporal_layers=$num_conv_temporal_layers num_temporal_filter=1 num_epochs=1000 $conv_temporal_filter num_spatial_filter=$num_spatial_filter use_early_stopping=False dropout=$dropout
    done
  done
    done
  done
  done
done
