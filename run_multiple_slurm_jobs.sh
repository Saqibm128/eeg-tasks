#!/usr/bin/env bash

for num_conv_spatial_layers in 3 4 5
do
    for num_conv_temporal_layers in 3 4 5
    do
      for conv_spatial_filter in "conv_spatial_filter_2_2" "conv_spatial_filter_3_3"
      do
          for conv_temporal_filter in "conv_temporal_filter_1_7" "conv_temporal_filter_2_3"
            do
              for num_spatial_filter in 50 100 200 400
              do
                echo sbatch -n 6 --mem-per-cpu 6G -t 1:00:00 -p gpu --gres=gpu:1 run.sh model_name=$(hexdump -n 16 -e '4/4 "%08X" 1 "\n"' /dev/urandom).h5 n_process=6 use_vp=False num_conv_spatial_layers=$num_conv_spatial_layers num_conv_temporal_layers=$num_conv_temporal_layers $conv_spatial_filter num_spatial_filter=$num_spatial_filter $conv_temporal_filter
              done
        done
      done
    done
done
