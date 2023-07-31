#!/bin/bash

python train_ppi.py --cuda_id 2 --model gcn --criterion sigmoid --hidden_dim 16 --log_dir ./log/gcn_ppi_16_3_1_0_0_30_feature_base --k 3 --eta 1 --val_test_batch_size 2 --epochs 1 --run_times 5 --normalize True --gnn_lr 0.01 --vae_lr 0.00001 --sim_function feature_base &
python train_ppi.py --cuda_id 1 --model gcn --criterion sigmoid --hidden_dim 16 --log_dir ./log/gcn_ppi_16_3_0.9_0_0_30_feature_base --k 3 --eta 0.9 --val_test_batch_size 2 --epochs 30 --run_times 5 --normalize True --gnn_lr 0.01 --vae_lr 0.00001 --sim_function feature_base &
python train_ppi.py --cuda_id 2 --model gcn --criterion sigmoid --hidden_dim 16 --log_dir ./log/gcn_ppi_16_3_0.7_0_0_30_feature_base --k 3 --eta 0.7 --val_test_batch_size 2 --epochs 30 --run_times 5 --normalize True --gnn_lr 0.01 --vae_lr 0.00001 --sim_function feature_base &
python train_ppi.py --cuda_id 3 --model gcn --criterion sigmoid --hidden_dim 16 --log_dir ./log/gcn_ppi_16_3_0.5_0_0_30_feature_base --k 3 --eta 0.5 --val_test_batch_size 2 --epochs 30 --run_times 5 --normalize True --gnn_lr 0.01 --vae_lr 0.00001 --sim_function feature_base &
python train_ppi.py --cuda_id 4 --model gcn --criterion sigmoid --hidden_dim 16 --log_dir ./log/gcn_ppi_16_3_0.2_0_0_30_feature_base --k 3 --eta 0.2 --val_test_batch_size 2 --epochs 30 --run_times 5 --normalize True --gnn_lr 0.01 --vae_lr 0.00001 --sim_function feature_base &
python train_ppi.py --cuda_id 5 --model gcn --criterion sigmoid --hidden_dim 16 --log_dir ./log/gcn_ppi_16_3_0.1_0_0_30_feature_base --k 3 --eta 0.1 --val_test_batch_size 2 --epochs 30 --run_times 5 --normalize True --gnn_lr 0.01 --vae_lr 0.00001 --sim_function feature_base &
python train_ppi.py --cuda_id 6 --model gcn --criterion sigmoid --hidden_dim 16 --log_dir ./log/gcn_ppi_16_3_0_0_0_30_feature_base --k 3 --eta 0 --val_test_batch_size 2 --epochs 30 --run_times 5 --normalize True --gnn_lr 0.01 --vae_lr 0.00001 --sim_function feature_base &
