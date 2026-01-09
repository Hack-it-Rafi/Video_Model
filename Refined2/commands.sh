python preprocess.py --dataset_root /path/to/dataset_root --processed_root /path/to/processed_root --device cuda
python train.py --processed_root /path/to/processed_root --epochs 8 --batch_size 4 --device cuda
python infer.py --recording /path/to/dataset_root/recording_001 --processed_root /path/to/processed_root --device cuda

