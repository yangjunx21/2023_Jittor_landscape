import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)

args = parser.parse_args()

print('训练生成数据存储在./checkpoints/bs4vae中')
subprocess.call(f'CUDA_VISIBLE_DEVICES="0,1,4,5" python spade_train.py --name bs4vae --dataset_mode custom --gpu_ids 0,1,4,5 --label_dir {args.input_path}/labels --image_dir {args.input_path}/imgs --label_nc 29 --no_instance --use_vae --batchSize 1', shell=True)
