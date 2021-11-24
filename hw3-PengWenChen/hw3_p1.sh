wget "https://www.dropbox.com/s/kd0vrho96dxsm6w/problem1_model_v2_e90.pth?dl=1"
# TODO: create shell script for running your VAE model
python3 p1.py --test_generate_path=$1 --checkpoint_dir='./problem1_model_v2_e90.pth?dl=1'

# Example
# python3 p1.py $1 
