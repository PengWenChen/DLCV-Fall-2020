wget "https://www.dropbox.com/s/6hozia33mevdcsd/problem2_e40.pth?dl=1"
# TODO: create shell script for running your GAN model
python3 p2.py --test_generate_path=$1 --checkpoint_dir='./problem2_e40.pth?dl=1'
# Example
# python3 p2.py $1 
