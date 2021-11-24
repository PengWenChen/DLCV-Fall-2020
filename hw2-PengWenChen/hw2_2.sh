wget "https://www.dropbox.com/s/aumd5hn51ykyhoi/p2_1_model.pth?dl=1"
python3 hw2_2.py --test_img_dir=$1 --output_dir=$2 --best=$false --checkpoint_path 'p2_1_model.pth?dl=1'
