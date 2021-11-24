wget 'https://www.dropbox.com/s/jtd6qqcsykh4ubn/p1_model.pth?dl=1' 
python3 hw2_1.py --test_img_dir=$1 --output_dir=$2 --checkpoint_dir './p1_model.pth?dl=1'
