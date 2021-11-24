wget "https://www.dropbox.com/s/q37xp7rrq97jbx3/p2_2_model.pth?dl=1"
python3 hw2_2.py --test_img_dir=$1 --output_dir=$2 --best 'best' --checkpoint_path './p2_2_model.pth?dl=1'
