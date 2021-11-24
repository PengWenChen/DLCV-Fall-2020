if [ "$2" = "mnistm" ]
then
    echo "Downloading mnistm model..."
    wget "https://www.dropbox.com/s/mofmix1ilofofsw/p4_mnistm_best.pth?dl=1"
elif [ "$2" = "svhn" ]
then
    echo "Downloading svhn model..."
    wget "https://www.dropbox.com/s/ee7ltdo7svfx2el/p4_svhn_best.pth?dl=1"
elif [ "$2" = "usps" ]
then
    echo "Downloading usps model..."
    wget "https://www.dropbox.com/s/vc3ojnxr8ne78jv/p4_usps_best.pth?dl=1"
else
    echo "Downloading model..."
    wget "https://www.dropbox.com/s/mofmix1ilofofsw/p4_mnistm_best.pth?dl=1"
    wget "https://www.dropbox.com/s/ee7ltdo7svfx2el/p4_svhn_best.pth?dl=1"
    wget "https://www.dropbox.com/s/vc3ojnxr8ne78jv/p4_usps_best.pth?dl=1"
fi
# TODO: create shell script for running your improved UDA model
python3 p4.py --target_test_dir=$1 --target_name=$2 --csv_output_dir=$3 --mode='test'
# Example
# python3 p4.py $1 $2 $3