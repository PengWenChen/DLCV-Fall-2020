if [ "$2" = "mnistm" ]
then
    echo "Downloading mnistm model..."
    wget "https://www.dropbox.com/s/cfonimeriqmb2un/p3_mnistm_best.pth?dl=1"
elif [ "$2" = "svhn" ]
then
    echo "Downloading svhn model..."
    wget "https://www.dropbox.com/s/zq19f5vur21fsf0/p3_svhn_best.pth?dl=1"
elif [ "$2" = "usps" ]
then
    echo "Downloading usps model..."
    wget "https://www.dropbox.com/s/7xjzbzbuzzepy80/p3_usps_best.pth?dl=1"
else
    echo "Downloading model..."
    wget "https://www.dropbox.com/s/cfonimeriqmb2un/p3_mnistm_best.pth?dl=1"
    wget "https://www.dropbox.com/s/zq19f5vur21fsf0/p3_svhn_best.pth?dl=1"
    wget "https://www.dropbox.com/s/7xjzbzbuzzepy80/p3_usps_best.pth?dl=1"
fi
# TODO: create shell script for running your DANN model
python3 p3.py --target_test_dir=$1 --target_name=$2 --csv_output_dir=$3 --mode='test'
# Example
# python3 p3.py $1 $2 $3