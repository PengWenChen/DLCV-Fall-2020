# DLCV Final Project ( Medical-Imaging )

# How to run your code?
The final result on the kaggle was produced by ensembling with 3 models.
### Download our models
The bash file downloads the 3 models: model1.pth?dl=1, model2.pth?dl=1, model3.pth?dl=1

    bash get_model.sh
### Inference our result on kaggle

    cd Single_Train_Multi_Test
    python main.py --mode test_multi_ensemble --output_csv_name {the output.csv, i.e. "test.csv"} --test_dir {The test image directory, i.e. "../Blood_data/test"}
If you want to ensemble with other models, which have the same architecture as ours, you can add the options below.

    --model1 {your model1}
    --model2 {your model2}
    --model3 {your model3}

Next, you need to change the csv form to kaggle form.

    cd ..
    python to_kaggle.py --pred_csv_path {i.e. "./Single_Train_Multi_Test/test.csv"} --out_csv_path OUT_CSV_PATH {the output csv, i.e. ./pred_kaggle.csv}
### To train our three models.
We first train the models with single images, and then fine tune with sequence images. (sequence length=5)
<br> If you want to split data for train and validataion, please set `--validation_split=True`

     cd Single_Train_Multi_Test
##### model1
The single image model we used: epoch 8

    python main.py --mode train_single --checkpoint_path {model saving directory, i.e. "./"} --model_name {give a name to the model, i.e. "model1"} --seed 123
The sequence images fine tuning model we used: iteration 800

    python main.py --mode finetune_with_multi --model_dir {the model you want to fine tune, i.e. "./model1.pth"} --seed 123 --checkpoint_path {model saving directory, i.e. "./"} --model_name {give a name to the model, i.e. "model1_finetune"}

##### model2
The single image model we used: epoch 11

    python main.py --mode train_single --checkpoint_path {model saving directory} --model_name {give a name to the model, i.e. model2} --seed 413
The sequence images fine tuning model we used: iteration 600    
    
    python main.py --mode finetune_with_multi --model_dir {the model you want to fine tune, i.e. "./model2.pth"} --seed 413 --checkpoint_path {model saving directory, i.e. "./"} --model_name {give a name to the model, i.e. "model2_finetune"}
##### model3
The single image model we used: epoch 9

    python main.py --mode train_single --checkpoint_path {model saving directory} --model_name {give a name to the model, i.e. model2} --seed 65
The sequence images fine tuning model we used: iteration 600       
    
    python main.py --mode finetune_with_multi --model_dir {the model you want to fine tune, i.e. "./model3.pth"} --seed 65 --checkpoint_path {model saving directory, i.e. "./"} --model_name {give a name to the model, i.e. "model3_finetune"}

### To see the score on validation data.
If you want to see the score on validation data, you might have splitted the data for train and validation. 
<br> Thus, you probably want `--validation_split` to be true.

     cd Single_Train_Multi_Test
single image model

    python main.py --mode train_single --model_dir {the model you want to see the validation score, i.e. "./model1.pth"} --seed {the seed above} --validation_only=True --validation_split=True
sequence image model    
    
    python main.py --mode finetune_with_multi --model_dir {the model you want to see the validation score, i.e. "./model1_finetune.pth"} --seed {the seed above} --validation_only=True --validation_split=True

### Inference without ensemble.
single image model

    python main.py --mode "test_single" --output_csv_name {the output.csv, i.e. "./test.csv"} --model_dir {model, i.e. "./model1.pth"} --test_dir {The test image directory, i.e. "../Blood_data/test"} 
sequence image model

    python main.py --mode "test_multi" --output_csv_name {the output.csv, i.e. "./test.csv"} --model_dir {model, i.e. "./model1_finetune.pth"} --test_dir {The test image directory, i.e. "../Blood_data/test"} 
# Usage
To start working on this final project, you should clone this repository into your local machine by using the following command:

    git clone https://github.com/DLCV-Fall-2020/medical-imaging-<team_name>.git
Note that you should replace `<team_name>` with your own team name.

For more details, please click [this link](https://docs.google.com/presentation/d/1T8Wh9rM5zCiuMVCulDCZwX9JZZ9Mqgd0Yr3uqgPpe1I/edit?usp=sharing) to view the slides of Final Project - Medical Imaging. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `Blood_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/file/d/1XY5twJuNLU-tJL-sr5-efTMPqbOtodS6/view?usp=sharing) and unzip the compressed file manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> 1. Please do not disclose the dataset! Also, do not upload your get_dataset.sh to your (public) Github.
> 2. You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `Blood_data` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

### Evaluation
We will use F2-score to evaluate your model. Please refer to the introduction ppt for more details.

# Submission Rules
### Deadline
2021/1/22 11:59 GMT+8

### Late Submission Policy
#### Late Submission is NOT allowed for final project!

### Academic Honesty
-   Taking any unfair advantages over other class members (or letting anyone do so) is strictly prohibited. Violating university policy would result in an **F** grade for this course (**NOT** negotiable).    
-   If you refer to some parts of the public code, you are required to specify the references in your report (e.g. URL to GitHub repositories).      
-   You are encouraged to discuss homework assignments with your fellow class members, but you must complete the assignment by yourself. TAs will compare the similarity of everyone’s submission. Any form of cheating or plagiarism will not be tolerated and will also result in an **F** grade for students with such misconduct.

> 🆕 ***NOTE***  
> For the sake of conformity, please use the `python3` command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.

### Packages
This homework should be done using python3.6. For a list of packages you are allowed to import in this assignment, please refer to `requirments.txt` for more details.

You can run the following command to install all the packages listed in `requirements.txt`:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.


# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under Final Project FAQ section in FB group
