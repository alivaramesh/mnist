Instructions to use to the repo:

Setup:

- The Anaconda environment that I  have used for all the experiments is provided at "environment.yml".
If you are using Anaconda, run the following command to set up the environment:
(keep in mind that this environment is not minimalistic and may contain unnecessary packages)
$ conda env create -f environment.yml
Then activate the environment:
$ source activate mnist

- Move the 5 test variants (t1,t2,t3,t4, and clean) to the following relatiev address: "data/test_sets"

Training:

- For any new training, a directory will be created in "exps" directory, referred to as "save-dir". Name of the directory will reflect the training configuration

- Once training is finished, the model will be saved in save-dir in a file named "mnist.pth"

- The training log will be printed to the stdout and also will be written to a file called "log" in save-dir

- The results will be plotted and saved to a file named "results.jpg" in the save-dir

- Example command to train a model (for more information about the oprtions please refer to the main.py):
$ cd /path/to/main/repo/dir
$ python main.py --epochs 10  --batch-size 64 --lr 1 --gamma .7 --drop_out none --erase_b .6 --erase_w 1.0

- To reproduce the experiments with desired arguments, you may use the bash script "exps_train.sh" in the main directory

- All the models that I have trained can be found in the exps directory

- I have run all the experiments on GPU, but for CPU experiments you can use argument 'no-cuda'

Testing 
- To run tests on an existing model, pass argument --test and then provide path to the model using argument 'model-path.'\
This will run the model on the 5 variants and store the predictions in a directory with path <model_path>_test_<time_stamp>.\ 
For each variant, there will be a npy file with an array where for each sample the first number is the true class and\
the second number is the prediction. 

Please note that vis.py is used for internal visualizations and analysis. It contains useful functionalities for analysis but is not necessary for training and testing. 