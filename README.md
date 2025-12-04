# GeCo: Towards Effective GAN-based Fashion Compatibility Modeling and Retrieval

This repository contains the official codebase to reproduce the experiments of the article "_GeCo: Towards Effective GAN-based Fashion Compatibility Modeling and Retrieval_". The codebase was developed and tested on Ubuntu 22.04 LTS; however, it can be executed on other operating systems with the necessary adjustments to environment variables, activation of Python environments, or configuration of additional utilities (e.g., unzip).


## Usage

### Prerequisites

The experiments can be run on both CPU and GPU, however, we highly recommend using a GPU for optimal performance. To use a GPU, ensure that all necessary NVIDIA drivers and CUDA toolkits are installed. A possible working environment configuration includes:

```
Nvidia drivers: 535.161.07
Cuda: 11.8
```
#### Using Conda

We suggest creating virtual environments with all the required libraries using Conda. For the installation of Conda, please refer to [the official documentation](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).

To create the environment run the following command:

```sh
conda env create -f environment.yml
```

Once the installation finishes, you can activate the environment by running:

```sh
conda activate geco_env
```

#### Using Venv

```sh
python -m venv geco_env
source geco_env/bin/activate
pip install -r requirements.txt
```

#### Using the CPU

Running the experiments on a CPU is not recommended; however, we also provide a requirements file for this scenario.

```sh
python -m venv geco_env
source geco_env/bin/activate
pip install -r requirements_cpu.txt
```

### Datasets

To run the experiments, it is mandatory to download the datasets and place them in the appropriate folders. All datasets must be stored in the `datasets` folder. Each dataset, identified by a folder with its name, contains two subfolders: `files` and `img`. The `files` subfolder contains the necessary CSV files, which are already provided, while the `img` subfolder is where the images must be downloaded.

For the FashionVC and ExpReduced datasets, the images can be downloaded as follows:

```sh
git clone https://bitbucket.org/Jay_Ren/fashion_recommendation_tkde2018_code_dataset.git
```

Once the download is complete, run the following commands to unzip the images into the corresponding dataset folders:

```sh
unzip ./fashion_recommendation_tkde2018_code_dataset/img.zip -d ./datasets/ExpReduced
unzip ./fashion_recommendation_tkde2018_code_dataset/FashionVC/img.zip -d ./datasets/FashionVC
```

To download the images for the FashionTaobaoTB dataset, make sure Node.js is installed on your machine. Then, proceed with the following steps:

```sh
cd ./datasets/FashionTaobao-TB
npm init -y
npm install superagent csv-parser cli-progress
npm install -g typescript
npm install -g tsc
tsc index.ts
node index.js
```

After the download finishes, you need to preprocess the images. Please, run the following command to resize the images:
```sh
python3 resize_imgs.py
```

### Running the experiments

As explained in the paper, to replicate the results of the proposed two-stage model, you should first train the _custom GAN_ model and then the _GeCo_ model. Ensure that you are always in the root directory of the project before running the experiments. Then run:

```sh
export PYTHONPATH=.
```

You can train the _GAN module_ by running the following command:
```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 custom_GAN/train_gan.py --dataset dataset_name
```

You can also specify other training arguments, such as the number of epochs, the learning rate, and the number of workers. However, we have set all these by default according to our setup.

Once the GAN model is ready, you can train the GeCo model via the following command:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 GeCo/train_geco.py --dataset dataset_name --alpha_values 0.25 --beta_values 0.75 --gamma_values 0.01 --tau_values 0.1
```

As with the training of the _custom GAN module_, you can specify additional training arguments such as "dataset" and "num_workers". Additionally, you must specify some hyperparameters of the model: "alpha_values", "beta_values", "gamma_values", and "tau_values". For these arguments, the script accepts both lists of elements or single scalars, allowing for hyperparameter exploration. Once training finishes, the performance of the model on the test set is saved in a CSV file, which by default is saved in the _GeCo_ directory with the name "out.csv".

### Training the baselines

Additionally, we provide the code for reimplementing all the baselines in the corresponding directory. The structure for the training scripts is exactly the same as the training file for the _GeCo_ model. You can train the three baselines by running:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 baselines/BPRDAE/train_bprdae.py --dataset dataset_name
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 baselines/MGCM/train_mgcm.py --dataset dataset_name 
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 baselines/Pix2PixCM/train_pix2pixcm.py --dataset dataset_name  
```

The results are then saved in the corresponding output files.
