# ICS5200-Dissertation
This repository contains the codebase for my MSc in Artificial Intelligence Dissertation at the University of Malta, which falls under study unit ***ICS5200***.

## Getting Started with the Application

### Prerequisities
- Python 3.10+

### 1. Clone the repository
The first step is to clone the entire repository, ensuring you have the available code and resources to start the program:

```bash
git clone https://github.com/Matthias-VP-UoM/ICS5200-Dissertation.git
```

### 2. Install package dependencies
In order to use the program, the correct dependencies must be installed. These can be installed using the provided requirements.txt file, which can be done using the below command:

```bash
pip install -r requirements.txt
```

**Important: Kindly note that although the versions of the libraries indicated in the requirements.txt file will work fine with the code provided, the use of Google Colab may result in these libraries utilising a more recent version. Additional details on the use of Colab in this project is provided below.**

### 3. Gather the necessary data and files
Due to file size constraints, the repository does not include the full dataset or the models that were trained using the training scripts provided.

Instead, these can be accessed by using the following Google link:
https://drive.google.com/drive/folders/1dgEqFXd9tdGMVD-4JgXmYyqXCsUZj4Ah?usp=sharing

To use the files contained in the link, first download and then upload them to the following directories:
- The **"models"** folder should be placed inside the "outputs" directory in the root of the project working directory.
- The **"dataset_final"** and **"dataset_raw"** folders should be placed in the root of the project working directory.

**Note: If it asks you to replace any files, kindly refrain from doing so, as this will only allow you to place the missing files which are not available in this repository.**

## Note Regarding the Use of Google Colab

Throughout the development of this pipeline, only the **train_dl_baselines** and **train_hybrid_models** Jupyter notebooks were run using Google Colab due to the need to train, store and use high resource models (mainly the deep learning and hybrid models implemented in this project), which could not be run on the computational resources that were available on the machine used.

While the rest of the Python notebooks can be run normally without necessarily needing to use Google Colab, kindly feel free to use the software for these notebooks if you prefer.
