
Some useful instructions for AWS Sagemaker.

### Make a conda env available in jupyter notebooks

**Step 1:** Source `~/.bashrc` to make the Anaconda CLI work

```shell
source ~/.bashrc
```

**Step 2:** Create a conda env (if not done already)

```shell
mkdir ~/SageMaker/conda_envs
conda create --path ~/SageMaker/conda_envs/NAME python=3.11
```

**Step 3:** Activate the conda env you want to make available

```shell
conda activate ~/SageMaker/conda_envs/NAME
```

**Step 4:** Install the kernel (needed to make the environment available in jupyter notebooks)

```shell
python -m ipykernel install --user --name=NAME
```
