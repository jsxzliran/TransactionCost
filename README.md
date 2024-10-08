# TransactionCost
This project aims at finding a way to mitigate particular trading strategy considering transaction cost. To see how it works, we suggest using anaconda/miniconda with some python IDE like Pycharm or Jupyter notebook or MS Vscode. High performance graph card is required for training.

## Installation

1. [Download and install Anaconda](https://www.anaconda.com/products/individual)
2. Enter CMD mod with administration account
3. Use the following codes, replace myenv with the name you prefer
3. Notice this will require high memory and performance of graphic card, with at least 32GB system memory and 16GB GPU memory
4. Alternatively, one can choose to use Google Colab service
  ```
  conda create --name myenv
  ```
4. Activate your environment with:
  ```
  conda activate myenv
  ```
5. Download the files and move to the folder in CMD, use the following codes to install packages for your environment:
  ```
  conda install --file requirements.txt
  ```
6. Don't forget to install the pytorch:
```
conda install pytorch torchvision torchaudio cudatoolkit=12.4 -c pytorch -c conda-forge
```
Here you need to figure out your particular graph card and related drivers file to install

7. And example IDE is jupyterlab:
```
conda install -c conda-forge jupyterlab
```
8. Open OneStockTradingCost.ipynb with your IDE(with the pre-set environment) and run it!
9. The multi-assets models(>1) might require large memory of graph cards and I use google colab(the first few lines of notebook is for colab if you do not use just ignore)
