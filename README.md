### \# Distribution Adaptive Decomposition Forecasting (DADF) Framework



This repository contains the implementation of the DADF framework for non-stationary sequence forecasting (with a focus on wind speed prediction), alongside the data preprocessing, model training, and evaluation scripts. 



#### \## 1. Repository Structure



The project pipeline is modularized. The core directories and scripts are organized as follows:



 \- `main.py`: The main entry point to run the pipeline, set hyperparameters, and trigger parameter sensitivity analysis.

 \- `pipeline.py`: The core `TimeSeriesForecasting` class handling data splitting, dynamic time warping extension (DTWE), signal decomposition, and ensemble predictions.

 \- `module/`

 &#x20; - `preprocessing/`: Scripts for data preprocessing and STL signal decomposition.

 &#x20; - `models/`: Implementations of Extreme Learning Machines (ELM) and Sparrow Search Algorithm (SSA) optimization (utilizing L2 regularization for robust generalization).

 &#x20; - `utils/`: Includes the core implementation of the DTW extension (`dtw\_extension.py`).

 &#x20; - `index/`: Evaluation metrics (MAE, RMSE, MAPE, IA) and statistical tests.

 \- `requirements.txt`: Software versions and dependencies required to reproduce the environment.


#### \## 2. Software Versions \& Environment



 The code has been developed and tested using \*\*Python 3.12 / 3.9\*\*. To ensure full reproducibility, please set up the environment using the provided requirements file:



 ```bash

 \# Create a virtual environment (optional but recommended)

 conda create -n dadf\_env python=3.12

 conda activate dadf\_env

 

 \# Install dependencies

 pip install -r requirements.txt

