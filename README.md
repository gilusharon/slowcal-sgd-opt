

# Slowcal-SGD Optimizations: Improving on Slowcal-SGD and Implementing it for Non-Convex Optimization

[![arXiv](https://img.shields.io/badge/arXiv-2304.04169-B31B1B.svg)](https://arxiv.org/abs/2304.04169) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This is the official repository for **SLowcalSGD Optimization**, based on the paper **"SLowcal-SGD: Slow Query Points Improve Local-SGD for Stochastic Convex Optimization"**, authored by Tehila Dahan and Kfir Y. Levy, and accepted at **NeurIPS 2024**. This repository is an extension and refactoring of the original [**SLowcalSGD**](https://github.com/dahan198/slowcal-sgd.git) repository. 

This repository uses the implementation of SLowcalSGD from the original repository, and adds the non-stochastic case.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/gilusharon/slowcal-sgd-opt
cd slowcal-sgd
```

### 2. Install Dependencies

First, ensure that PyTorch is installed. You can install it by selecting the appropriate command based on your environment from [PyTorch's official website](https://pytorch.org/get-started/locally/).

#### Install Other Dependencies

After installing PyTorch, install the remaining dependencies using:

```bash
pip install -r requirements.txt
```

---

## Prepare the Dataset

By default, the code downloads the MNIST dataset automatically and stores it in the `./data` directory.

---

## Running Experiments

### Custom Training with SLowcalSGD

If you want to perform custom training using SLowcalSGD, run the following command:

```bash
python main.py --config_folder_path ./config --dataset mnist --model logistic_regression --optimizer SLowcalSGD
```

You can modify the training parameters, such as the number of workers, batch size, learning rate, and local iterations, by editing the configuration files or passing them as command-line arguments.

---

## Results and Logging

- **Weights & Biases (wandb)**: This repository supports logging with [Weights & Biases](https://wandb.ai/) for experiment tracking and visualization. Ensure you have a `wandb.yaml` file in the `config` directory with your project and entity name.
  
  Example `wandb.yaml`:
  ```yaml
  project: "SLowcalSGD"
  entity: "your-wandb-username"
  ```

- **Metrics**: Training metrics, such as loss and accuracy, are logged during the training process and saved in the 'results' directory.

---

## Citation

If you find this code useful in your research, please consider citing the original paper:

```
@inproceedings{slowcal_sgd2024,
  title={SLowcal-SGD: Slow Query Points Improve Local-SGD for Stochastic Convex Optimization},
  author={Tehila Dahan and Kfir Yehuda Levy},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
