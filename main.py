import argparse
from json import load
from slowcal_sgd.dataset import DATASET_REGISTRY
from slowcal_sgd.model import MODEL_REGISTRY
from slowcal_sgd.optimizer import OPTIMIZER_REGISTRY
from torch.utils.data import DataLoader
from slowcal_sgd.utils import set_seed, get_device, split_dataset
from slowcal_sgd.worker import Worker
from slowcal_sgd.trainer import TRAINER_REGISTRY


def parse_arguments():
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Training script for synchronous Byzantine machine learning.")

    # Argument definitions
    parser.add_argument('--workers_num', type=int, default=16,
                        help='Number of workers used for training.')
    parser.add_argument('--config_folder_path', type=str, default='./config',
                        help='Path to the configuration folder.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist'],
                        help='Dataset to be used.')
    parser.add_argument('--model', type=str, default='logistic_regression', choices=MODEL_REGISTRY.keys(),
                        help='Model architecture to be used.')
    parser.add_argument('--epoch_num', type=int, default=1, help='Number of epochs for training.')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Interval (in epochs) for evaluation.')
    parser.add_argument('--local_iterations_num', type=int, default=64,
                        help='Number of local iterations per worker.')
    parser.add_argument('--optimizer', type=str, default='SLowcalSGD',
                        choices=['LocalSGD', 'SLowcalSGD', 'MinibatchSGD'],
                        help='Optimizer to be used for training.')
    parser.add_argument('--learning_rate', type=float,
                        default=0.1, help='Learning rate for the optimizer.')
    parser.add_argument('--use_alpha_t', action='store_true',
                        help='Enable use of alpha_t=t in the optimizer.')
    parser.add_argument('--query_point_momentum', type=float, default=0.1,
                        help='Fixed momentum for the query point if alpha_t is not used.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--seed', type=int, default=3, help='Random seed for reproducibility.')
    parser.add_argument('--use_wandb', action='store_true', help='Enable logging with Weights & Biases.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer.')
    parser.add_argument('--experiment_name', type=str,
                        help='Name of the experiment for logging and identification.')
    parser.add_argument('--dirichlet_alpha', type=float, default=None,
                        help='Alpha parameter for Dirichlet distribution to control data heterogeneity among workers. '
                             'If set, data will be sampled non-uniformly across workers based on this parameter.')

    return parser.parse_args()


def get_dataloaders(data_args):
    """Loads the dataset and prepares dataloaders for training and testing."""
    dataset = DATASET_REGISTRY[data_args.dataset]()
    minibatch_size = data_args.batch_size * data_args.workers_num
    test = DataLoader(dataset.testset, batch_size=minibatch_size, shuffle=False)
    batch_size = data_args.batch_size if data_args.optimizer in ["LocalSGD", "SLowcalSGD"] else minibatch_size
    train = split_dataset(dataset=dataset.trainset, num_splits=data_args.workers_num, batch_size=batch_size,
                                      seed=data_args.seed)
    return train, test


def get_worker_optimizer(opt_args, opt_model):
    """Initializes the optimizer for a worker model."""
    # Configure optimizer and parameters
    if opt_args.optimizer in ["LocalSGD", "MinibatchSGD"]:
        optimizer = OPTIMIZER_REGISTRY["sgd"]
        hyperparameters = {
            "lr": opt_args.learning_rate,
            "momentum": 0.0,
            "weight_decay": opt_args.weight_decay
        }
    else:
        optimizer = OPTIMIZER_REGISTRY["anytime_sgd"]
        hyperparameters = {
            "lr": opt_args.learning_rate,
            "gamma": opt_args.query_point_momentum,
            "use_alpha_t": opt_args.use_alpha_t,
            "weight_decay": opt_args.weight_decay
        }
    return optimizer(opt_model.parameters(), **hyperparameters)


def init_workers(w_args, w_dataloaders):
    workers_arr = []
    for i in range(w_args.workers_num):
        worker_model = MODEL_REGISTRY[args.model]().to(device)
        worker_optimizer = get_worker_optimizer(w_args, worker_model)
        workers_arr.append(Worker(worker_optimizer, train_dataloaders[i], worker_model, device))
    return workers_arr


if __name__ == "__main__":
    """Main function for initializing and running the training process."""
    args = parse_arguments()
    set_seed(args.seed)

    device = get_device()
    model = MODEL_REGISTRY[args.model]().to(device)
    train_dataloaders, test_dataloader = get_dataloaders(args)

    workers = init_workers(args, w_dataloaders=train_dataloaders)

    trainset_length = len(DATASET_REGISTRY[args.dataset]().trainset)

    # Initialize trainer
    trainer = TRAINER_REGISTRY[args.optimizer](model, test_dataloader, args, workers, device,
                                               trainset_length=trainset_length, experiment_name=args.experiment_name)

    # Start training
    trainer.train(epoch_num=args.epoch_num, eval_interval=args.eval_interval)
