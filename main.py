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
    with open("arguments.json", "r") as f:
        all_args = load(f)
    for arg_name, arg in all_args.items():
        if "type" in arg:
            arg_type = eval(arg["type"])
            del arg["type"]
            parser.add_argument(f"--{arg_name}", type=arg_type, **arg)
        else:
            parser.add_argument(f"--{arg_name}", **arg)

    return parser.parse_args()


def get_dataloaders(data_args):
    """Loads the dataset and prepares dataloaders for training and testing."""
    dataset = DATASET_REGISTRY[data_args.dataset]()
    minibatch_size = data_args.batch_size * data_args.local_iterations_num
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
        workers_arr.append(Worker(worker_optimizer, w_dataloaders[i], worker_model, device))
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
