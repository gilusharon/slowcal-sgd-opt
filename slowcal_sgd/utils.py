import torch
import numpy as np
import random
import inspect
import matplotlib.pyplot as plt
import csv
import json
from os import path, listdir
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
import seaborn as sns


def split_dataset(dataset, num_splits, batch_size, seed, shuffle=True, dirichlet_alpha=None, plot_distribution=False):
    """
    Splits a dataset into `num_splits` subsets, ensuring class distribution is preserved using stratified k-fold splitting.
    Optionally, controls data heterogeneity via a Dirichlet distribution and provides an option to visualize class distribution.

    Args:
        dataset (torch.utils.data.Dataset): The input dataset to be split.
        num_splits (int): The number of subsets to generate.
        batch_size (int): Batch size to be used for the resulting DataLoader instances.
        seed (int): Seed value for reproducibility of random operations.
        shuffle (bool): Whether to shuffle data within each subset. Defaults to True.
        dirichlet_alpha (float, optional): Parameter controlling heterogeneity when using a Dirichlet distribution. Defaults to None.
        plot_distribution (bool, optional): Flag to indicate whether to plot the class distribution across subsets. Defaults to False.

    Returns:
        list[torch.utils.data.DataLoader]: A list containing DataLoader instances for each generated subset.
    """
    set_seed(seed)

    # Convert dataset targets to a NumPy array
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    total_samples = len(dataset)

    if dirichlet_alpha is not None:
        # Create class-wise indices for Dirichlet-based splitting
        indices_per_class = {c: np.where(targets == c)[0] for c in range(num_classes)}
        subset_indices = [[] for _ in range(num_splits)]

        # Generate Dirichlet proportions for each class
        proportions = np.random.dirichlet([dirichlet_alpha] * num_splits, num_classes)

        # Initialize ordered proportions and calculate the threshold
        ordered_proportions = np.zeros_like(proportions)
        proportions_threshold = num_classes / num_splits
        samples_per_split = total_samples // num_splits

        # Assign proportions ensuring subset balance
        for c in range(num_classes):
            for i, p in enumerate(proportions[c]):
                assigned = False
                for j, op in enumerate(ordered_proportions[c]):
                    if ((ordered_proportions[:, j].sum() + p < proportions_threshold
                         or ordered_proportions[:, j].sum() == 0) and op == 0):
                        ordered_proportions[c, j] = p
                        assigned = True
                        break
                if not assigned:
                    indices = np.where(ordered_proportions[c] == 0)
                    min_idx = np.argmin(ordered_proportions[:, indices[0]].sum(axis=0))
                    ordered_proportions[c, [indices[0][min_idx]]] = p

        # Normalize proportions to ensure consistent subset sizes
        classes_amounts = ordered_proportions * total_samples // num_classes

        for c, indices in indices_per_class.items():
            random_indices = np.random.permutation(indices)
            current_idx = 0
            split_sizes = classes_amounts[c].astype(int)

            for i, size in enumerate(split_sizes):
                subset_indices[i].extend(random_indices[current_idx:current_idx + size])
                current_idx += size

        subsets = [Subset(dataset, indices) for indices in subset_indices]
    else:
        # Use StratifiedKFold for class-balanced splits
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
        subsets = [Subset(dataset, indices) for _, indices in skf.split(np.zeros(len(targets)), targets)]

    if plot_distribution:
        plot_class_distribution(subsets, num_classes)

    # Create DataLoader instances for each subset
    dataloaders = [DataLoader(subset, batch_size=batch_size, shuffle=shuffle) for subset in subsets]

    return dataloaders


def plot_class_distribution(subsets, num_classes, palette=None):
    """
    Plots the class distribution across subsets as a scatter plot.
    The size of each scatter point represents the number of samples, and the y-axis represents the class ID.

    Args:
        subsets (list[torch.utils.data.Subset]): List of dataset subsets.
        num_classes (int): Number of unique classes in the dataset.
        palette (list or None, optional): List of colors to use for plotting. If None, a color palette is generated dynamically.
    """
    class_counts = np.zeros((len(subsets), num_classes))

    for i, subset in enumerate(subsets):
        targets = np.array([subset.dataset.targets[idx] for idx in subset.indices])
        counts = np.bincount(targets, minlength=num_classes)
        class_counts[i] = counts

    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate a dynamic palette if not provided
    if palette is None:
        palette = sns.color_palette('tab10', num_classes)  # Use tab10 for up to 10 distinct colors

    # Plot each class with its corresponding color
    for c in range(num_classes):
        ax.scatter(
            np.arange(len(subsets)),
            [c] * len(subsets),
            s=class_counts[:, c],  # Scale size by number of samples
            color=palette[c % len(palette)],  # Cycle through palette if needed
            alpha=0.7
        )

    ax.set_xlabel('Worker ID')
    ax.set_ylabel('Class ID')
    plt.tight_layout()
    plt.savefig(f'class_distribution_{len(subsets)}.png')
    plt.show()


def set_seed(seed):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to be set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_valid_args(object_class, **kwargs):
    """
    Filters keyword arguments to only include those valid for the specified object's initializer.

    Args:
        object_class (type): The class whose initializer signature is used for filtering.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        dict: A dictionary containing only valid arguments for the specified object's initializer.
    """
    init_signature = inspect.signature(object_class.__init__)
    valid_params = set(init_signature.parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return filtered_kwargs


def get_device():
    """
    Returns the appropriate computing device (CPU, CUDA, or MPS) based on availability.

    Returns:
        torch.device: The available computing device.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def check_for_invalid_folders(results_dir: str = "results"):
    """
    Checks for invalid folder names in the results directory. Valid folders should be named 'run' followed by a number.
    Raises an error if any invalid folder names are found.
    Args:
        results_dir (str): The results folder to verify.
    """
    if not path.exists(results_dir):
        raise FileNotFoundError(f"Results folder: {results_dir} not found.")
    dirs = listdir(results_dir)
    if not dirs:
        raise FileNotFoundError("Results folder is empty. Please run the training script first.")
    for d in dirs:
        try:
            assert d[:3] == "run"
            int(d[3:])
        except (ValueError, AssertionError):
            raise ValueError(f"Invalid folder name {d} found in results folder. Did you change anything manually?")


def plot_from_path(results_dir, run_dir, axes, params: list = None):
    """
    Reads a CSV file and plots the test loss and accuracy from the results.

    Args:
        f (file): The file object to read from.
        axes (matplotlib.axes.Axes): The axes to plot on.
        params (list): Parameters for the plot label.
    """
    test_loss = []
    test_accuracy = []
    with open(path.join(results_dir, run_dir, "results.csv"), "r", newline='') as f:
        reader = csv.DictReader(f)
        for entry in reader:
            test_loss.append(float(entry["Test Loss"]))
            test_accuracy.append(float(entry["Test Accuracy"]))
    iterations = np.arange(len(test_loss))
    label = run_dir if params is None else ", ".join(params)
    axes[0].plot(iterations, test_loss, label=label)
    axes[1].plot(iterations, test_accuracy, label=label)


def verify_params(params: list):
    with open("arguments.json", "r") as f:
        all_args = json.load(f)
    for param in params:
        assert(param in all_args), f"Parameter {param} not found in arguments.json. Please check the parameter name."


def get_plots_dirs(results_dir: str, params: list) -> list:
    """
    Returns a list of the directories of the most recent experiments that have a unique combination of parameters.
    Args:
        results_dir (str): The results folder to check.
        params (dict): The parameters to filter the directories by.
    """
    to_plot = []
    seen = []
    # Sort experiments from most recent to oldest
    for d in sorted(listdir(results_dir), key=lambda x: -int(x[3:])):
        with open(path.join(results_dir, d, "params.json"), "r", newline='') as f:
            params_dict = json.load(f)
            for param in params_dict:
                if (params is not None) and (param not in params):
                    del params_dict[param]
            if params_dict in seen:
                continue
            seen.append(params_dict)
            to_plot.append(d)
    return to_plot


def plot_results(params: list = None, experiment_name: str = None):
    """
    Plots the data from the results folder as a line graph. Graph includes test loss and test accuracy and
    always uses the most recent results.
    Args:
        params (list of strings): Which parameter combination does the function consider a distinct experiment to graph?
        If two experiments are identical regarding the parameters on the list, only the most recent one will be graphed.
        experiment_name (str): Used to determine the results folder. Defaults to 'results' if None.
    """
    results_dir = 'results' if experiment_name is None else f'results-{experiment_name}'
    check_for_invalid_folders(results_dir)
    if params is not None:
        verify_params(params)

    plot, axes = plt.subplots(nrows=2, ncols=1)
    plot.tight_layout()

    for d in get_plots_dirs(results_dir, params):
        plot_from_path(results_dir, d, axes, params=params)

    axes[0].set_title("Test Loss")
    axes[1].set_title("Test Accuracy")
    axes[0].legend()
    axes[1].legend()
    plt.show()
