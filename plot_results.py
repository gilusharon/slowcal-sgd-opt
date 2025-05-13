from slowcal_sgd.utils import plot_results
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting function for experiment results.")
    parser.add_argument(
        "--params",
        type=str,
        default=None
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None
    )
    args = parser.parse_args()
    plot_results(params=args.params, experiment_name=args.experiment_name)
