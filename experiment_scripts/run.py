import subprocess

# Iterate over parameter combinations and execute commands
def run_experiment(config: dict):
    for seed in config["seeds"]:
        for batch_size in config["batch_sizes"]:
            for worker_count in config["worker_counts"]:
                for local_iteration_num in config["local_iteration_nums"]:
                    for optimizer in config["optimizers"]:
                        # Construct the command
                        command = [
                            'python', 'main.py',
                            '--workers_num', str(worker_count),
                            '--config_folder_path', config["config_folder_path"],
                            '--dataset', config["dataset"],
                            '--model', config["model"],
                            '--epoch_num', str(config["num_epochs"]),
                            '--eval_interval', str(config["eval_interval"]),
                            '--local_iterations_num', str(local_iteration_num),
                            '--optimizer', optimizer,
                            '--learning_rate', str(config["learning_rates"][optimizer]),
                            '--batch_size', str(batch_size),
                            '--seed', str(seed),
                            '--weight_decay', str(config["weight_decay"]),
                            '--experiment_name', config["experiment_name"]
                        ]

                        # Add optional flags
                        if config["use_alpha_t"]:
                            command.append('--use_alpha_t')
                        if config["use_wandb"]:
                            command.append('--use_wandb')

                        # Execute the command
                        subprocess.run(command, check=True)
