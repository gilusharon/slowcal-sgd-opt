from run import run_experiment
import json
path = "./config/simple_conv_tuning.json"
with open(path, 'r') as f:
    config = json.load(f)

run_experiment(config)