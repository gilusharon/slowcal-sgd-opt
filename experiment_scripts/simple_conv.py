from run import run_experiment
import json
path = "./config/slowcal_simple_conv.json"
with open(path, 'r') as f:
    config = json.load(f)

run_experiment(config)
