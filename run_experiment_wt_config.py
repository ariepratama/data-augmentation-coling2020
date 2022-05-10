import os
import subprocess as sub
import sys
from datetime import datetime

import yaml

DATETIME_FORMAT = "%Y-%m-%dT%H-%M-%S"
if __name__ == "__main__":
    experiment_config_file_path = sys.argv[1]
    sizes = []
    non_terminals = []
    generated_sentences = []
    augmentations = []
    seeds = []
    with open(experiment_config_file_path, "r") as f:
        experiment_config = yaml.safe_load(f)
        now = datetime.now()
        now_str = now.strftime(DATETIME_FORMAT)

        for config in experiment_config["experiments"]:
            size = config["size"]
            run_datetime = now_str
            replaced_non_terminal = config["replaced_non_terminal"]
            num_generated_sentences = config["num_generated_sentences"]
            augmentation = config["augmentation"]
            time = str(config["time"])
            n_replace_non_terminals = config["n_replace_non_terminals"]

            random_seeds = config["random_seeds"]
            for n_generated_sentence in num_generated_sentences:
                for n_replace_no_terminal in n_replace_non_terminals:
                    for random_seed in random_seeds:
                        os.environ["SIZE"] = config["size"]
                        os.environ["RUN_DATETIME"] = now_str
                        os.environ["REPLACED_NON_TERMINAL"] = config["replaced_non_terminal"]
                        os.environ["NUM_GENERATED_SENTENCES"] = str(n_generated_sentence)
                        os.environ["AUGMENTATION"] = config["augmentation"]
                        os.environ["RANDOM_SEED"] = str(random_seed)
                        cmd = "bash submit_job.sh {} {} {} {} {} {} {} {}".format(
                                size,
                                run_datetime,
                                replaced_non_terminal,
                                n_generated_sentence,
                                augmentation,
                                random_seed,
                                time,
                                n_replace_no_terminal
                            )
                        print(f"running '{cmd}'")
                        sub.run(cmd.split())
