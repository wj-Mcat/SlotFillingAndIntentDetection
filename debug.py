import json
import shutil
import sys

from allennlp.commands import main

## 对Allennlp代码进行调试

config_file = "joint_models/config.json"

# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": -1}})

serialization_dir = "./output"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debu  gging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "joint_models",
    "-o", overrides,
]



main()