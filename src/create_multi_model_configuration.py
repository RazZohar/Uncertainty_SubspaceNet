import json
from copy import deepcopy


def create_subarray_config_from_file(input_filename: str, output_filename: str, L: int):
    # Load the full dictionary (containing "system_model") from file
    with open(input_filename, 'r') as f:
        data = json.load(f)

    # Extract system_model only
    system_model = data["system_model"]

    # Construct the output config
    config = {
        "L": L,
        "subarray_config": [{"system_model": deepcopy(system_model)} for _ in range(L)]
    }

    # Write to output file
    with open(output_filename, 'w') as f:
        json.dump(config, f, indent=2)


# Example usage
input_file = "system_model.json"
output_file = "subarray_config.json"
L = 4

create_subarray_config_from_file(input_file, output_file, L)
