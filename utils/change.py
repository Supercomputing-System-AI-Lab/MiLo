data_dict = {}

with open("/home/exouser/MiLo/utils/kurtosis_mixtral.txt", "r") as f:
    for line in f:
        if ":" in line:
            key, value = line.strip().split(": ", 1)
            data_dict[key] = float(value)

import json
with open("/home/exouser/MiLo/utils/Mixtral_kurtosis_values.json", "w") as f:
    json.dump(data_dict, f, indent=4)
