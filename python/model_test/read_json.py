import json

with open("/home/sunwei/Github_origin/HPC_PRACTICE/python/model/config.json", "r") as f:
    config = json.load(f)

print(config["architectures"][0])
print(config["vocab_size"])

try:
	print(config["bias"])
except:
    print("No bias found in config")
    