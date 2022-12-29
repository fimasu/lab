import yaml

with open("sample.yaml", "r") as yaml_file:
    yaml_data: dict = yaml.safe_load(yaml_file)

if yaml_data is None:
    yaml_data = dict()

print("before")
print(yaml_data)
yaml_data["test"] = "fuga"
print("after")
print(yaml_data)

with open("sample.yaml", "w") as yaml_file:
    yaml.safe_dump(yaml_data, yaml_file, sort_keys=False)
