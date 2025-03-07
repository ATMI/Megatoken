from types import SimpleNamespace
import yaml


def dict_to_namespace(d):
	if isinstance(d, dict):
		return SimpleNamespace(**{
			k: dict_to_namespace(v)
			for k, v in d.items()
		})
	return d


def load_config(config_path):
	with open(config_path, "r") as f:
		config = yaml.safe_load(f)
	return dict_to_namespace(config)
