import json
from config import Config
from config_persistence import save_config

# Create a config instance with default trigger_bus values
config = Config()

# Save config to config.json
save_config(config)

print("Config with trigger_bus saved to config.json.")
