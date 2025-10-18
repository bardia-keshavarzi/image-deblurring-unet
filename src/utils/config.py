import yaml
from pathlib import Path

class Config:
    """Loads configuration from YAML file"""
    
    def __init__(self, config_path="configs/config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self):
        """Load YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key, default=None):
        """
        Get config value using dot notation
        
        Example:
            config.get('traditional.unsharp_mask.sigma')
            Returns: 1.0
        """
        keys = key.split('.')  # Split "traditional.wiener.noise" into ["traditional", "wiener", "noise"]
        value = self._config
        
        # Navigate through nested dictionaries
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default  # Key not found
        
        return value