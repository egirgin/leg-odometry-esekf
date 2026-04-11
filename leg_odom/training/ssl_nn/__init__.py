"""Self-supervised neural contact training (training-only, runtime wiring deferred)."""

from leg_odom.training.ssl_nn.config import default_ssl_train_config_path, load_ssl_train_config

__all__ = ["default_ssl_train_config_path", "load_ssl_train_config"]
