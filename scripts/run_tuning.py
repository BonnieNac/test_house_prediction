"""Hyperparameter search: tune the best models and save the best one in models/ (make tune, options: --help)."""

from test_house_prediction.core.models.tune import main
from test_house_prediction.core.utils import setup_file_logging

if __name__ == "__main__":
    setup_file_logging("tune")
    main()
