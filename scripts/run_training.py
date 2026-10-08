"""Training: compare the models and save the best one in models/ (make train, options: --help)."""

from test_house_prediction.core.models.train import main
from test_house_prediction.core.utils import setup_file_logging

if __name__ == "__main__":
    setup_file_logging("train")
    main()
