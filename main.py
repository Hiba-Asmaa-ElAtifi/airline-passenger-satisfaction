
from pathlib import Path
import yaml

# Load the project configuration
PROJECT_DIR = Path(__file__).resolve().parent

with open(PROJECT_DIR / "configs" / "config.yaml", "r") as file:
    config = yaml.safe_load(file)

def main():
    print("Airline Passenger Satisfaction")
    print("Project configuration loaded successfully!")

    print("Training dataset:", config["data"]["train_path"])
    print("Testing dataset:", config["data"]["test_path"])
    print("Target:", config["data"]["target"])

    # Next: data loading, preprocessing and model training
    # These steps will be connected when the modules are ready.

if __name__ == "__main__":
    main()
