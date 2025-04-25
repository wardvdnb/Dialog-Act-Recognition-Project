import os
import json
import pandas as pd

def format_json(train_path, test_path):
    """
    Format the JSON files for the training and testing datasets.
    For readability purposes, ensures proper structure in the train/test set files.
    This code can be ignored for the actual project.
    """
    train_path = train_path if train_path else os.path.join(os.getcwd(), 'SGD Dataset', 'TrainSet.json')
    test_path = test_path if test_path else os.path.join(os.getcwd(), 'SGD Dataset', 'TestSet.json')

    def format_json(path):
        # For readability purposes, ensure proper structure in the train/test set files
        with open(path, "r") as f:
            data = json.load(f)

        with open(path.replace(".json", "_formatted.json"), "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    format_json(train_path)
    format_json(test_path)

def compute_avg_training_time(file_path):
    df = pd.read_csv(file_path)
    print(f"Average training time is: {str(df['Time (s)'].aggregate('mean'))} seconds")