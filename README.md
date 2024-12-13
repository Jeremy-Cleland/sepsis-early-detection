# Sepsis Prediction Project

## Overview

This project aims to predict sepsis in patients using machine learning models. The workflow includes data preprocessing, feature engineering, model training, evaluation, and testing on external data.

## Project Structure

```bash
sepsis_prediction/
│
├── data/
│   ├── raw/
│   │   └── Dataset.csv
│   ├── processed/
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│   │   └── val_data.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── evaluation.py
│   └── utils.py
│
├── tests/
│   └── test_data_processing.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/sepsis_prediction.git
    cd sepsis_prediction
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the main script:**
gi

    ```bash
    python main.py
    ```

## Usage

- **Data Processing:** Located in `src/data_processing.py`.
- **Feature Engineering:** Located in `src/feature_engineering.py`.
- **Model Training:** Located in `src/models.py`.
- **Evaluation:** Located in `src/evaluation.py`.
- **Utility Functions:** Located in `src/utils.py`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

[MIT License](LICENSE)

# predicting-sepsis
