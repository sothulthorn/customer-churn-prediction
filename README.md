# Customer Churn Prediction

A deep learning project that predicts whether a bank customer will churn (leave the bank) using an Artificial Neural Network (ANN) built with TensorFlow/Keras. Includes an interactive Streamlit web app for real-time predictions.

## Dataset

The project uses the **Churn_Modelling.csv** dataset containing 10,000 bank customer records with the following features:

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Customer's country (France, Germany, Spain) |
| Gender | Male or Female |
| Age | Customer's age |
| Tenure | Number of years as a bank customer |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether the customer has a credit card (0/1) |
| IsActiveMember | Whether the customer is an active member (0/1) |
| EstimatedSalary | Customer's estimated salary |
| **Exited** | **Target variable** - whether the customer churned (0/1) |

## Model Architecture

A Sequential ANN with the following layers:

```
Input (12 features) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Sigmoid)
```

- **Optimizer:** Adam (learning_rate=0.01)
- **Loss:** Binary Crossentropy
- **Callbacks:** EarlyStopping (patience=10, restore_best_weights=True), TensorBoard
- **Validation Accuracy:** ~86%

## Preprocessing Pipeline

1. Drop irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
2. Label encode `Gender` (Female=0, Male=1)
3. One-hot encode `Geography` (France, Germany, Spain)
4. Standard scale all features

Fitted encoders and scaler are saved as pickle files:
- `label_encoder_gender.pkl`
- `onehot_encoder_geo.pkl`
- `scaler.pkl`

## Project Structure

```
customer-churn-prediction/
├── app.py                      # Streamlit web application
├── experiments.ipynb            # Model training and experimentation notebook
├── prediction.ipynb             # Inference/prediction demo notebook
├── model.h5                     # Trained Keras model
├── Churn_Modelling.csv          # Dataset
├── label_encoder_gender.pkl     # Fitted LabelEncoder for Gender
├── onehot_encoder_geo.pkl       # Fitted OneHotEncoder for Geography
├── scaler.pkl                   # Fitted StandardScaler
├── logs/                        # TensorBoard training logs
├── requirements.txt             # Python dependencies
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.10+

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd customer-churn-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Linux/macOS
   .venv\Scripts\activate           # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

This launches a web interface where you can input customer details and get a churn probability prediction.

### Retrain the Model

Open and run `experiments.ipynb` in Jupyter to retrain the model. This will regenerate:
- `model.h5`
- `label_encoder_gender.pkl`
- `onehot_encoder_geo.pkl`
- `scaler.pkl`

### View Training Logs

```bash
tensorboard --logdir logs/fit
```

## Key Dependencies

- TensorFlow 2.20
- scikit-learn 1.8
- Streamlit 1.54
- pandas, numpy
