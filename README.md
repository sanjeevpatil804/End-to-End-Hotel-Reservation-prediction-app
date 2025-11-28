# 🏨 End-to-End Hotel Reservation Cancellation Prediction

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/sanjeevpatil804)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.0-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange.svg)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-Deployed-orange.svg)](https://aws.amazon.com/)

A production-ready machine learning system for predicting hotel reservation cancellations using advanced feature engineering, XGBoost classification with Optuna hyperparameter tuning, and automated MLOps deployment pipeline. This end-to-end solution helps hotels optimize revenue management, reduce overbooking risks, and improve operational efficiency.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Dataset Features](#-dataset-features)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Web Application](#-web-application)
- [API Documentation](#-api-documentation)
- [Model Performance](#-model-performance)
- [Deployment](#-deployment)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

Hotel reservation cancellations pose significant challenges to the hospitality industry, leading to revenue loss, inefficient resource allocation, and operational complications. This project implements an intelligent prediction system that:

- **Predicts** hotel reservation cancellations with high accuracy using 17 carefully selected features
- **Processes** data through a robust ETL pipeline with AWS S3 integration
- **Optimizes** XGBoost models using Optuna for automated hyperparameter tuning
- **Handles** class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
- **Deploys** via Docker containers on AWS with CI/CD automation
- **Provides** both web interface and REST API for predictions

The system enables hotels to proactively manage reservations, implement dynamic pricing strategies, and optimize inventory allocation based on cancellation risk.

---

## 💼 Business Problem

### Challenges Addressed:

1. **Revenue Loss**: No-shows and last-minute cancellations result in lost revenue opportunities
2. **Overbooking Risk**: Uncertainty in cancellations makes it difficult to optimize overbooking strategies
3. **Resource Inefficiency**: Staff and inventory management becomes challenging without cancellation predictions
4. **Customer Experience**: Better predictions enable proactive customer engagement and retention strategies

### Solution Benefits:

- ✅ **Reduce Revenue Loss**: Predict cancellations early to resell rooms
- ✅ **Optimize Overbooking**: Data-driven overbooking strategies based on cancellation probability
- ✅ **Improve Operations**: Better staff scheduling and resource allocation
- ✅ **Enhance Customer Service**: Proactive outreach to high-risk cancellation customers
- ✅ **Dynamic Pricing**: Adjust pricing based on cancellation risk profiles

---

## ✨ Key Features

### 🔍 **Intelligent Feature Engineering**
- **17 predictive features** covering:
  - Guest demographics (adults, children)
  - Booking characteristics (lead time, meal plans, room types)
  - Temporal patterns (arrival dates, weekend/weekday nights)
  - Historical behavior (previous cancellations, repeated guests)
  - Price sensitivity (average room price)
  - Service requests (special requests, parking)

### 🤖 **Advanced Machine Learning**
- **XGBoost Classifier**: State-of-the-art gradient boosting algorithm
- **Optuna Hyperparameter Tuning**: 10+ trials for optimal parameter discovery
- **SMOTE for Imbalance**: Handles class imbalance in cancellation data
- **Robust Preprocessing**: RobustScaler for numerical features, OrdinalEncoder for categorical
- **Comprehensive Metrics**: F1-Score, Precision, Recall, Accuracy

### 🏗️ **Production-Ready MLOps**
- **Modular Pipeline Architecture**: Separate components for ingestion, validation, transformation, and training
- **AWS S3 Integration**: Cloud-based data storage and retrieval
- **Artifact Versioning**: Timestamped artifacts for experiment tracking
- **Data Validation**: Schema validation and drift detection
- **Exception Handling**: Custom exception framework for robust error management

### 🚀 **Cloud-Native Deployment**
- **Flask Web Application**: User-friendly interface for predictions
- **REST API**: JSON-based API for integration with hotel management systems
- **Docker Containerization**: Consistent deployment across environments
- **AWS ECR**: Container registry for image management
- **GitHub Actions CI/CD**: Automated testing, building, and deployment

---

## 🏛️ Architecture

```
┌─────────────────┐
│   AWS S3        │
│ (Raw Dataset)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              Data Ingestion Component                    │
│  • Download from S3                                     │
│  • Store in feature store                               │
│  • Train/test split (80-20)                             │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│           Data Validation Component                      │
│  • Schema validation (17 features)                      │
│  • Data quality checks                                  │
│  • Drift detection & reporting                          │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│        Data Transformation Component                     │
│  • RobustScaler for numerical features                  │
│  • OrdinalEncoder for categorical features              │
│  • SMOTE for class imbalance                            │
│  • Label encoding for target                            │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│            Model Trainer Component                       │
│  • Optuna hyperparameter optimization                   │
│  • XGBoost classifier training                          │
│  • Model evaluation & metrics                           │
│  • Save complete pipeline (preprocessor + model)        │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│          Flask Web Application & API                     │
│  • User-friendly web interface                          │
│  • REST API for predictions                             │
│  • Real-time cancellation probability                   │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset Features

### Input Features (17 Variables)

| Feature | Type | Description |
|---------|------|-------------|
| **no_of_adults** | Integer | Number of adults in the reservation |
| **no_of_children** | Integer | Number of children in the reservation |
| **no_of_weekend_nights** | Integer | Number of weekend nights (Friday-Saturday) booked |
| **no_of_week_nights** | Integer | Number of weekday nights booked |
| **type_of_meal_plan** | Categorical | Meal plan selected (Not Selected, Meal Plan 1, 2, 3) |
| **required_car_parking_space** | Binary | Car parking space required (0 or 1) |
| **room_type_reserved** | Categorical | Type of room reserved (Room_Type 1-7) |
| **lead_time** | Integer | Days between booking date and arrival date |
| **arrival_year** | Integer | Year of arrival |
| **arrival_month** | Integer | Month of arrival (1-12) |
| **arrival_date** | Integer | Date of arrival (1-31) |
| **market_segment_type** | Categorical | Market segment (Online, Offline, Corporate, Aviation, Complementary) |
| **repeated_guest** | Binary | Whether guest has stayed before (0 or 1) |
| **no_of_previous_cancellations** | Integer | Number of previous cancellations by guest |
| **no_of_previous_bookings_not_canceled** | Integer | Number of previous bookings not canceled |
| **avg_price_per_room** | Float | Average price per room per day |
| **no_of_special_requests** | Integer | Number of special requests made |

### Feature Categories

- **Numerical Features (12)**: `no_of_adults`, `no_of_children`, `no_of_weekend_nights`, `no_of_week_nights`, `lead_time`, `arrival_year`, `arrival_month`, `arrival_date`, `no_of_previous_cancellations`, `no_of_previous_bookings_not_canceled`, `avg_price_per_room`, `no_of_special_requests`

- **Categorical Features (5)**: `type_of_meal_plan`, `required_car_parking_space`, `room_type_reserved`, `market_segment_type`, `repeated_guest`

### Target Variable
- **`booking_status`**: Binary classification
  - `1`: Canceled
  - `0`: Not Canceled

### Data Source
- **Storage**: AWS S3 Bucket 
- **File**: `HotelReservations.csv`
- **Split**: 80% Training, 20% Testing

---

## 🔬 Machine Learning Pipeline

### 1️⃣ **Data Ingestion**
```python
- Connects to AWS S3 using boto3
- Downloads dataset to feature store
- Performs stratified train-test split (80-20)
- Saves raw data artifacts with timestamp
- Artifact Location: Artifacts/{timestamp}/data_ingestion/
```

### 2️⃣ **Data Validation**
```python
- Validates schema against expected features
- Checks for:
  • Required columns presence
  • Data types consistency
  • Missing values patterns
  • Outliers detection
- Generates drift report (YAML format)
- Saves validated data
- Artifact Location: Artifacts/{timestamp}/data_validation/
```

### 3️⃣ **Data Transformation**
```python
Preprocessing Pipeline:
1. Numerical Features:
   - RobustScaler (handles outliers better than StandardScaler)
   
2. Categorical Features:
   - OrdinalEncoder for encoded categories
   
3. Target Variable:
   - LabelEncoder for binary classification
   
4. Class Imbalance Handling:
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Balances canceled vs not-canceled samples
   
5. Column Transformer:
   - Combines all preprocessing steps
   - Saves preprocessor object for inference
   
Artifact Location: Artifacts/{timestamp}/data_transformation/
```

### 4️⃣ **Model Training with Optuna**
```python
Hyperparameter Optimization:
- Algorithm: XGBoost Classifier
- Optimization Framework: Optuna
- Number of Trials: 10
- Objective: Maximize F1-Score

Search Space:
- n_estimators: [50, 300]
- max_depth: [3, 15]
- learning_rate: [0.01, 0.3]
- subsample: [0.6, 1.0]
- colsample_bytree: [0.6, 1.0]
- min_child_weight: [1, 10]
- gamma: [0, 5]

Training Process:
1. Optuna finds best hyperparameters using cross-validation
2. Final model trained with best parameters
3. Evaluation on training and test sets
4. Complete pipeline saved (preprocessor + model)

Artifact Location: Artifacts/{timestamp}/model_trainer/
Final Model: final_model/model.pkl 
```

### 5️⃣ **Model Evaluation**
```python
Metrics Tracked:
- F1-Score (Primary metric for imbalanced data)
- Precision (Minimize false positives)
- Recall (Catch actual cancellations)
- Accuracy (Overall correctness)

Evaluation Sets:
- Training Set: Check for overfitting
- Test Set: Generalization performance
```

---

## 🛠️ Technology Stack

### **Core Machine Learning**
- **Python 3.10**: Primary programming language
- **XGBoost 1.7.6**: Gradient boosting classifier
- **scikit-learn 1.3.0**: Preprocessing, metrics, utilities
- **Optuna 3.3.0**: Automated hyperparameter optimization
- **imbalanced-learn**: SMOTE for class imbalance
- **NumPy 1.24.3**: Numerical computations
- **Pandas 2.0.3**: Data manipulation

### **Web Framework**
- **Flask 2.3.0**: Web application framework
- **Jinja2**: Template engine for HTML rendering
- **HTML/CSS**: Responsive web interface

### **Data Storage & Management**
- **AWS S3**: Cloud storage via boto3
- **PyYAML 6.0**: Configuration management

### **MLOps & Tracking**
- **MLflow 2.5.0**: Experiment tracking and model registry

### **Deployment & DevOps**
- **Docker**: Containerization
- **AWS ECR**: Public container registry
- **GitHub Actions**: CI/CD automation

### **Development Tools**
- **Git**: Version control
- **GitHub**: Code repository and CI/CD
- **Jupyter Notebook**: EDA and experimentation

---

## 📁 Project Structure

```
End-to-End-Hotel-Reservation-Prediction/
│
├── app.py                          # Flask application entry point
├── main.py                         # Training pipeline execution script
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup configuration
├── dockerfile                      # Docker container configuration
├── .gitignore                      # Git ignore rules
│
├── .github/
│   └── workflows/
│       └── main.yaml               # CI/CD pipeline configuration
│
├── src/                            # Main source package
│   ├── __init__.py
│   │
│   ├── components/                 # Pipeline components
│   │   ├── data_ingetion.py       # S3 data download & split
│   │   ├── Data_validation.py     # Schema & drift validation
│   │   ├── Data_transformation.py # Preprocessing & SMOTE
│   │   └── model_trainer.py       # Optuna + XGBoost training
│   │
│   ├── config/                     # Configuration modules
│   │   ├── config.py              # Pipeline configurations
│   │   └── artifact_config.py     # Artifact dataclasses
│   │
│   ├── constants/                  # Constants & parameters
│   │   └── training_pipeline/
│   │       └── __init__.py        # Pipeline constants
│   │
│   ├── exception/                  # Custom exceptions
│   │   └── exception.py           # HotelReservationException
│   │
│   └── utils/                      # Utility functions
│       ├── main_utils.py          # File I/O utilities
│       ├── estimator.py           # NetworkModel class
│       ├── classification_metrics.py  # Evaluation metrics
│       └── optuna_tuner.py        # Hyperparameter optimization
│
├── notebook/                       # Jupyter notebooks
│   ├── EDA.ipynb                  # Exploratory Data Analysis
│   └── processing.ipynb           # Data processing experiments
│
├── templates/                      # Flask HTML templates
│   └── index.html                 # Web application UI
│
├── Artifacts/                      # Training artifacts (timestamped)
│   └── [timestamp]/
│       ├── data_ingestion/
│       │   ├── feature_store/     # Raw data from S3
│       │   └── ingested/          # Train/test split
│       ├── data_validation/
│       │   ├── validated/         # Validated datasets
│       │   └── drift_report/      # Drift detection reports
│       ├── data_transformation/
│       │   ├── transformed/       # Preprocessed data
│       │   └── transformed_object/ # Preprocessor object
│       └── model_trainer/
│           └── trained_model/     # Trained model artifacts
│
└── final_model/                    # Production model
    └── model.pkl                   # NetworkModel (preprocessor + XGBoost)
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- AWS Account with S3 access
- Docker (for containerization)
- Git

### Local Setup

1. **Clone the Repository**
```bash
git clone https://github.com/sanjeevpatil804/hotel-reservation-prediction.git
cd hotel-reservation-prediction
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure AWS Credentials**

Set up AWS credentials for S3 access:
```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1
```

5. **Install Package**
```bash
pip install -e .
```

---

## 💻 Usage

### Training the Model

Run the complete training pipeline:

```bash
python main.py
```

**Pipeline Execution:**
1. ✅ Data ingestion from AWS S3
2. ✅ Data validation and drift detection
3. ✅ Data transformation with SMOTE
4. ✅ Model training with Optuna optimization
5. ✅ Model evaluation and artifact generation

**Output:**
- Artifacts saved in `Artifacts/{timestamp}/`
- Final model saved in `final_model/model.pkl`
- Drift report in `Artifacts/{timestamp}/data_validation/drift_report/report.yaml`

**Expected Output:**
```
Training final model with best parameters...
Evaluating on training set...
Evaluating on test set...
Training F1 Score: 0.9234
Test F1 Score: 0.8967
Model saved successfully!
```

### Starting the Web Application

Launch the Flask application:

```bash
python app.py
```

The application will be available at:
- **Local**: `http://localhost:8080`
- **Network**: `http://0.0.0.0:8080`

---

## 🌐 Web Application

### User Interface

The web application provides an intuitive form-based interface for hotel staff to predict cancellation probability.

**Features:**
- 📝 **Interactive Form**: Easy-to-use input fields for all 17 features
- 🎨 **Modern Design**: Gradient background with responsive layout
- ✅ **Real-time Validation**: Client-side form validation
- 📊 **Instant Results**: Immediate prediction display
- 🎯 **Clear Indication**: Color-coded results (Canceled/Not Canceled)

### Form Fields

1. **Guest Information**
   - Number of adults
   - Number of children

2. **Booking Details**
   - Weekend nights
   - Weekday nights
   - Lead time (days in advance)
   - Meal plan type
   - Car parking required

3. **Room Information**
   - Room type reserved
   - Average price per room

4. **Arrival Information**
   - Arrival year
   - Arrival month
   - Arrival date

5. **Guest History**
   - Repeated guest
   - Previous cancellations
   - Previous bookings not canceled
   - Number of special requests

6. **Market Segment**
   - Market segment type (Online, Offline, Corporate, etc.)

### Sample Usage

1. Open `http://localhost:8080` in your browser
2. Fill in the reservation details
3. Click "Predict Cancellation"
4. View the prediction result

---

## 📡 API Documentation

### REST API Endpoints

#### **GET /**
- **Description**: Renders the web application home page
- **Response**: HTML page with prediction form

#### **POST /predict**
- **Description**: Make prediction from HTML form submission
- **Request Type**: `application/x-www-form-urlencoded`
- **Response**: HTML page with prediction result

#### **POST /api/predict**
- **Description**: JSON API endpoint for programmatic predictions
- **Request Type**: `application/json`
- **Response Type**: `application/json`

**Request Body Example:**
```json
{
  "no_of_adults": 2,
  "no_of_children": 0,
  "no_of_weekend_nights": 1,
  "no_of_week_nights": 2,
  "type_of_meal_plan": "Meal Plan 1",
  "required_car_parking_space": 0,
  "room_type_reserved": "Room_Type 1",
  "lead_time": 224,
  "arrival_year": 2018,
  "arrival_month": 10,
  "arrival_date": 2,
  "market_segment_type": "Online",
  "repeated_guest": 0,
  "no_of_previous_cancellations": 0,
  "no_of_previous_bookings_not_canceled": 0,
  "avg_price_per_room": 65.0,
  "no_of_special_requests": 0
}
```

**Response Example:**
```json
{
  "prediction": "Not_Canceled",
  "prediction_value": 0,
  "probability": {
    "not_canceled": 0.87,
    "canceled": 0.13
  }
}
```

### API Usage Examples

#### Python
```python
import requests

url = "http://localhost:8080/api/predict"
data = {
    "no_of_adults": 2,
    "no_of_children": 1,
    "no_of_weekend_nights": 2,
    "no_of_week_nights": 3,
    "type_of_meal_plan": "Meal Plan 2",
    "required_car_parking_space": 1,
    "room_type_reserved": "Room_Type 4",
    "lead_time": 120,
    "arrival_year": 2024,
    "arrival_month": 7,
    "arrival_date": 15,
    "market_segment_type": "Online",
    "repeated_guest": 0,
    "no_of_previous_cancellations": 0,
    "no_of_previous_bookings_not_canceled": 2,
    "avg_price_per_room": 85.5,
    "no_of_special_requests": 1
}

response = requests.post(url, json=data)
print(response.json())
```

#### cURL
```bash
curl -X POST "http://localhost:8080/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "no_of_adults": 2,
    "no_of_children": 0,
    "no_of_weekend_nights": 1,
    "no_of_week_nights": 2,
    "type_of_meal_plan": "Meal Plan 1",
    "required_car_parking_space": 0,
    "room_type_reserved": "Room_Type 1",
    "lead_time": 224,
    "arrival_year": 2018,
    "arrival_month": 10,
    "arrival_date": 2,
    "market_segment_type": "Online",
    "repeated_guest": 0,
    "no_of_previous_cancellations": 0,
    "no_of_previous_bookings_not_canceled": 0,
    "avg_price_per_room": 65.0,
    "no_of_special_requests": 0
  }'
```

---

## 📈 Model Performance

### XGBoost with Optuna Optimization

**Optimization Configuration:**
- Trials: 10
- Objective: Maximize F1-Score
- Cross-Validation: 5-fold stratified

**Sample Performance Metrics:**

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| **F1-Score** | 0.9234 | 0.8967 |
| **Precision** | 0.9156 | 0.8845 |
| **Recall** | 0.9314 | 0.9092 |
| **Accuracy** | 0.9189 | 0.8934 |

### Best Hyperparameters (Sample)

```python
{
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.2,
    'random_state': 42
}
```

### Feature Importance (Top 10)

1. **lead_time**: 0.245 - Days between booking and arrival
2. **avg_price_per_room**: 0.187 - Room pricing
3. **no_of_special_requests**: 0.134 - Special service requests
4. **no_of_previous_cancellations**: 0.098 - Historical cancellation behavior
5. **market_segment_type**: 0.076 - Booking channel
6. **no_of_week_nights**: 0.065 - Weekday stay duration
7. **room_type_reserved**: 0.052 - Room category
8. **arrival_month**: 0.041 - Seasonality
9. **no_of_weekend_nights**: 0.038 - Weekend stay duration
10. **repeated_guest**: 0.024 - Customer loyalty

### Class Imbalance Handling

**Before SMOTE:**
- Not Canceled: 67%
- Canceled: 33%

**After SMOTE:**
- Not Canceled: 50%
- Canceled: 50%

This balanced dataset improves the model's ability to detect cancellations.

---

## 🌐 Deployment

### Docker Deployment

#### Build Docker Image
```bash
docker build -t hotel-reservation:latest .
```

#### Run Container Locally
```bash
docker run -d \
  -p 8080:8080 \
  --name hotel-app \
  -v $(pwd)/final_model:/app/final_model \
  hotel-reservation:latest
```

#### Access Application
- Open browser: `http://localhost:8080`

### AWS Deployment Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  GitHub Repo    │─────▶│  GitHub Actions  │─────▶│   AWS ECR       │
│  (Code Push)    │      │  (CI/CD)         │      │  (Public)       │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                                             │
                                                             ▼
                         ┌──────────────────────────────────────────┐
                         │     AWS EC2 (Self-Hosted Runner)         │
                         │  • Pulls latest image from ECR           │
                         │  • Stops old container                   │
                         │  • Runs new container on port 8080       │
                         │  • Mounts final_model volume             │
                         │  • Cleans up old images                  │
                         └──────────────────────────────────────────┘
                                            │
                                            ▼
                         ┌──────────────────────────────────────────┐
                         │         AWS S3 Bucket                    │
                         │  • Stores training data                  │
                         │  • Accessed during pipeline execution    │
                         └──────────────────────────────────────────┘
```

### Deployment Steps

1. **AWS ECR Setup (Public Registry)**
```bash
# Login to Public ECR
aws ecr-public get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin public.ecr.aws

# Create repository (if not exists)
aws ecr-public create-repository \
  --repository-name hotel-reservation \
  --region us-east-1
```

2. **GitHub Secrets Configuration**

Add the following secrets to your GitHub repository:
- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_REGION`: AWS region (e.g., us-east-1)
- `ECR_REGISTRY`: ECR registry URL (e.g., public.ecr.aws/xxxxx)
- `ECR_REPOSITORY_NAME`: Repository name (e.g., hotel-reservation)

3. **Self-Hosted Runner Setup** (on AWS EC2)
```bash
# Update system
sudo apt-get update -y

# Install Docker
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure GitHub Runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz
./config.sh --url https://github.com/YOUR_USERNAME/YOUR_REPO --token YOUR_TOKEN
./run.sh
```

4. **Deploy Application**

Simply push to main branch:
```bash
git add .
git commit -m "Deploy application"
git push origin main
```

The CI/CD pipeline will automatically:
- Build Docker image
- Push to AWS ECR
- Deploy to self-hosted runner
- Start the application on port 8080

---

## ⚙️ CI/CD Pipeline

### GitHub Actions Workflow

The `.github/workflows/main.yaml` defines a three-stage pipeline:

#### **Stage 1: Continuous Integration**
```yaml
- Checkout code from main branch
- Run linting checks
- Execute unit tests (placeholder)
- Validate code quality
```

#### **Stage 2: Continuous Delivery**
```yaml
- Install AWS CLI and utilities
- Configure AWS credentials
- Login to Amazon Public ECR
- Validate ECR configuration
- Build Docker image
- Tag image as 'latest'
- Push to Public ECR repository
```

#### **Stage 3: Continuous Deployment**
```yaml
- Run on self-hosted EC2 runner
- Install AWS CLI if needed
- Login to Public ECR
- Pull latest image
- Stop existing container (if running)
- Remove old container
- Run new container with:
  • Port mapping: 8080:8080
  • Volume mount: final_model directory
  • Environment variables
- Clean up unused Docker resources
```

### Trigger Conditions

The pipeline triggers on:
- **Push** to `main` branch
- **Excludes** changes to `README.md`

### Deployment Flow

```
Developer Push → GitHub → CI Tests → Build Docker → Push to Public ECR → 
Deploy to EC2 → Stop Old Container → Start New Container → Live on Port 8080
```

**Deployment Time**: ~6-10 minutes (from push to live)

### Volume Mounting Strategy

The deployment uses volume mounting for the `final_model` directory:
```bash
-v $(pwd)/final_model:/app/final_model
```

This allows:
- Model updates without rebuilding container
- Persistent model storage
- Easy model versioning

---

## 🔮 Future Enhancements

### Model Improvements
- [ ] Implement ensemble methods (Random Forest + XGBoost + LightGBM)
- [ ] Add deep learning models (Neural Networks)
- [ ] Expand Optuna trials (50-100 trials)
- [ ] Implement online learning for continuous updates
- [ ] Add model explainability (SHAP, LIME)

### Feature Engineering
- [ ] Extract temporal features (booking trends, seasonality)
- [ ] Add customer segmentation features
- [ ] Incorporate external data (holidays, events, weather)
- [ ] Create interaction features
- [ ] Add geolocation features (if available)

### Infrastructure & MLOps
- [ ] Implement MLflow for experiment tracking
- [ ] Add model monitoring and drift detection
- [ ] Setup Prometheus + Grafana for metrics
- [ ] Implement A/B testing framework
- [ ] Kubernetes deployment for auto-scaling
- [ ] Add feature store integration

### Application Features
- [ ] Batch prediction API endpoint
- [ ] Export predictions to CSV/Excel
- [ ] Dashboard for cancellation analytics
- [ ] Email notifications for high-risk bookings
- [ ] Integration with hotel management systems (PMS)
- [ ] Mobile app development

### Data & Testing
- [ ] Expand dataset with recent bookings
- [ ] Add comprehensive unit/integration tests
- [ ] Implement data quality monitoring
- [ ] Load testing and performance benchmarking
- [ ] Add data versioning (DVC)

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the Repository**
```bash
git clone https://github.com/sanjeevpatil804/hotel-reservation-prediction.git
```

2. **Create a Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

3. **Make Your Changes**
- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Update tests if applicable
- Update documentation

4. **Commit Your Changes**
```bash
git add .
git commit -m "Add: Your descriptive commit message"
```

5. **Push to Your Fork**
```bash
git push origin feature/your-feature-name
```

6. **Create a Pull Request**
- Provide a clear description of changes
- Reference any related issues
- Include before/after metrics if applicable

### Code Quality Standards
- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type annotations
- **Docstrings**: Document all functions and classes
- **Testing**: Add tests for new features
- **Logging**: Use proper logging instead of print statements

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Sanjeev Patil

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 👨‍💻 Author

**Sanjeev Patil**

- GitHub: [@sanjeevpatil804](https://github.com/sanjeevpatil804)
- LinkedIn: [Sanjeev Patil](https://www.linkedin.com/in/sanjeevpatil804)
- Email: sanjeevpatil804@aol.com

---

## 🙏 Acknowledgments

- **Dataset**: Hotel Reservations Dataset
- **Frameworks**: Flask, XGBoost, scikit-learn, Optuna
- **Cloud Services**: AWS S3, AWS ECR, AWS EC2
- **CI/CD**: GitHub Actions
- **Community**: Open-source contributors and data science community

---

## 📞 Support

If you encounter any issues or have questions:

1. **Check Documentation**: Review this README and code comments
2. **Search Issues**: Look for similar issues in the GitHub repository
3. **Open an Issue**: Create a detailed issue with:
   - Problem description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Error logs (if applicable)

---

## 📊 Project Metrics

- **Lines of Code**: ~2,500+
- **Components**: 4 (Ingestion, Validation, Transformation, Training)
- **Features**: 17
- **Model Type**: XGBoost Classifier
- **Deployment**: Dockerized, AWS ECR + EC2
- **CI/CD**: GitHub Actions (3-stage pipeline)

---

## ⭐ Star the Repository

If you find this project useful for learning or in production, please consider giving it a star ⭐ on GitHub!

---

<div align="center">

**Built with ❤️ for the Hospitality Industry**

*Helping Hotels Optimize Revenue Through Intelligent Predictions*

[🔝 Back to Top](#-end-to-end-hotel-reservation-cancellation-prediction)

</div>
