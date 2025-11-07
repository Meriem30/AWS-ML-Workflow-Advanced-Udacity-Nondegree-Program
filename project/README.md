# 🚲 vs 🏍️ SageMaker Image Classification Project

**AWS Machine Learning Workflow - Udacity Advanced ML Program**


## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Project Objectives](#-project-objectives)
- [Business Problem](#-business-problem)
- [Solution Architecture](#-solution-architecture)
- [Technologies Used](#-technologies-used)
- [Dataset](#-dataset)
- [Implementation Details](#-implementation-details)

---
## 🧠 Project Overview

This project demonstrates the **end-to-end development and deployment of a scalable, event-driven machine learning application on AWS**.
Built entirely with **AWS SageMaker, Lambda, and Step Functions**, it delivers a binary image classification system that distinguishes between bicycles and motorcycles using modern MLOps practices and serverless architecture.
The project was developed as part of the **AWS Machine Learning Fundamentals Nanodegree Program** on Udacity.

---
## 🎯 Project Objectives

>🏗️ Build
>
>Develop and deploy a production-ready deep learning model on Amazon SageMaker, applying best practices in tuning, versioning, and reproducibility.
>
>🔗 Integrate
> 
>Connect the model with AWS Lambda functions for data preprocessing, inference, and automation within a serverless architecture.
>
>🔄 Orchestrate
>
>Automate the end-to-end ML workflow using AWS Step Functions, enabling scalable, event-driven model operations.
>
>🚀 Deploy & Monitor
>
>Implement serverless inference with Amazon SageMaker Model Monitor for continuous performance tracking, data drift detection, and lifecycle management.


---

## 💼 Business Problem

**Scones Unlimited** needs to optimize their delivery operations by automatically routing delivery professionals based on their vehicle type:
- **Bicyclists** → Assigned to nearby orders
- **Motorcyclists** → Assigned to distant orders


### 🚧 Challenge

Manual vehicle verification is a time-consuming and error-prone process.
By automating image classification, this project aims to streamline vehicle identification and support their team on broader applications such as:

 - Detecting bicycles and motorcycles in roadway imagery or surveillance feeds

 - Enhancing traffic monitoring and reporting workflows

 - Supporting automated quality checks and visual inspections in real-world operations

This challenge demonstrates how computer vision and AWS-based automation can reduce human effort, improve data-driven decision-making

---


## 🏗️ Solution Architecture

The project implements a **complete MLOps pipeline on AWS**, integrating data processing, training, deployment, and monitoring into an automated, serverless workflow.

```text
                                                                ┌──────────────────────────────┐
                                                                │         Data Sources         │
                                                                │    (Images: Motorcycles &    │
                                                                │           Bicycles)          │
                                                                └──────────────┬───────────────┘
                                                                               │
                                                                               ▼
                                                                ┌──────────────────────────────┐
                                                                │          Amazon S3           │
                                                                │   (Stores raw & processed    │
                                                                │          datasets)           │
                                                                └──────────────┬───────────────┘
                                                                               │
                                                                               ▼
                                                                ┌──────────────────────────────┐
                                                                │          AWS Lambda          │
                                                                │        (Preprocessing)       │
                                                                │  - Cleans & structures data  │
                                                                │  - Triggers training workflow│
                                                                └──────────────┬───────────────┘
                                                                               │
                                                                               ▼
                                                                ┌──────────────────────────────┐
                                                                │      AWS Step Functions      │
                                                                │  (Orchestrates ML Pipeline)  │
                                                                └──────────────┬───────────────┘
                                                                               │
                                                         ┌─────────────────────┼─────────────────────┐
                                                         ▼                     ▼                     ▼
                                                ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
                                                │    SageMaker    │  │    SageMaker    │  │    CloudWatch   │
                                                │    Training     │  │    Evaluation   │  │      Logs       │
                                                │  - Trains image │  │  - Validates    │  │  - Monitors     │
                                                │    model        │  │    accuracy     │  │    metrics      │
                                                └─────────┬───────┘  └─────────────────┘  └─────────────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────────────────┐
                                                │     SageMaker Deployment     │
                                                │          Endpoint            │
                                                │  (Real-time inference API)   │
                                                └──────────────┬───────────────┘
                                                               │
                                                               ▼
                                                ┌──────────────────────────────┐
                                                │         AWS Lambda           │
                                                │      (Inference Handler)     │
                                                │     - Handles prediction     │
                                                │      requests                │
                                                │     - Returns classification │
                                                │      results                 │
                                                └──────────────┬───────────────┘
                                                               │
                                                               ▼
                                                ┌──────────────────────────────┐
                                                │    SageMaker Model Monitor   │
                                                │    - Detects data drift      │
                                                │    - Triggers retraining     │
                                                └──────────────────────────────┘

```

### 🧩 Workflow Components

1. **Data Preparation Layer**
   - ETL operations on CIFAR-100 dataset
   - Image preprocessing and augmentation
   - Train/Test split with stratification

2. **Model Training Layer**
   - SageMaker training jobs with hyperparameter optimization
   - Transfer learning using pre-trained ImageNet models
   - Model evaluation and validation

3. **Deployment Layer**
   - SageMaker real-time inference endpoints with data capture
   - Auto-scaling configuration
   - Model versioning and rollback capabilities

4. **Inference Pipeline**
   - Serverless workflow with confidence thresholding
   - Lambda function for image preprocessing
   - Lambda function for inference invocation
   - Lambda function for confidence threshold filtering

5. **Orchestration Layer**
   - Step Functions state machine
   - Error handling and retry logic
   - Monitoring and logging

---

## 🛠️ Technologies Used

### 🔹 AWS Services
- **Amazon S3**: Data storage and model artifacts
- **Amazon SageMaker**: Model training, tuning, and monitoring
- **AWS Lambda**: Serverless compute for inference and event-driven triggers  
- **AWS Step Functions**: Workflow orchestration and automation 
- **IAM**: Security and access management 
- **Amazon CloudWatch**: Logging, monitoring, and metric visualization  

### 🔹 ML/Data Science Stack
- **Python 3.11+**: Primary programming language
- **PyTorch/TensorFlow**: Deep learning frameworks
- **scikit-learn**: ML utilities and metrics
- **NumPy/Pandas**: Data manipulation
- **Boto3**: AWS SDK for Python


## 📊  Dataset
- **CIFAR-100**: filtered to contain bicycle and motorcycle images
- **Classes**: 2 (bicycle, motorcycle)
- **Training Images**: ~1,000 for both target classes
- **Test Images**: ~200 for both target classes
- **Image Size**: 32x32x3 RGB
- **Format**: PNG files uploaded to S3

### Data Distribution
| Class | Training | Test | Label |
|-------|----------|------|-------|
| Bicycle | ~500 | ~100 | 0 |
| Motorcycle | ~500 | ~100 | 1 |

---


## 🔨 Implementation Details

### Phase 1: Data Preparation & ETL
```python
# Extract bicycle and motorcycle images from CIFAR-100
# Transform images to appropriate format
# Load to S3 bucket for SageMaker access
- Downloaded & filtered CIFAR-100 dataset
- Extracted bicycle & motorcycle (classes 8, 48) images
- Organized data into train/test directories
- Uploaded to S3: proper directory structure

```
### Phase 2: Model Training

**Training Configuration:**

- **Algorithm**: Image Classification (built-in `MXNet-based` SageMaker algorithm)

- **Instance Type**: `ml.p3.2xlarge` (GPU-accelerated)

- **Hyperparameters**:
```yaml
  image_shape: "3,32,32"
  num_classes: 2
  Learning rate: 0.1
  Batch size: 32
  Epochs: 30
  Optimizer: SGD
  Image augmentation: Disabled
```

### 3. Model Deployment

**Endpoint Configuration**:

- **Instance**: `ml.m5.xlarge`
- **Initial instance count**: 1
- **Auto Scaling**: Configured for production load
- **Data Capture**: 100% sampling for monitoring

### 4. Monitoring Implementation

#### Statistical Monitoring
- **Baseline Generation**: From test dataset predictions
- **Schedule**: Hourly monitoring jobs
- **Metrics**: 
  - Prediction confidence distribution
  - Input data quality checks
  - Model performance drift detection

#### Data Capture Analysis
```JSON
// Captured data structure: 
{
  "captureData": {
    "endpointInput": "<base64_image>",
    "endpointOutput": "[confidence_scores]"
  },
  "eventMetadata": { 
    "eventId": "...",
    "inferenceTime": "2025-09-25T11:38:12Z"
  }
}
```