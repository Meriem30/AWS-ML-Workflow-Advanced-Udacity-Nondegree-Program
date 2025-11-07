# 🚲 vs 🏍️ SageMaker Image Classification Project

**AWS Machine Learning Workflow - Udacity Advanced ML Program**


## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Project Objectives](#-project-objectives)
- [Business Problem](#-business-problem)

---
## 🧠 Project Overview

This project demonstrates the **end-to-end development and deployment of a scalable, event-driven machine learning application on AWS**.
Built entirely with **AWS SageMaker, Lambda, and Step Functions**, it delivers a binary image classification system that distinguishes between bicycles and motorcycles using modern MLOps practices and serverless architecture.
The project was developed as part of the **AWS Machine Learning Fundamentals Nanodegree Program** on Udacity.

---
## 🎯 Project Objectives

🏗️ Build

Develop and deploy a production-ready deep learning model on Amazon SageMaker, applying best practices in tuning, versioning, and reproducibility.

🔗 Integrate

Connect the model with AWS Lambda functions for data preprocessing, inference, and automation within a serverless architecture.

🔄 Orchestrate

Automate the end-to-end ML workflow using AWS Step Functions, enabling scalable, event-driven model operations.

🚀 Deploy & Monitor

Implement serverless inference with Amazon SageMaker Model Monitor for continuous performance tracking, data drift detection, and lifecycle management.


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

---