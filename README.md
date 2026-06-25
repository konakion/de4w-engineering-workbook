DE4W Live Activity Recognition App

Project Goal

In this project, we build a complete end-to-end machine learning application for human activity recognition using smartphone accelerometer data.

The goal is to demonstrate the entire machine learning pipeline:

Sensor Data → Feature Engineering → Machine Learning → Deployment → Live Prediction

Students will train a machine learning model on the WISDM dataset, deploy the model as a web service, and finally perform live activity recognition using real-time smartphone sensor data streamed from phyphox.

Learning Objectives

After completing this project, students should be able to:

* Load and explore raw wearable sensor data
* Understand windowing and segmentation
* Extract meaningful features from time-series sensor data
* Train and evaluate machine learning models
* Understand subject-wise evaluation and generalization
* Export and reuse trained models
* Build prediction APIs using FastAPI
* Create dashboards using Streamlit
* Integrate live smartphone sensor streams
* Deploy a complete end-to-end machine learning pipeline

Project Pipeline

Smartphone Sensor Data

↓

Windowing

↓

Feature Engineering

↓

Random Forest Model

↓

Model Export

↓

FastAPI Service

↓

Streamlit Dashboard

↓

Live Activity Prediction

Dataset

Dataset:

wisdm-dataset/raw/phone/accel

Target activities:

* walking
* sitting
* standing

Technologies

* Python
* uv
* pandas
* NumPy
* scikit-learn
* FastAPI
* Streamlit
* phyphox

Evaluation

Two evaluation strategies are compared:

Random Split

Training and test data may contain samples from the same subjects.

Subject-wise Split

Training and test data contain different subjects.

This better reflects real-world deployment scenarios.

Live Demonstration

The system performs live activity recognition using smartphone accelerometer data streamed from phyphox.

Pipeline:

phyphox

↓

LiveBuffer

↓

Feature Extraction

↓

Random Forest

↓

Prediction

↓

Dashboard

Important Note

For live demonstrations, the smartphone and laptop must be connected to the same local network.

Some university WiFi networks (e.g. eduroam) may block direct device-to-device communication.

Recommended:

* Smartphone hotspot
* Local WiFi network
* Travel router

Expected Outcome

Students build a complete machine learning application connecting:

Sensors

↓

Data Engineering

↓

Machine Learning

↓

Deployment

↓

Live Prediction