# Census Income Classification & Customer Segmentation
**Candidate:** Abhishek Sharma  
**Role:** CCB Risk Program Senior Associate (Req: 210689528)  
**Date:** April 17th, 2026  

---

## 1. Project Overview
This repository contains a comprehensive data science solution developed for the JPMC CCB Risk Program take-home project. The objective was to analyze 1994-1995 U.S. Census Bureau data to deliver two primary business requirements:

1.  **Income Classification:** A predictive model (XGBoost) to identify individuals earning >$50,000 annually.
2.  **Customer Segmentation:** A clustering framework (K-Means) to categorize the population into actionable marketing personas.

---

## 2. Repository Structure
* `code.py`: The main execution script containing the end-to-end ML pipeline.
* `Final_JPMC_Report_Abhishek_Sharma.pdf`: A formal business report translating technical metrics into marketing strategy.
* `requirements.txt`: Python dependencies required to replicate the environment.
* `census-bureau.data`: The primary dataset.
* `census-bureau.columns`: Feature headers and metadata.
* `.gitignore`: Configuration to prevent unnecessary files (like .venv) from being tracked.

---

## 3. Environment Setup
The project was developed in a **Python 3.12** environment on macOS. To ensure reproducibility, follow these steps:

### **For macOS / Linux**

### **Step 1: Create a Virtual Environment**
```bash
python3 -m venv .venv
```

### **Step 2: Activate the Environment**
```bash
source .venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Technical Requirements (macOS / Apple Silicon)**

This project utilizes XGBoost, which relies on the OpenMP (libomp) runtime for multi-threaded processing on macOS. If you encounter a library loading error on Apple Silicon, install the runtime via Homebrew

```bash
brew install libomp
```

### **For Windows**

### **Step 1: Create a Virtual Environment**
```bash
python -m venv .venv
```

### **Step 2: Activate the Environment**
```bash
.venv\Scripts\activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

## 4. Execution

To execute the classification and segmentation pipelines, ensure the data files are in the root directory and run:

```bash
python code.py
```

## 5. Model Methodology & Risk Considerations

* **Data Integrity:** Identified and preserved "Not in universe" entries as valid categorical signals rather than assuming missing data. This captures significant life-stage patterns (children/retirees) that are highly predictive of income.

* **Imbalance Management:** Addressed the 25:1 class imbalance using Stratified Sampling and Cost-Sensitive Learning (scale_pos_weight).

* **Scalability:** Built using Scikit-Learn Pipelines and ColumnTransformers to ensure a modular, production-ready codebase that prevents data leakage during preprocessing.

