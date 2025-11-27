# Multiple Linear Regression via QR Decomposition

## 📊 Project Overview

The goal is to determine the optimal coefficients that minimize the error between predicted and actual house prices based on specific property features.

### The Features
The model predicts the **Price** based on four independent variables:

| Feature | Variable Name | Coefficient Symbol | Description |
| :--- | :--- | :--- | :--- |
| **Gross Living Area** | `square_meters` | **$\alpha$ (Alpha)** | Total habitable space. |
| **Room Count** | `num_rooms` | **$\beta$ (Beta)** | Total number of functional rooms. |
| **Location** | `loc_score` | **$\gamma$ (Gamma)** | A score from 1-5 indicating desirability. |
| **Age** | `age_years` | **$\delta$ (Delta)** | Age of the property in years. |
---

## 📈 Visualizations
**Residual Plot:**  
Plots Residuals vs. Predicted Price. Includes a "Perfect Fit" line to visually check for bias.

**Partial Dependence Plots:**  
A 2x2 grid showing the marginal effect of each feature on the house price.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Jupyter Notebook

### Installation

1.  **Clone the repository** (or download the files):
    ```bash
    git clone https://github.com/mattia-3rne/Multiple-Linear-Regression-via-QR-Decomposition.git
    ```

2.  **Install dependencies**:
    This project requires `numpy`, `pandas`, `matplotlib`, and `scipy`.
    ```bash
    pip install -r requirements.txt
    ```

### Running the Analysis

1.  Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

2.  Open the main notebook file (`main.ipynb`).

3.  **Run All Cells**:
    The notebook is self-contained. It will:
    * Automatically generate the synthetic dataset.
    * Construct the matrices.
    * Perform the QR Decomposition.
    * Display the solved coefficients.
    * Generate the visualization plots.

---

## 📂 Project Structure

* `dataset_generation.py`: Script to generate the data.
* `requirements.txt`: Python package dependencies.
* `dataset.csv`: The generated house dataset.
* `main.ipynb`: The main jupyter notebook.

* `README.md`: Project documentation.
