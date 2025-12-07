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

## 🧮 Mathematical Methodology

This project moves beyond standard algebraic solutions by employing **QR Decomposition**, a cornerstone technique in numerical linear algebra. This method is specifically chosen for its numerical stability compared to the standard Normal Equations ($A^TA\mathbf{x}=A^T\mathbf{b}$), which can be prone to rounding errors when the feature matrix is ill-conditioned.

### 1. Matrix Formulation
We formulate the regression problem as an overdetermined system of linear equations $A \mathbf{x} = \mathbf{b}$.

**The Design Matrix ($A$):**
Constructed by augmenting the feature vectors with a column of ones (the intercept).

$$
A = \begin{bmatrix}
1 & \text{area}_1 & \text{rooms}_1 & \text{loc}_1 & \text{age}_1 \\
1 & \text{area}_2 & \text{rooms}_2 & \text{loc}_2 & \text{age}_2 \\
\vdots & \vdots & \vdots & \vdots & \vdots \\
1 & \text{area}_{100} & \text{rooms}_{100} & \text{loc}_{100} & \text{age}_{100}
\end{bmatrix} \in \mathbb{R}^{100 \times 5}
$$

**The Target Vector ($\mathbf{b}$):**
Contains the observed prices.

$$
\mathbf{b} = \begin{bmatrix} \text{price}_1 \\ \text{price}_2 \\ \vdots \\ \text{price}_{100} \end{bmatrix} \in \mathbb{R}^{100}
$$

**The Parameter Vector ($\mathbf{x}$):**
The coefficients we need to solve for.

$$
\mathbf{x} = \begin{bmatrix} \theta \\ \alpha \\ \beta \\ \gamma \\ \delta \end{bmatrix} \in \mathbb{R}^{5}
$$

### 2. QR Decomposition Structure
We decompose the design matrix $A$ into two specific components using the **Economic QR Decomposition**:

$$A = Q R$$

1.  **Q (Orthogonal Matrix):** An $m \times n$ ($100 \times 5$) matrix with orthonormal columns.
    * *Orthonormality Property:* The columns $\mathbf{q}_i$ satisfy the condition $\mathbf{q}_i^T \mathbf{q}_j = 1$ if $i=j$ and $0$ if $i \neq j$.
    * In matrix notation, this results in the Identity matrix: $Q^T Q = I_n$.
2.  **R (Upper Triangular Matrix):** An $n \times n$ ($5 \times 5$) matrix where all entries below the main diagonal are zero.

$$
Q = \begin{bmatrix}
| & | & & | \\
\mathbf{q}_1 & \mathbf{q}_2 & \dots & \mathbf{q}_5 \\
| & | & & |
\end{bmatrix}, \quad
R = \begin{bmatrix}
r_{11} & r_{12} & r_{13} & r_{14} & r_{15} \\
0 & r_{22} & r_{23} & r_{24} & r_{25} \\
0 & 0 & r_{33} & r_{34} & r_{35} \\
0 & 0 & 0 & r_{44} & r_{45} \\
0 & 0 & 0 & 0 & r_{55}
\end{bmatrix}
$$

### 3. Why Q is easily inverted
The computational efficiency of QR decomposition relies heavily on the orthogonality of $Q$. In general linear algebra, finding the inverse of a matrix $M^{-1}$ is computationally expensive $(O(n^3))$ and numerically unstable.

However, for an orthogonal matrix, **the transpose is the inverse**:
$$Q^{-1} = Q^T$$

**Why?**
Since the columns of $Q$ are orthonormal, multiplying $Q$ by its transpose results in the Identity matrix:

$$Q^T Q = I$$

This unique property allows us to "invert" or move $Q$ to the other side of an algebraic equation simply by transposing it ($Q^T$), an operation that costs almost nothing computationally compared to Gaussian elimination.

### 4. The Solution Derivation
We start with the Least Squares objective, which is equivalent to solving the Normal Equations:

$$A^T A \mathbf{x} = A^T \mathbf{b}$$

Substitute $A = Q R$ into the equation:

$$(Q R)^T (Q R) \mathbf{x} = (Q R)^T \mathbf{b}$$

Distribute the transpose operation (remembering that $(AB)^T = B^T A^T$):

$$R^T Q^T Q R \mathbf{x} = R^T Q^T \mathbf{b}$$

Apply the orthogonality property ($Q^T Q = I$) to simplify the left side:

$$R^T (I) R \mathbf{x} = R^T Q^T \mathbf{b}$$

$$R^T R \mathbf{x} = R^T Q^T \mathbf{b}$$

Assuming $A$ has full rank, $R$ is invertible. We multiply both sides by $(R^T)^{-1}$ to isolate the remaining terms:

$$\mathbf{R \mathbf{x} = Q^T \mathbf{b}}$$

### 5. Solving via Back-Substitution
The system has now been reduced to:

$$R \mathbf{x} = \mathbf{y}, \quad \text{where } \mathbf{y} = Q^T \mathbf{b}$$

Because $R$ is **Upper Triangular**, we do not need complex inversion algorithms to find $\mathbf{x}$. We use **Back-Substitution**:
1.  Solve for the last variable: $x_5 = y_5 / r_{55}$
2.  Substitute $x_5$ into the row above to solve for $x_4$.
3.  Continue upwards until all coefficients are found.

This final step is computationally cheap ($O(n^2)$), making the entire process highly efficient.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Jupyter Notebook

### Installation

1.  **Clone the repository** (or download the files):
    ```bash
    git clone https://github.com/mattia-3rne/multiple-linear-regression-via-qr-decomposition.git
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

* `main.ipynb`: The main jupyter notebook.
* `dataset_generation.py`: Script to generate the data.
* `requirements.txt`: Python package dependencies.
* `README.md`: Project documentation.




