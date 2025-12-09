# Multiple Linear Regression via QR Decomposition

## 📊 Project Overview

The goal of this project is to determine the optimal coefficients that minimize the error between predicted and actual house prices based on specific property features.

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

This project moves beyond standard algebraic solutions by employing **QR Decomposition**, a cornerstone technique in numerical linear algebra. This method is specifically chosen for its numerical stability compared to the standard Normal Equations, which can be prone to rounding errors when the feature matrix is ill-conditioned.

### The Model
We model the relationship as a linear combination of features:

$$
y = \theta + \alpha x_1 + \beta x_2 + \gamma x_3 + \delta x_4 + \epsilon
$$

Our goal is to determine the optimal parameter vector $\mathbf{x} = [\theta, \alpha, \beta, \gamma, \delta]^T$ that minimizes the sum of squared errors:

$$\sum \epsilon^2 = \| \mathbf{b} - \mathbf{A}\mathbf{x} \|^2$$


### 1. Matrix Formulation
We formulate the regression problem as an overdetermined system of linear equations $A \mathbf{x} = \mathbf{b}$.

**The Design Matrix:**
Constructed by augmenting the feature vectors with a column of ones for the intercept.

$$
A = \begin{bmatrix}
1 & \text{area}_1 & \text{rooms}_1 & \text{loc}_1 & \text{age}_1 \\
1 & \text{area}_2 & \text{rooms}_2 & \text{loc}_2 & \text{age}_2 \\
\vdots & \vdots & \vdots & \vdots & \vdots \\
1 & \text{area}_{m} & \text{rooms}_{m} & \text{loc}_{m} & \text{age}_{m}
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$

**The Target Vector:**
Contains the observed prices.

$$
\mathbf{b} = \begin{bmatrix} \text{price}_1 \\ \text{price}_2 \\ \vdots \\ \text{price}_{m} \end{bmatrix} \in \mathbb{R}^{m}
$$

**The Parameter Vector:**
The coefficients we need to solve for.

$$
\mathbf{x} = \begin{bmatrix} \theta \\ \alpha \\ \beta \\ \gamma \\ \delta \end{bmatrix} \in \mathbb{R}^{n}
$$

### 2. QR Decomposition Structure
We decompose the design matrix $A$ into two specific components using **QR Decomposition**:

$$A = Q R$$

1.  **Orthogonal Matrix Q:** An $m \times n$ matrix with orthonormal columns.
2.  **Upper Triangular Matrix R:** An $n \times n$ matrix where all entries below the main diagonal are zero.

$$
Q = \begin{bmatrix}
| & | & & | \\
\mathbf{q}_1 & \mathbf{q}_2 & \dots & \mathbf{q}_n \\
| & | & & |
\end{bmatrix}, \quad
R = \begin{bmatrix}
r_{11} & r_{12} & \dots & r_{1n} \\
0 & r_{22} & \dots & r_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & r_{nn}
\end{bmatrix}
$$

### 3. Why Q is easily inverted
The computational efficiency of QR decomposition relies heavily on the orthogonality of $Q$. In general linear algebra, finding the inverse of a matrix $M^{-1}$ is computationally expensive with $O(n^3)$  and numerically unstable.

However, for an orthogonal matrix, the transpose is the inverse:

$$Q^{-1} = Q^T$$

Since the columns of $Q$ are orthonormal, multiplying $Q$ by its transpose results in the Identity matrix:

$$Q^T Q = I$$

This unique property allows us to move $Q$ to the other side of an algebraic equation simply by transposing it, an operation that costs almost nothing computationally compared to Gaussian elimination.

### 4. The Solution Derivation
We start with the Least Squares objective, which is equivalent to solving the Normal Equations:

$$A^T A \mathbf{x} = A^T \mathbf{b}$$

Substitute $A = Q R$ into the equation:

$$(Q R)^T (Q R) \mathbf{x} = (Q R)^T \mathbf{b}$$

Distribute the transpose operation:

$$R^T Q^T Q R \mathbf{x} = R^T Q^T \mathbf{b}$$

Apply the orthogonality property to simplify the left side:

$$R^T R \mathbf{x} = R^T Q^T \mathbf{b}$$

Assuming $A$ has full rank, $R$ is invertible. We multiply both sides by $(R^T)^{-1}$ to isolate the remaining terms:

$$\mathbf{R \mathbf{x} = Q^T \mathbf{b}}$$

### 5. Solving via Back-Substitution

Because $R$ is Upper Triangular, we do not need complex inversion algorithms to find $\mathbf{x}$. We use Back-Substitution:
1.  Solve for the last variable: $x_n$
2.  Substitute $x_n$ into the row above to solve for $x_{n-1}$.
3.  Continue upwards until all coefficients are found.

This final step is computationally cheap with $O(n^2)$, making the entire process highly efficient.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Jupyter Notebook

### Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/mattia-3rne/multiple-linear-regression-via-qr-decomposition.git](https://github.com/mattia-3rne/multiple-linear-regression-via-qr-decomposition.git)
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Analysis**:
    ```bash
    jupyter notebook
    ```

---

## 📂 Project Structure

* `main.ipynb`: The main jupyter notebook.
* `dataset_generation.py`: Script to generate the data.
* `requirements.txt`: Python package dependencies.
* `README.md`: Project documentation.