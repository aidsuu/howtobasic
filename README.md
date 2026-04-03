# How To Basic 🤓

A repository containing educational resources and implementations for quantum computing and quantum machine learning concepts.

## 📂 Project Structure

### 1. **Atom - Hydrogen Orbital Simulator** (`atom/Hidrogren/`)
A 3D visualization tool for hydrogen atom orbitals. This project simulates and visualizes the probability density of electron orbitals based on quantum numbers.

#### Features
- **Orbital Sampling**: Generates sample points from hydrogen atom wavefunctions
- **3D Visualization**: Creates interactive 3D scatter plots using Plotly
- **Multiple Formats**: Exports results to HTML and JSON formats
- **Quantum Number Support**: Works with arbitrary quantum numbers (n, l, m) that satisfy quantum mechanical constraints

#### Quick Start
```bash
cd atom/Hidrogren
pip install -r requirements.txt
python main.py --n 2 --l 1 --m 0 --samples 50000 --outdir data/samples
```

#### Command-line Arguments
- `--n`: Principal quantum number (default: 2, must be ≥ 1)
- `--l`: Angular momentum quantum number (default: 1, must satisfy 0 ≤ l ≤ n-1)
- `--m`: Magnetic quantum number (default: 0, must satisfy -l ≤ m ≤ l)
- `--samples`: Number of sample points (default: 50000)
- `--seed`: Random seed for reproducibility (default: 42)
- `--outdir`: Output directory for results (default: data/samples)
- `--complex`: Use complex orbitals (default: real orbitals)
- `--kind`: Type of real orbital, 'c' or 's' (default: c)

#### Output
The simulator generates:
- **HTML files**: Interactive 3D visualizations viewable in any web browser
- **JSON files**: Raw data for further analysis (coordinates and probability values)

#### Project Components
- `hydrogen3d/wavefunction.py`: Hydrogen atom wavefunction calculations
- `hydrogen3d/sampling.py`: Monte Carlo sampling of orbital points
- `hydrogen3d/visualize.py`: 3D plotting and visualization
- `hydrogen3d/io_utils.py`: Data import/export utilities
- `hydrogen3d/constants.py`: Physical constants and helper functions

#### Sample Data
Pre-generated sample visualizations are available in `data/samples/`:
- `orbital_n3_l2_m1.html` and `.json`
- `orbital_n10_l3_m1.html` and `.json`

### 2. **Quantum Tunneling 1D** (`quantum_tunneling_1d/`)
A Python implementation of 1D quantum tunneling phenomenon where a Gaussian wavepacket collides with a square potential barrier.

#### Physical Model
This project simulates quantum tunneling based on the **Time-Dependent Schrödinger Equation**:

$$i\hbar\frac{\partial\psi(x,t)}{\partial t} = \left[-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x)\right]\psi(x,t)$$

For a **square barrier potential**:

$$V(x) = \begin{cases}
0 & \text{if } x < a \\
V_0 & \text{if } a \leq x \leq b \\
0 & \text{if } x > b
\end{cases}$$

#### Initial Wavepacket
The simulation starts with a **Gaussian wavepacket** with initial momentum:

$$\psi(x,0) = \exp\left[-\frac{(x-x_0)^2}{4\sigma^2}\right]\exp(ik_0x)$$

#### Energy and Transmission
- **Central Energy**: $E_0 = \frac{\hbar^2 k_0^2}{2m}$
- **For sub-barrier tunneling** ($E_0 < V_0$), the decay factor is: $\kappa = \frac{\sqrt{2m(V_0 - E_0)}}{\hbar}$
- **Approximate transmission coefficient** for sufficiently thick barrier: $T \approx e^{-2\kappa L}$, where $L = b - a$

#### Quick Start
```bash
cd quantum_tunneling_1d
pip install -r requirements.txt
python main.py --V0 1.5 --k0 1.2 --width 10.0 --steps 5000
python main.py --V0 1.5 --k0 1.2 --width 10.0 --steps 5000 --animate
```

#### Command-line Arguments
- `--V0`: Barrier height (default: 1.5)
- `--k0`: Initial wave number (default: 1.2)
- `--width`: Barrier width (default: 10.0)
- `--steps`: Number of time steps (default: 5000)
- `--animate`: Show animation of wavepacket evolution

#### Numerical Method
The simulation uses the **Crank-Nicolson algorithm**, a stable finite-difference scheme for time evolution of the Schrödinger equation.

#### Project Components
- `tunneling/solver.py`: Crank-Nicolson time-stepping solver
- `tunneling/potential.py`: Square barrier potential definition
- `tunneling/wavepacket.py`: Initial Gaussian wavepacket generation
- `tunneling/params.py`: Simulation parameters (spatial/temporal grid, physical constants)
- `tunneling/visualize.py`: Probability density visualization and animation

#### Output
The simulation calculates and displays:
- **Transmission coefficient**: Probability of wavepacket tunneling through the barrier
- **Reflection coefficient**: Probability of wavepacket reflecting from the barrier
- **Probability density**: $|\psi(x,t)|^2$ at the final time
- **Comparison**: Numerical vs. theoretical tunneling probabilities

#### Physical Insights
- Demonstrates quantum tunneling: particles can penetrate barriers even when $E_0 < V_0$
- Exponential dependence of transmission on barrier width and height
- Non-relativistic quantum mechanics in 1D

---

### 3. **QML - Quantum Machine Learning** (`QML/`)

#### [Quantum Support Vector Regression](QML/qsvr.md)
A comprehensive tutorial on converting classical Support Vector Regression (SVR) to Quantum Support Vector Regression (QSVR) using Qiskit.

**Topics Covered:**
- Introduction to quantum machine learning concepts
- Setting up the Qiskit environment
- Data preprocessing and preparation
- Classical SVR model training and tuning
- Quantum kernel implementation
- QSVR model development
- Model comparison and evaluation

## 📋 Requirements

### Hydrogen Orbital Simulator
- numpy
- scipy
- plotly

### Quantum Tunneling 1D
- numpy
- scipy
- matplotlib

### Quantum Machine Learning
- Qiskit (and related packages)
- scikit-learn
- numpy
- matplotlib

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd howtobasic
   ```

2. **Explore Hydrogen Orbitals**
   ```bash
   cd atom/Hidrogren
   pip install -r requirements.txt
   python main.py
   ```

3. **Simulate Quantum Tunneling**
   ```bash
   cd quantum_tunneling_1d
   pip install -r requirements.txt
   python main.py --V0 1.5 --k0 1.2 --animate
   ```

4. **Learn Quantum Machine Learning**
   - Navigate to `QML/qsvr.md` and follow the tutorial step-by-step

## 📚 Educational Purpose

This repository is designed as an educational resource for learning:
- **Quantum Physics**: Hydrogen atom wavefunctions, electron orbital visualization, and quantum tunneling phenomena
- **Computational Methods**: Monte Carlo sampling, numerical visualization, and finite-difference schemes (Crank-Nicolson)
- **Differential Equations**: Solving the time-dependent Schrödinger equation numerically
- **Quantum Computing**: Introduction to quantum algorithms using Qiskit
- **Machine Learning**: Classical and quantum approaches to regression problems

## 🔧 Quantum Number Constraints

For hydrogen atom orbitals, quantum numbers must satisfy:
- **n** (principal): must be ≥ 1
- **l** (angular momentum): must satisfy 0 ≤ l ≤ n-1
- **m** (magnetic): must satisfy -l ≤ m ≤ l

These constraints ensure physically valid quantum states of the hydrogen atom.

## 📝 Notes

### Hydrogen Orbital Simulator
- All orbital calculations use the Born interpretation of the wavefunction (probability density)
- 3D visualizations show the spatial distribution of electron probability
- The simulator supports both complex and real orbital representations
- Results are reproducible using fixed random seeds

### Quantum Tunneling 1D
- The Crank-Nicolson solver is unconditionally stable for the Schrödinger equation
- Transmission and reflection coefficients are computed from the final probability distribution
- The theoretical tunneling approximation is valid for thick barriers and sub-barrier tunneling
- Animation helps visualize the wavepacket splitting at the barrier interface
- Numerical results converge to theoretical predictions with sufficient spatial and temporal resolution

---

**For questions or contributions, please refer to the individual project documentation.**
