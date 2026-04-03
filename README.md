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

### 2. **QML - Quantum Machine Learning** (`QML/`)

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

3. **Learn Quantum Machine Learning**
   - Navigate to `QML/qsvr.md` and follow the tutorial step-by-step

## 📚 Educational Purpose

This repository is designed as an educational resource for learning:
- **Quantum Physics**: Hydrogen atom wavefunctions and electron orbital visualization
- **Computational Methods**: Monte Carlo sampling and numerical visualization
- **Quantum Computing**: Introduction to quantum algorithms using Qiskit
- **Machine Learning**: Classical and quantum approaches to regression problems

## 🔧 Quantum Number Constraints

For hydrogen atom orbitals, quantum numbers must satisfy:
- **n** (principal): must be ≥ 1
- **l** (angular momentum): must satisfy 0 ≤ l ≤ n-1
- **m** (magnetic): must satisfy -l ≤ m ≤ l

These constraints ensure physically valid quantum states of the hydrogen atom.

## 📝 Notes

- All orbital calculations use the Born interpretation of the wavefunction (probability density)
- 3D visualizations show the spatial distribution of electron probability
- The simulator supports both complex and real orbital representations
- Results are reproducible using fixed random seeds

---

**For questions or contributions, please refer to the individual project documentation.**
