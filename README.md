# Hybrid Control and AI for Industrial Furnaces

This repository contains the work developed for an industrial challenge proposed by RONAL Group. The objective of the project was to model the temperature dynamics of an aluminum holding furnace and compare a physics-based approach with a data-driven approach based on artificial intelligence.

The project combines concepts from control engineering, system identification, machine learning, and industrial process analysis.

---

## Project Overview

Two different approaches were developed and compared.

The first approach uses modern control theory to obtain a mathematical representation of the furnace through transfer functions, state-space models, controller design, and state observers.

The second approach uses operational data to learn the furnace dynamics through an LSTM Autoencoder and Dynamic Mode Decomposition with Control (DMDc). The objective was to investigate whether latent representations obtained from machine learning could reproduce the behavior of the physical system.

---

## Industrial Context

Aluminum holding furnaces are used to maintain molten aluminum at the desired operating temperature before manufacturing processes. These systems are affected by multiple disturbances, including heat losses, extraction of material, process delays, and thermal inertia.

Understanding these dynamics is important for process monitoring, forecasting, optimization, and future control applications.

This project was developed in collaboration with RONAL Group as part of an industrial challenge focused on furnace modeling and analysis.

---

## Methodology

### Control Engineering Approach

The control-based model includes:

- Transfer function modeling
- Delay approximation using Padé approximation
- State-space representation
- Pole placement controller design
- State observer design
- Simulink implementation
- Dynamic response analysis

### Artificial Intelligence Approach

The data-driven model includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Correlation analysis
- Principal Component Analysis (PCA)
- LSTM Autoencoder
- Latent state extraction
- Dynamic Mode Decomposition with Control (DMDc)
- State-space identification from latent variables

---

## Repository Structure

```text
datos/          Dataset files
notebooks/      Exploratory analysis and model development
src/            Python source code
simulink/       Simulink models and MATLAB files
salidas/        Generated figures and outputs

README.md
requirements.txt
config.json
LICENSE
RONAL_Industrial_Furnace_Control_Project.pdf
```

---

## Simulink and MATLAB Files

The `simulink` directory contains the Simulink implementation used for the control model.

It also includes the file:

```text
calculos_sistema.mlx
```

This MATLAB Live Script contains the calculations used during the control design stage. The file was developed to simplify and document the mathematical procedures required throughout the project, including state-space analysis, controller design, and supporting calculations.

---

## Main Results

The control model provided an interpretable representation of the furnace dynamics and allowed the implementation of control strategies using modern control techniques.

The machine learning model was able to learn latent representations of the process and generate a state-space approximation through DMDc, allowing a direct comparison with the physics-based model.

The results suggest that control engineering and artificial intelligence can complement each other when analyzing complex industrial systems.

---

## Software Requirements

The project was developed using:

- Python 3.12
- MATLAB
- Simulink

Python dependencies are listed in:

```text
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## Reproducibility

Most project parameters can be modified through:

```text
config.json
```

Generated outputs are stored in:

```text
salidas/
```

---

## Report

The complete technical report can be found in:

```text
RONAL_Industrial_Furnace_Control_Project.pdf
```

---

## Team

This project was developed by a multidisciplinary team of Data Science and Mathematics students as part of the MA2008B industrial challenge in collaboration with RONAL Group.

### Contributors

- Felipe de Jesús Damián Rodríguez
- Facundo Bautista Barbera
- Gabriela Marissa Mosquera Orellana
- Pablo Alberto Ramos Roldán
- Yamil Núñez Sosa
