# Hybrid Control and AI for Industrial Furnaces

This repository contains the work developed for an industrial challenge proposed by RONAL Group. The goal of the project was to model the temperature dynamics of an aluminum holding furnace and compare a physics-based approach with a data-driven approach based on artificial intelligence.

The project combines concepts from modern control theory, system identification, machine learning, and industrial process analysis.

## Project Description

Two different modeling approaches were developed and compared.

The first approach uses modern control theory to obtain a mathematical representation of the furnace through transfer functions, state-space models, controller design, and state observers.

The second approach uses process data to learn the furnace dynamics through an LSTM Autoencoder and Dynamic Mode Decomposition with Control (DMDc). The objective was to determine whether latent representations obtained from machine learning could reproduce the behavior of the physical system.

## Industrial Context

Aluminum holding furnaces are used to maintain molten aluminum at the desired operating temperature before manufacturing processes. These systems are affected by several disturbances, including heat losses, extraction of material, process delays, and thermal inertia.

Understanding these dynamics is important for process monitoring, forecasting, and future control applications.

## Methodology

### Control Model

The control-based approach includes:

- Transfer function modeling
- Delay approximation using Padé approximation
- State-space representation
- Pole placement controller design
- State observer design
- Simulink implementation
- System response analysis

### Artificial Intelligence Model

The data-driven approach includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Correlation analysis
- Principal Component Analysis (PCA)
- LSTM Autoencoder
- Latent state extraction
- Dynamic Mode Decomposition with Control (DMDc)
- State-space identification from latent variables

## Repository Structure

```text
datos/          Dataset files
notebooks/      Exploratory analysis and experiments
src/            Python source code
simulink/       Simulink models and MATLAB files
salidas/        Generated results and figures

README.md
requirements.txt
config.json
LICENSE

