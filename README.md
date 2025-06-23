# Code repository for Article Entitled: Modeling-T-Helper-Cell-Dynamics-and-Cytokine-Profiles-with-Quantum-Neural-Networks
## Abstract
Understanding the dynamics of T helper 1 (Th1) and T helper 2 (Th2) cells is a key challenge in designing immunotherapy and has garnered significant interest over the past decade. Although quantum machine intelligence has emerged as a novel approach to dynamic modeling, it has yet to be applied to this specific biological problem. In this study, we introduce an architecture of Quantum Neural Networks (QNNs) inspired by the biological mechanisms of T-cell differentiation. The findings show that the proposed model effectively captures Th1/Th2 dynamics and cytokine profiles across five case studies. Furthermore, additional insights from the model inference reveal T cell dominance patterns within patient groups and disease stratification.

## Model archicture

![Quantum Circuit](quantumCircuit.jpg)

## Biological analogy to quantum entanglement

| **Biological Concept**                                       | **Quantum Analog**                         |
|--------------------------------------------------------------|---------------------------------------------|
| Presence of a cell or molecule                               | Qubit state $\ket{1}$                       |
| Absence of a cell or molecule                                | Qubit state $\ket{0}$                       |
| Interaction/evolution over time                              | Arbitrary rotation gate $R(\phi,\theta,\omega)$ |
| Conditional activation                                       | CNOT / Toffoli gate                         |
| Inter-cellular regulatory effect                             | Entanglement (CZ or CNOT layout)           |
| Signal inhibition (e.g., by IFN-$\gamma$, IL-10)             | Controlled flipping to $\ket{0}$           |

## Datasets
### Overview
In the numerical validation, we demonstrate the proof of concept for the proposed model through five case studies, summarized in Table below. We address two primary problems:
- Modeling of Th1 and Th2 Dynamics
- Modeling of T cell-induced cytokine dynamics (or cytokine profiles)
For each problem, we categorize the learning tasks into two types:
- Endpoint prediction: Given a single-valued measurement at the end of the monitoring period, we predict the retrospective dynamics of T cells and their cytokines.
- Temporal modeling: Given T cells' retrospective temporal data and cytokine dynamics, we predict their future dynamics through curve-fitting.

| **Case study** | **Th1/Th2 Dynamics** | **Cytokine Profiles** | **End-point** | **Temporal Data** | **Group-wise** |
|----------------|----------------------|------------------------|----------------|-------------------|----------------|
| Dynamical Analysis of T cells in HIV/HCV Patients during ART [Kang et al., 2012](#) | x | x | x | x | x |
| Relationship between T cell balance and Metabolic Profiles [Matia et al., 2021](#) | x | x | x |   |   |
| Insulin-dependent diabetics with newly diagnosed breast cancer [Wintrob et al., 2017](#) |   | x | x |   | x |
| Stratification of tuberculous from a malignant pleural effusion (TPE/MPE) [Zeng et al., 2022](#) |   | x | x |   | x |
| Inflammatory cytokines and Th1/Th2 Balance as prognostic markers for hepatocellular carcinoma after transarterial chemoembolization [Lee et al., 2019](#) |   | x | x |   | x |


### Result summary of T cell dynamics by patient groups

| **Case study**         | **Disease**            | **Positive** | **Negative** | **Th1 Dominant** | **Th2 Dominant** | **Th1/Th2 Ratio Range** |
|------------------------|------------------------|--------------|--------------|------------------|------------------|--------------------------|
| Kang et al., 2012      | HIV/HCV co-infection   | x            |              |                  | x                | [0, 1.2]                 |
|                        | HIV mono-infection     | x            |              | x                |                  | [0, 1.2]                 |
|                        | HCV mono-infection     | x            |              | x                |                  | [1, 5]                   |
|                        | Healthy control        | x            |              |                  | x                | [0, 1]                   |
| Matia et al., 2021     | Abdominal Obesity      |              | x            |                  | x                | [0, 1]                   |
|                        | Abdominal Obesity      | x            |              |                  | x                | [0, 1.5]                 |
|                        | High Blood Pressure    |              | x            |                  | x                | [0, 1]                   |
|                        | High Blood Pressure    | x            |              |                  | x                | [0, 1.5]                 |
|                        | Hyperglycemia          |              | x            |                  | x                | [0, 0.8]                 |
|                        | Hyperglycemia          | x            |              |                  | x                | [0, 0.8]                 |
|                        | Hypertriglyceridemia   |              | x            |                  | x                | [0, 0.8]                 |
|                        | Hypertriglyceridemia   | x            |              |                  | x                | [0, 2]                   |
|                        | Hypercholesterolemia   |              | x            |                  | x                | [0, 0.8]                 |
|                        | Hypercholesterolemia   | x            |              |                  | x                | [0, 2]                   |
|                        | High LDL-C             |              | x            |                  | x                | [0, 0.8]                 |
|                        | High LDL-C             | x            |              |                  | x                | [0, 1]                   |
|                        | Low HDL-C              |              | x            |                  | x                | [0, 1.2]                 |
|                        | Low HDL-C              | x            |              |                  | x                | [0, 0.8]                 |
|                        | High hsCRP             |              | x            |                  | x                | [0, 0.8]                 |
|                        | High hsCRP             | x            |              |                  | x                | [0, 1.6]                 |
|                        | Insulin Resistance     |              | x            |                  | x                | [0, 1]                   |
|                        | Insulin Resistance     | x            |              |                  | x                | [0, 1.5]                 |
| Wintrob et al., 2017   | Control                | x            |              | x                |                  | [0, 1.2]                 |
|                        | No insulin             | x            |              | x                |                  | [1, 5]                   |
|                        | Any insulin            | x            |              | x                |                  | [1, 5]                   |
| Zeng et al., 2022      | TPE                    | x            |              |                  | x                | [0, 1]                   |
|                        | MPE                    | x            |              | x                |                  | [1, 7]                   |

## Code usage
- Naive model architecture using CNOT gates: `2_model_cnot_elayout.py`
- Alternative model architecture using CZ gates: `1_model_cz_elayout.py`

Evaluation code on benchmarking datasets:
```python
eval_[dataset]_[idx].py
```
