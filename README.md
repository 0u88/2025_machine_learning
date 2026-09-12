# Machine Learning 2025 — Coursework

My assignments for **Machine Learning** (Fall 2025) at **National Yang Ming Chiao Tung University (NYCU)**, taught by Prof. Te-Sheng Lin. The course covers neural network approximation theory, classical classifiers, and the math behind score-based generative models.

Each week has a written report (PDF). Most weeks also include code and a short discussion question (`.md`).

## Highlights

- **Weather grid classification and regression (Week 4).** I parsed a temperature grid from Taiwan's Central Weather Administration open data (`O-A0038-003`) into two datasets. A neural network classifies whether each grid point has a valid reading. A second network predicts the temperature at each point.
- **Piecewise regression by combining models (Week 6).** A classifier decides whether a location has data, and a regressor predicts its value only where it does. I compare MLP and random-forest variants.
- **Gaussian Discriminant Analysis from scratch (Week 6).** An implementation of GDA applied to the weather classification task.
- **Runge function approximation (Weeks 2–3).** Fitting the Runge function, and then the function together with its derivative, using a neural network.

## Weekly Overview

| Week | Topic | Materials |
|---|---|---|
| 1 | Backpropagation and SGD for a sigmoid neuron (handwritten derivation) | [Report](Week_1/week1_hw.pdf) |
| 2 | Backpropagation with respect to the input; approximating the Runge function with a neural network | [Report 1](Week_2/week2_hw_1.1.pdf) · [Report 2](Week_2/week2_hw_2.pdf) · [Code](Week_2/week2_hw_2.py) · [Question](Week_2/week2_hw_1.2.md) |
| 3 | Polynomial approximation by shallow tanh networks; fitting a function and its derivative | [Report 1](Week_3/week3_hw_1.1.pdf) · [Report 2](Week_3/week3_hw_2.pdf) · [Code](Week_3/week3_hw_2.py) · [Question](Week_3/week3_hw_1.2.md) |
| 4 | Weather grid classification and regression with neural networks | [Report](Week_4/week4_hw_2.pdf) · [Code](Week_4/code/) · [Question](Week_4/week4_hw_1.md) |
| 5 | Multivariate normal distribution and LDA | [Report](Week_5/week5_hw.pdf) · [Question](Week_5/week5_hw_3.md) |
| 6 | Gaussian Discriminant Analysis; piecewise regression via classification + regression | [GDA report](Week_6/week6_hw_2.pdf) · [Piecewise report](Week_6/week6_hw_3.pdf) · [Code](Week_6/code/) · [Question](Week_6/week6_hw_1.md) |
| 7 | Score matching and score-based generative models | [Report](Week_7/week7_hw_1.pdf) · [Question](Week_7/week7_hw_2.md) |
| 8 | Sliced score matching and SDEs | [Report](Week_8/week8_hw_1.pdf) · [Question](Week_8/week8_hw_2.md) |
| 10 | From the forward SDE to the probability flow ODE | [Report](Week_10/week10_hw_1.pdf) · [Essay (Chinese)](Week_10/week10_hw_2.pdf) · [Question](Week_10/week10_hw_3.md) |

## Project Structure

```
.
├── Week_N/
│   ├── weekN_hw_*.pdf     # Written reports
│   ├── weekN_hw_*.md      # Discussion questions
│   └── code/              # Code for that week (Weeks 4 and 6)
├── requirements.txt
└── LICENSE
```

Each week is self-contained. Week 6 reuses the Week 4 data pipeline, so both folders include their own copy of the data.

## Running the Code

```bash
pip install -r requirements.txt
```

Run each script from its own folder, because data paths are relative:

```bash
# Week 2–3: Runge function approximation
cd Week_2 && python week2_hw_2.py

# Week 4: build the datasets from the CWA XML, then train
cd Week_4/code
python preprocess.py       # O-A0038-003.xml -> classification_data.csv, regression_data.csv
python classification.py
python regression.py
python figure.py           # plot the datasets

# Week 6: open the notebooks
cd Week_6/code && jupyter notebook
```

## Data

`O-A0038-003.xml` is a gridded temperature product from the [Central Weather Administration (CWA) Open Data platform](https://opendata.cwa.gov.tw/). It is a 120 × 67 grid over Taiwan at 0.03° resolution, where `-999` marks points without data.
