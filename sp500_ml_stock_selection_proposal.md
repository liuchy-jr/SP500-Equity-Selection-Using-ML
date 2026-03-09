# Machine Learning--Based Cross-Sectional Stock Selection for the S&P 500

## Team Composition & Initial Exploration

Our team consists of three members.

-   **Chenyu Liu** is primarily responsible for data preprocessing,
    feature engineering, and exploratory analysis.
-   **Yi Lu** focuses on Random Forest and XGBoost model development,
    evaluation, and interpretation of results.
-   **Yukang Luo** is responsible for Logistic Regression modeling as
    well as risk assessment and mitigation strategies.

All members collaborate on experimental design and report writing.

During the initial brainstorming phase, we explored several potential
topics, including:

-   Time-series forecasting of individual stock prices\
-   Index-level return prediction\
-   Volatility modeling

Ultimately, we selected a **cross-sectional stock selection problem**
because it aligns more closely with real-world investment
decision-making and allows us to evaluate machine learning models in a
realistic out-of-sample setting.

------------------------------------------------------------------------

# Problem Statement & Motivation

The goal of this project is to predict **next-month stock performance
rankings among S&P 500 constituents** using machine learning models
trained on technical indicators derived from historical price and volume
data.

Specifically, we focus on **August month-end observations** and attempt
to predict **September cross-sectional stock returns**.

This restriction is **intentional rather than arbitrary**. Our objective
is not to model generic month-to-month stock price movements across the
entire calendar year, but rather to study a targeted investment setting
in which price signals may contain stronger firm-specific information.

August is a particularly meaningful formation month because it occurs
shortly after the **second-quarter earnings reporting season**. By this
time, a substantial amount of updated corporate information has been
released and incorporated into market prices. Institutional investors
often adjust their holdings after Q2 earnings announcements, making
August-end prices a potentially informative reflection of post-earnings
sentiment and portfolio rebalancing behavior.

By using **August signals to predict September returns**, we aim to
examine whether machine learning models can capture this **post-earnings
adjustment effect** in a cross-sectional stock selection setting.

Although focusing only on August reduces the number of temporal decision
points, it may also reduce the inclusion of noisier months in which
short-term price movements are less closely related to earnings-related
information.

------------------------------------------------------------------------

# Data Description & Collection Plan

We use historical daily stock data for companies included in the **S&P
500 index** from **February 2013 to February 2018**. The dataset
contains standard OHLCV variables:

-   Open\
-   High\
-   Low\
-   Close\
-   Volume

In our working environment, the dataset is stored locally at:

    E:\MSE 623\project\archive\individual_stocks_5yr

Each file in this directory corresponds to **one individual stock**
containing its daily OHLCV history.

Example data loading procedure:

``` python
import os
import pandas as pd

data_path = r"E:\MSE 623\project\archive\individual_stocks_5yr"

files = os.listdir(data_path)

all_data = []

for file in files:
    df = pd.read_csv(os.path.join(data_path, file))
    df["ticker"] = file.replace(".csv","")
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
```

------------------------------------------------------------------------

# Feature Engineering

To construct predictive signals, we generate technical indicators
derived from the raw OHLCV data. These features capture **momentum,
trend, volatility, and trading activity**.

We approximate **one month as 21 trading days**.

  -----------------------------------------------------------------------
  Category                Feature Name            Definition /
                                                  Calculation
  ----------------------- ----------------------- -----------------------
  Momentum                Momentum_1M             (Close_t /
                                                  Close\_{t-21}) - 1

  Momentum                Momentum_3M             (Close_t /
                                                  Close\_{t-63}) - 1

  Trend                   Price_to_MA20           Close_t / MA20_t

  Trend                   MA_20_slope             Slope of regression on
                                                  last 20 days of closing
                                                  prices

  Volatility              Volatility_20d          Std of daily returns
                                                  over last 20 days

  Volatility              High_Low_Range          High_t / Low_t

  Volume                  Volume_Ratio            Volume_t /
                                                  Volume_MA20_t

  Volume                  1M_Accum_Vol_Change     Change in cumulative
                                                  monthly trading volume
                                                  vs previous month
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Planned Machine Learning Approach

The project is formulated as a **supervised learning problem**.

The prediction target is the **next-month stock return following each
August observation**.

We evaluate three models:

### Logistic Regression

A simple interpretable baseline model.

### Random Forest

Captures nonlinear relationships among technical indicators.

### XGBoost

A gradient boosting algorithm known for strong performance on structured
tabular data.

These models allow us to analyze the trade-off between
**interpretability and predictive performance**.

------------------------------------------------------------------------

# Verification, Evaluation & Comparison Plan

To avoid **look-ahead bias**, we use **time-series cross-validation**
rather than standard random k-fold validation.

Random k-fold CV is inappropriate for financial datasets because it
mixes observations across time and can introduce information leakage.

Instead we implement **expanding-window validation**:

  Training Data   Validation Data
  --------------- -----------------
  2013            2014
  2013--2014      2015
  2013--2015      2016

After hyperparameter tuning using these folds, the final model is
retrained on:

    2013–2016

and evaluated on the **2017 holdout dataset**.

------------------------------------------------------------------------

# Portfolio Evaluation

Evaluation includes both **statistical metrics** and **investment
performance metrics**.

### Statistical Metrics

Logistic Regression: - Accuracy - AUC

Random Forest / XGBoost: - RMSE - MAE

### Portfolio-Based Evaluation

Stocks are ranked according to predicted performance.

We construct a **top 20% portfolio** consisting of the highest-ranked
stocks and compare its realized return against a **baseline
equal-weighted S&P 500 portfolio**.

------------------------------------------------------------------------

# Implementation Workflow (Jupyter Notebook)

The final deliverable will be a **Jupyter Notebook (.ipynb)**
implementing the full machine learning pipeline.

Main notebook steps:

1.  Data Loading\
2.  Data Preprocessing\
3.  Feature Construction\
4.  Target Construction\
5.  Model Training\
6.  Time-Series Cross Validation\
7.  Final Out-of-Sample Testing\
8.  Portfolio Construction\
9.  Robustness Analysis

------------------------------------------------------------------------

# Risks & Mitigation Strategies

A primary risk is the limited number of yearly observations due to the
**August-only design**. This is mitigated by leveraging the
**cross-sectional dimension** (hundreds of stocks each year) and using
time-series validation.

Another risk is **model overfitting**, particularly for ensemble models.
This will be addressed through regularization, conservative
hyperparameter tuning, and comparison with simpler models.

------------------------------------------------------------------------

# Project Management Plan

**Weeks 1--2**\
Data preprocessing and feature construction.

**Weeks 3--4**\
Model implementation and training.

**Weeks 5--6**\
Time-series cross-validation, evaluation, and robustness analysis.

**Weeks 7--8**\
Final notebook completion, report writing, and presentation preparation.
