# Columbia Fintech Bootcamp: Module #12 Challenge

---

Feature Summary: Credit Risk Classification

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this Challenge, I use various techniques to train and evaluate models with imbalanced classes. I used a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

- Includes the following tasks:

    * Split the Data into Training and Testing Sets
    * Create various machine learning supervised learning models:  Logistic Regression, Decision Tree, Random Forest, KNN
    * Using all the above models, made predictions using various scaling formats: Original data, random oversampled data, random undersampled data, Cluster Centroid, SMOTE, and SMOTEEN
    * Additionally, evaluated the impact of scaling (normalizing) the input/original data set and its impact on prediction performance with the various models and scaling techniques.

---

## Technologies

This project leverages python 3.9 with the following packages:
* [pandas](https://github.com/pandas-dev/pandas) - A powerful data analysis toolkit.
* [numpy](https://numpy.org/) - A core library for scientific computing in Python
* [sklearn](https://scikit-learn.org/) - Simple and efficient tools for predictive data analysis
* [imblearn](https://imbalanced-learn.org/) - Provides tools when dealing with classification with imbalanced classes
* [matplotlib](https://matplotlib.org/) - Tools for creating static, animated, and interactive visualizations
* [seaborn](https://seaborn.pydata.org/) - Statistical data visualization tools

This project leverages python 3.9 with the following packages:

Jupyter Lab 3.3.2 is required

- *Jupyter Lab is primarily used as a web-based development environment for the notebooks, code, and data associated with this project.  Its flexible interface allows users to configure and arrange workflows in data science*

---

## Installation Guide

Before running the application first install the following dependencies.

```python
  pip install pandas
  pip install numpy
  pip install sklearn
  pip install imblearn
  pip install matplotlib
  pip install seaborn

```
*Assumption made for module challenge: the* **sys** *and Path module will not be required to be explicitely called out in Installation guide section*

To run Jupyter Lab, need to install Anaconda:
* [Anaconda](https://docs.anaconda.com/anaconda/install/) - an open-source distribution of the Python
---

## Usage

To run the Forecasting Net Prophet application, simply clone the repository and run the **credit_risk_resampling.ipynb** script in Jupyter Lab:

---


## Contributors

Contributors:
- Stratis Gavnoudias
---

## License

GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
