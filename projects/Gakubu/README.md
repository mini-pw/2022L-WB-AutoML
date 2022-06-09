# Gakubu
## Authors:
- Hubert Bujakowski
- Mikołaj Gałkowski
- Kacper Kurowski

## Chosen framework - [Autogluon](https://arxiv.org/abs/2003.06505)

## Files description:
- files prepared for respective milestones
- article presenting our results
- presentation of our results
- [`utils.py`](https://github.com/MI2-Education/2022L-WB-AutoML/blob/main/projects/Gakubu/utils.py) - implementation of KFoldCrossValidation for binary classification problems (AutoGluon)
- [`notebook.ipynb`](https://github.com/MI2-Education/2022L-WB-AutoML/blob/main/projects/Gakubu/notebook.ipynb) shows how to run functions implemented in [`utils.py`](https://github.com/MI2-Education/2022L-WB-AutoML/blob/main/projects/Gakubu/utils.py)
- [`framework.py`](https://github.com/MI2-Education/2022L-WB-AutoML/blob/main/projects/Gakubu/framework.py), [`preprocessing/preprocessing.py`](https://github.com/MI2-Education/2022L-WB-AutoML/blob/main/projects/Gakubu/preprocessing/preprocessing.py) - files containing source code of our implementation (GakubuFramework)
- [plots](https://github.com/MI2-Education/2022L-WB-AutoML/tree/main/projects/Gakubu/plots) folder - contains plots of our results, code which creates them
- [data](https://github.com/MI2-Education/2022L-WB-AutoML/tree/main/projects/Gakubu/data) - contains the outcomes of the aforementioned frameworks' benchmarks


## How to run files?
- Autogluon -> import function from file [`utils.py`](https://github.com/MI2-Education/2022L-WB-AutoML/blob/main/projects/Gakubu/utils.py)
- Gakubu framework -> import [`GakubuFramework`](https://github.com/MI2-Education/2022L-WB-AutoML/blob/42a0c6baff1a4c1a355ce1205c4123b66f70a13c/projects/Gakubu/framework.py#L22) class from [`framework.py`](https://github.com/MI2-Education/2022L-WB-AutoML/blob/main/projects/Gakubu/framework.py), inside class there is docstring with more details
