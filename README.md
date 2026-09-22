# Fake News Detection with TF-IDF

A Python undergraduate coursework project comparing text-classification approaches for fake news detection. The historical source explores TF-IDF features, K-nearest neighbours, logistic regression and a linear support-vector classifier.

## Source files
| File | Purpose |
| --- | --- |
| `KNN.py` | KNN experiments for neighbour counts 1–13, error plot and classification metrics |
| `Logistic Regression.py` | Logistic-regression baseline and evaluation |
| `SVM.py` | Linear SVM baseline and evaluation |
| `clean.py` | Earlier exploratory feature-extraction and classification script; despite the filename, it is not a standalone dataset-cleaning pipeline |

The scripts use `CountVectorizer` and `TfidfTransformer`, then split feature arrays into training and test portions. Output includes accuracy, a classification report and custom per-class precision, recall and F-measure calculations.

## Local setup
```sh
python -m venv .venv
# Activate the virtual environment using your shell's activation command.
python -m pip install -r requirements.txt
python KNN.py
python "Logistic Regression.py"
python SVM.py
```

Run from the repository root after supplying `train2.csv` (see [data requirements](DATA.md)). The KNN script opens a plot window, which must be closed to continue to the final printed evaluation. Requirements were inferred from source imports; a historical dependency lockfile was not supplied.

## Evaluation caveats
This repository preserves the original coursework rather than silently substituting a new experiment:

- Vocabulary and IDF weights are fitted before the train/test split, which leaks test-set information into feature preparation. A corrected evaluation should split raw text first and fit a pipeline only on training data.
- The KNN loop examines multiple settings on the same test split; its final predictions use the last fitted model (`k=13`), not an automatically selected optimum. Use a validation set or cross-validation for model selection.
- Scripts can use different split proportions, so their reported metrics are not a controlled comparison without aligning the experiment.
- TF-IDF arrays are densified, potentially requiring substantial memory on the full dataset.
- Custom metric calculations can divide by zero when a class has no predictions or no examples.

Python syntax was checked during repository preparation. The full training experiment was not rerun, and no fresh performance result is claimed. Historical coursework results should not be treated as independently reproduced benchmarks.

## Scope and provenance
This code belongs to the fake-news coursework identified by Lanjie Tang. Source logic is retained; documentation and dependency guidance were added for portfolio review. Virtual environments, editor files, large datasets, private reports and the unverified saved model were excluded. No new open-source licence is assigned to material whose reuse permissions have not been established.

## 中文简介
虚假新闻检测,包含 TF-IDF 特征提取、KNN、逻辑回归和 SVM 实验。保留原始算法代码，并说明数据格式、运行方式与实验方法的局限。未重新训练，也未新增性能指标。
