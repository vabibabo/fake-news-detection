# Data requirements

The original scripts expect `train2.csv` in the current working directory. The supplied file's header was checked and contains the columns used by the scripts:

| Column | Expected value |
| --- | --- |
| `content` | News/article text |
| `fake_label` | Binary target: `0` for real, `1` for fake, according to the source comments |
| `real_label` | Additional label column read by the scripts; its full dataset semantics were not independently established |

Obtain the original coursework dataset locally and place it beside `KNN.py`. Do not substitute another CSV merely by renaming it; confirm its schema and label meanings first. Provide enough examples for `min_df=5`, both classes and KNN neighbour counts up to 13.

The original dataset, preprocessing lineage and redistribution licence have not been fully documented. Therefore no article corpus or saved model is redistributed here. The original local source folder contains the required `train2.csv`; it is excluded from Git along with other CSV, spreadsheet and pickle files. The standalone stopword file is also omitted because these scripts do not load it.
