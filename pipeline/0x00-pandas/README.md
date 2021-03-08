# 0x00. Pandas

### Description

This project is about learning some basics of PANDAS.

### General Objectives

* What is pandas?
* What is a pd.DataFrame? How do you create one?
* What is a pd.Series? How do you create one?
* How to load data from a file
* How to perform indexing on a pd.DataFrame
* How to use hierarchical indexing with a pd.DataFrame
* How to slice a pd.DataFrame
* How to reassign columns
* How to sort a pd.DataFrame
* How to use boolean logic with a pd.DataFrame
* How to merge/concatenate/join pd.DataFrames
* How to get statistical information from a pd.DataFrame
* How to visualize a pd.DataFrame

### Mandatory Tasks

| File | Description |
| ------ | ------ |
| [0-from_numpy.py](0-from_numpy.py) | Creates a pd.DataFrame from a np.ndarray. |
| [1-from_dictionary.py](1-from_dictionary.py) | Creates a pd.DataFrame from a dictionary |
| [2-from_file.py](2-from_file.py) | Loads data from a file as a pd.DataFrame. |
| [3-rename.py](3-rename.py) | Rename the column Timestamp to Datetime, Convert the timestamp values to datatime values, Display only the Datetime and Close columns. |
| [4-array.py](4-array.py) | Takes the last 10 rows of the columns High and Close and convert them into a numpy.ndarray |
| [5-slice.py](5-slice.py) | Slice the pd.DataFrame along the columns High, Low, Close, and Volume_BTC, taking every 60th row |
| [6-flip_switch.py](6-flip_switch.py) | Alter the pd.DataFrame such that the rows and columns are transposed and the data is sorted in reverse chronological order |
| [7-high.py](7-high.py) | Sort the pd.DataFrame by the High price in descending order. |
| [8-prune.py](8-prune.py) | Remove the entries in the pd.DataFrame where Close is NaN |
| [9-fill.py](9-fill.py) | Fill in the missing data points in the pd.DataFrame |
| [10-index.py](10-index.py) | Index the pd.DataFrame on the Timestamp column |
| [11-concat.py](11-concat.py) | Index the pd.DataFrames on the Timestamp columns and concatenate them |
| [12-hierarchy.py](12-hierarchy.py) | Rearrange the MultiIndex levels such that timestamp is the first level |
| [13-analyze.py](13-analyze.py) | Descriptive statistics for all columns in pd.DataFrame except Timestamp |
| [14-visualize.py](14-visualize.py) | Visualize the pd.DataFrame |
