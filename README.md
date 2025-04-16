<H3>NAME:K KESAVA SAI</H3>
<H3>REGISTER NO: 212223230105.</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Opening the file
df = pd.read_csv("Churn_Modelling.csv")
df.head()
df.tail()
df.info()
df.columns
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/b14d24f0-8646-4115-ab7b-a06df9ebfecb)


```python
# Finding Missing Values
df.isnull().sum()
```

## OUTPUT :
![image](https://github.com/user-attachments/assets/ee5505db-2af2-48a1-a199-054726f2e22e)

```PYTHON
#Check for Duplicates
df["duplicated"] = df.duplicated()

df.loc[df["duplicated"] == True]
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/a5b8bb70-2ad8-4f29-bd0d-7d0e85b3f226)

```python
print("Outliers (Summary Statistics):\n", df.describe(), "\n")
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/b945f257-b8ef-4d29-83db-c5394baa047d)

```python
#Normalize the dataset
standard = StandardScaler()

cols = ['Surname','Geography','Gender','duplicated']
df_copy = df.drop(columns = cols)
df_copy = pd.DataFrame(standard.fit_transform(df_copy), columns=df_copy.columns)
df_copy.head()
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/7fda2294-1289-4eae-8aa3-e6d35ba33503)


```python
#split the dataset into input and output
input_df = df_copy.drop('Exited', axis = 1)
input_df.head()

output_df = df_copy['Exited']
output_df
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/ead693a1-67aa-4f2c-8b18-1de56c80083a)


```python
#splitting the data for training & Testing
x_train, x_test, y_train, y_test = train_test_split(input_df, output_df, test_size = 0.3, random_state =434)
```


```python
#Print the training data and testing data
x_train
y_train
x_test
y_test
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/9b9cfa88-4561-4746-b65c-d8493d460193)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


