## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
```py
import pandas as pd
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```
![image](https://github.com/user-attachments/assets/266fda41-8974-4a15-906f-073e3c3d530e)
```py
ls /content/drive/MyDrive
```
![image](https://github.com/user-attachments/assets/789fca42-5455-471a-8252-dfd4d8f3aedd)
```py
df=pd.read_csv('/content/drive/MyDrive/Encoding Data.csv')
df
```
![image](https://github.com/user-attachments/assets/6f7ae587-a06f-481a-8c9a-1cb1cd544c8a)
```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/555e9aa4-c66f-46af-a69c-e801a77147b6)
```py
 df['bo2']=e1.fit_transform(df[["ord_2"]])
 df
```
![image](https://github.com/user-attachments/assets/67ac55ff-5e00-4027-8d32-fe827c3c12b6)
```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/4da63035-11e4-4f5f-ae38-6aadd0eb1c6d)
```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/bf33ebb6-3365-4f2e-9cf5-7395d668642b)
```py
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/adc676ca-e82b-432f-8ae4-757cd1401db9)
```py
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/6fb3aa2a-b152-46c1-aa00-eaff840d0340)
```py
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/drive/MyDrive/data.csv")
df
```
![image](https://github.com/user-attachments/assets/f5377563-5220-40e7-9289-958d4da95248)
```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/38f63090-c1a3-4fa0-a325-2ca536f51411)
```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/74cf2f4a-771c-43f0-b8d3-ed827c820916)
```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/drive/MyDrive/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/13dc45a8-1cbd-4174-8e85-a8d8670f7883)
```py
df.skew()
```
![image](https://github.com/user-attachments/assets/d23d4d4b-fe0a-49ac-9db6-9c5d32f3688b)
```py
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/d34a9773-2440-4c1d-8b0f-f98e08e29f3c)
```py
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/7263a03a-0967-48aa-86c3-81c81ca8734c)
```py
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/306a2032-16ee-4670-a98b-44dce27ff5e3)
```py
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/d7f50902-4d50-4b38-807a-588703a26bc4)
```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/99aa265f-fa45-453f-8056-dcff51fb6e05)
```py
 df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
 df.skew()
```
![image](https://github.com/user-attachments/assets/0b1d0278-46ee-4f7d-8b01-2c3ee1abe8a7)
```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/ac3dc1b1-dd3c-4aa8-9df0-5bd67cea5cd0)
```py
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
```
```py
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/a828b2f8-1afd-449e-8cb6-ad5a15d538e5)
```py
sm.qqplot(df['Moderate Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/2bb3aed7-c754-4daf-8cd8-053cb432af9a)
```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c23ee17e-7e05-4247-911d-d50abd2b3da3)
```py
df
```
![image](https://github.com/user-attachments/assets/374d7632-fdac-4fa0-a382-8c3a2e2f5aea)
```py
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/5ea7385c-d4c9-42d6-ae14-0eb1050e393c)


      
# RESULT:
       We have performed Feature Encoding and Transformation on the given data set successfully.

       
