import seaborn as sns
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import collections 
get_ipython().run_line_magic('matplotlib', 'inline')



# recommend to use these pandas method by iris dataset
iris = datasets.load_iris()
df_feature_iris = pd.DataFrame(iris['data'],columns=['sepal_length','sepal_width','petal_length','petal_width'])
df_species_iris = pd.DataFrame(iris['target'],columns=['species'])
df=df_feature_iris
df.head()


# In[24]:

#specify mutiple term of column

df[(df['sepal_length']>=7)&(df['sepal_width']>=3)]

# a way just like query
df.query("sepal_length>=7 and sepal_width>=3")


# In[26]:


#get specified data tuope ('int','float','bool','object')

df.select_dtypes(['float']).head()


# In[27]:


"""
count at each index 
ex: make group by 'sepal_length' and count 'sepal_width' value in the range 

the index count that 'sepal_length is '4.5 and'sepal_width is '2.3 => 1
"""

df.groupby('sepal_length')['sepal_width'].value_counts().unstack().head()


# In[29]:


"""
by using two columns(A,B), count third column value

ex: make group of A column('sepal_length':4.3〜7.9）and B('sepal_width'),
count C column('sepal_width') mean value（2.0~4.4）

=>'sepal_length' is 4.5 and 'sepal_width'is 2.3 and  'petal_length' mean is 1.3
"""

df.pivot_table(index='sepal_length', columns='sepal_width',values='petal_length', aggfunc=np.mean).head()


# In[52]:


# delete specified data type
df.select_dtypes(exclude=np.int).head()


# In[ ]:


# count all column NaN value

print(df.isnull().sum())


# In[ ]:


# replace NaN value to other (ex:replace NaN to 0)
df=df.fillna(0)


# In[47]:


# z-score normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dff = sc.fit_transform(df)
dff=pd.DataFrame(dff)

# make same column name
listed=list(df.columns)
dff.columns = listed
df=dff
df.head()

# undo z-score normalization
inv_df=sc.inverse_transform(dff)
inv_df=pd.DataFrame(inv_df)
inv_df.columns = listed
inv_df


# In[ ]:


# delete some string from string value （'iraidate' is column name）
b=[]
for i,v in f2.iterrows():
    b.append(str(v['iraidate'])[:3])
# make DataFrame
f=pd.DataFrame(b, columns=['iraidate'])
# delete previous same column 
df=df.drop("iraidate", axis=1)
# concatenate and make same column 
df=pd.concat([df, f], axis=1)


# In[ ]:


# show pandas data scatter diagram

sns.jointplot('kaiin_id','safety_tesuryo', data=df)


# In[11]:


# indicate all cloumn name
df.columns


# In[69]:


# change column name
df=df.rename(columns={'last_login_year_cnts1': 'last_login_year_cnt1'})


# In[27]:


# delete some columns 
df=df.drop('kaiin_id', axis=1)


# In[17]:


# change all columns name（'df_colum_list' include all column name)
df.columns=df_colum_list


# In[72]:


# another way to change all columns name

f2=f2.ix[:,['brand_cnt1', 'brand_cnt2', 'brand_cnt3', 'buyer_city_weight',
       'buyer_rating_bad_weight', 'cate_cnt1', 'cate_cnt2', 'cate_cnt3',
       'cate_cnt4', 'claimflg_weight', 'coupon_cnt1', 'coupon_cnt2',
       'force_cancel_weight', 'safety_tesuryo_weight', 'season_id_cnt1',
       'season_id_cnt2', 'tanka', 'use_point_weight', 'last_login_year_cnt2',
       'updyear_weight', 'popularity_boost_weight', 'buyer_comment_weight',
       'last_login_year_cnt1']]


# In[85]:


# concatenate at vartical axis 
df=pd.concat([df, f])


# In[80]:


# concatenate at horizontal axis 
df=pd.concat([df, f], axis=1)


# In[ ]:


# make DataFrame with column name from list
df=pd.DataFrame(listed, columns=['行名'])


# In[87]:


# save or load DataFrame to or from csv file 

df.to_csv('feature.csv')
df=pd.read_csv('feature.csv')

