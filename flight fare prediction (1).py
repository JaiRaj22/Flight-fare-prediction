#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
pd.set_option('display.max_columns',None)
import pickle


# In[2]:


data = pd.read_excel('Data_Train.xlsx')
data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


data.Duration.value_counts()


# In[7]:


data.dropna(inplace=True)


# In[8]:


data.isnull().sum()


# In[9]:


data['journeyday'] = pd.to_datetime(data.Date_of_Journey, format='%d/%m/%Y').dt.day


# In[10]:


data['journeymonth'] = pd.to_datetime(data.Date_of_Journey, format='%d/%m/%Y').dt.month


# In[11]:


data.head()


# In[12]:


data = data.drop(['Date_of_Journey'], axis=1)


# In[13]:


data = data.rename(columns={"Airline":"Air"})


# In[14]:


data.head()


# In[15]:


data['D-hour'] = pd.to_datetime(data.Dep_Time).dt.hour
data['D-minute'] = pd.to_datetime(data.Dep_Time).dt.minute
data = data.drop(['Dep_Time'], axis=1)


# In[16]:


data['A-hour'] = pd.to_datetime(data.Arrival_Time).dt.hour
data['A-minute'] = pd.to_datetime(data.Arrival_Time).dt.minute
data = data.drop(['Arrival_Time'], axis=1)


# In[17]:


data.head()


# In[18]:


duration = list(data.Duration)
for i in range(len(duration)):
    if len(duration[i].split()) !=2:
        if 'h' in duration[i]:
            duration[i] = duration[i].strip()+ " 0m"
        else:
            duration[i] = "0h " + duration[i]
durationhours=[]
durationmins=[]
for i in range(len(duration)):
    durationhours.append(int(duration[i].split("h")[0]))
    durationmins.append(int(duration[i].split('m')[0].split()[-1]))


# In[19]:


data['durationhours'] = durationhours
data['durationmins'] = durationmins


# In[20]:


data = data.drop(['Duration'], axis=1)


# In[21]:


data.tail()


# In[22]:


data['Air'].value_counts()


# In[23]:


sns.catplot(y="Price", x="Air", data=data.sort_values("Price", ascending=False), kind='boxen', height=6, aspect=3)


# In[24]:


airline = data[['Air']]
airline = pd.get_dummies(airline, drop_first=True)
airline.tail()


# In[25]:


data['Source'].value_counts()


# In[26]:


sns.catplot(y='Price', x='Source', data=data.sort_values('Price', ascending=False), kind='boxen', height=6, aspect=3)


# In[27]:


source = data[['Source']]
source = pd.get_dummies(source, drop_first=True)
source.tail()


# In[28]:


destination = data[['Destination']]
destination = pd.get_dummies(destination, drop_first=True)
destination.tail()


# In[29]:


data['Route']


# In[30]:


a = data.Additional_Info=='No info'


# In[31]:


data = data.drop(['Route', 'Additional_Info'], axis=1)


# In[32]:


data.tail()


# In[33]:


data.Total_Stops.value_counts()


# In[34]:


data = data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4})


# In[35]:


datatrain = pd.concat([data, airline,source,destination],axis=1)


# In[36]:


datatrain.head()


# In[37]:


datatrain = datatrain.drop(['Destination', 'Source', 'Air'], axis=1)


# In[38]:


test_data = pd.read_excel('Test_set.xlsx')


# In[39]:


test_data.head()


# In[40]:


print("Test data Info")
print(test_data.info())

print()
print()

print("Null values :")
test_data.dropna(inplace = True)
print(test_data.isnull().sum())


# In[41]:


test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:  
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   
        else:
            duration[i] = "0h " + duration[i]           

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0])) 
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))  

test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace = True)

print("Airline")
print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first= True)

print()

print("Source")
print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first= True)

print()

print("Destination")
print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)
test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)

data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print()
print()


# In[42]:


data_test.shape


# In[43]:


datatrain.columns


# In[44]:


data_test.head()


# In[45]:


x = datatrain.loc[:,['Total_Stops', 'Price', 'journeyday', 'journeymonth', 'D-hour',
       'D-minute', 'A-hour', 'A-minute', 'durationhours', 'durationmins',
       'Air_Air India', 'Air_GoAir', 'Air_IndiGo',
       'Air_Jet Airways', 'Air_Jet Airways Business',
       'Air_Multiple carriers',
       'Air_Multiple carriers Premium economy', 'Air_SpiceJet',
       'Air_Trujet', 'Air_Vistara', 'Air_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]


# In[46]:


x.tail()


# In[47]:


y = datatrain.iloc[: ,1]


# In[48]:


y.head()


# In[49]:


plt.figure(figsize=(16,16))
sns.heatmap(data.corr(), annot=True)
plt.show()


# In[50]:


from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor()
etr.fit(x,y)


# In[51]:


print(etr.feature_importances_)


# In[52]:


plt.figure(figsize=(12, 8))
feature_imp = pd.Series(etr.feature_importances_, index=x.columns)
feature_imp.nlargest(20).plot(kind='barh')
plt.show()


# In[53]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)


# In[54]:


from sklearn.ensemble import RandomForestRegressor


# In[55]:


rfg = RandomForestRegressor()
rfg.fit(x_train,y_train)


# In[56]:


pred = rfg.predict(x_test)


# In[57]:


rfg.score(x_test,y_test)


# In[58]:


rfg.score(x_train, y_train)


# In[59]:


from sklearn import metrics


# In[60]:


metrics.r2_score(y_test,pred)


# In[61]:


from sklearn.metrics import mean_absolute_error as mae
mae(y_test, pred)


# In[62]:


from sklearn.metrics import mean_squared_error as mse
mse(y_test, pred)


# In[64]:


pickle.dump(rfg,open('flight_rf.pkl','wb'))
model=pickle.load(open('flight_rf.pkl','rb'))


# In[ ]:




