import os
import time
import quandl
import pandas as pd
import requests
from sklearn.externals import joblib
from BeautifulSoup import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

api_key = "Es7sBLK2aTbBe_Cjswh4"  #from Quandl account setting
US_states = [] #list of US States Abbreviation

#extracting US states abbreavation from wikipedia page
def state_list():

    response = requests.get("https://simple.wikipedia.org/wiki/List_of_U.S._states")
    soup = BeautifulSoup(response.content)

    rows = soup.findAll("tr")[1:50]

    for row in rows:
        cols = row.findAll('td')[0]
        US_states.append([element.text.strip() for element in cols if element])

    return pd.DataFrame(US_states, columns=['US_states'])

#preparing dataset with US states HPI data with percent change
def grab_initial_state_data():
    states = state_list()
    main_df = pd.DataFrame()

    for abbv in states ['US_states']:
        query = "FMAC/HPI_"+str(abbv)
        df = pd.DataFrame(quandl.get(query, authtaken=api_key), columns=[abbv])
        import ipdb;ipdb.set_trace()
        df[abbv] = (df[abbv]-df[abbv][0]) / df[abbv][0] * 100.0

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

#collecting and preparing for whole US HPI data
def HPI_Benchmark():
    df = quandl.get("FMAC/HPI_USA", authtoken=api_key)
    df["United States"] = (df["United States"]-df["United States"][0]) / df["United States"][0] * 100.0
    return df

#getting Mortgage rate data for the last 30 years
def mortgage_30y():
    df = quandl.get("FMAC/MORTG", trim_start="1975-01-01", authtoken=api_key)
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df=df.resample('1D')
    df=df.resample('M')
    return df

#getting US GDP data
def gdp_data():
    df = quandl.get("BCB/4385", trim_start="1975-01-01", authtoken=api_key)
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df=df.resample('M').mean()
    df.rename(columns={'Value':'GDP'}, inplace=True)
    df = df['GDP']
    return df

HPI_data = grab_initial_state_data()
m30 = mortgage_30y()
gdp = gdp_data()
HPI_Bench = HPI_Benchmark()
m30.columns=['M30']
HPI = HPI_data.join([m30,gdp,HPI_Bench])
HPI.dropna(inplace=True)

with open('US_HPI_data.pkl','wb') as f:
    joblib.dump(HPI,f)

print(HPI.corr())

#plotting HPI data to visualize correlation
HPI_data.plot()
plt.legend().remove()
plt.show()

#plotting US HPI data with other states
fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))
HPI_data.plot(ax=ax1)
HPI_Bench.plot(color='k',ax=ax1, linewidth=10)
plt.legend().remove()
plt.show()


#getting correlation matrix
HPI_State_Correlation = HPI_data.corr()
print(HPI_State_Correlation)
