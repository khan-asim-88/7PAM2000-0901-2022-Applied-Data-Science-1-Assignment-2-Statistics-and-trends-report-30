
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

df = pd.read_csv('API_19_DS2_en_csv_v2_4756035.csv')

dfc = pd.read_csv('Metadata_Country_API_19_DS2_en_csv_v2_4756035.csv')

dfc1 = dfc[~pd.isna(dfc['IncomeGroup'])]

df = df[['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code','2001', '2002',
         '2003', '2004','2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012',
         '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']]

df['mean'] = df[['2001', '2002', '2003', '2004', '2005', '2006', '2007',
       '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016',
       '2017', '2018', '2019', '2020', '2021']].mean(axis=1)


selected_indicators = ['Urban population (% of total population)', 'Urban population growth (annual %)', 'Population, total', 'Population growth (annual %)',
 'Mortality rate, under-5 (per 1,000 live births)', 'CO2 emissions (kt)', 'Energy use (kg of oil equivalent per capita)',
 'Electric power consumption (kWh per capita)','Energy use (kg of oil equivalent) per $1,000 GDP (constant 2017 PPP)',
 'Access to electricity (% of population)','Forest area (% of land area)','Arable land (% of land area)', 'Agricultural land (% of land area)']

countries = ['Italy','Australia','Jamaica','China','United Arab Emirates','Germany','France',
             'Malaysia','Japan','Romania','New Zealand','Morocco','United States','Pakistan','Thailand']


df = df.drop(['Country Code','Indicator Code'], axis=1)

def data_ingestion(df, indicator):
    df1 = df[df['Indicator Name'] == indicator]
    df1 = df1.drop(['Indicator Name'], axis=1)
    df1.index = df1.loc[:, 'Country Name']
    df1 = df1.drop(['Country Name'], axis=1)
    df2 = df1.transpose()
    return df1, df2
    

df_year, df_country = data_ingestion(df, 'Population, total')


for ind in selected_indicators:
    df_year, df_country = data_ingestion(df, ind)
    
    for i in df_year.columns:
        sns.swarmplot(y='Country Name',x=i, data=df_year.loc[countries, :].reset_index())
    
    plt.title(ind)
    plt.xlabel('2001-2021')
    plt.show()

for ind in selected_indicators:
    df_year, df_country = data_ingestion(df, ind)
    for i in countries:
        plt.plot(df_country[i], label=i)

    plt.title(ind)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=90)
    plt.show()


for i in selected_indicators:
    temp_df = df[df['Indicator Name'] == i]
    temp_df = temp_df.merge(dfc1,left_on='Country Name', right_on='TableName')
    temp_df.groupby(['Region'])['mean'].mean().sort_values(ascending=False).plot(kind='bar')
    plt.title(i)
    plt.show()
    


df1, df2 = data_ingestion(df, 'Arable land (% of land area)')
df3, df4 = data_ingestion(df, 'Forest area (% of land area)')

df1_mean = df1.mean(axis=1).reset_index().rename({0:'arable land'}, axis=1)
df3_mean = df3.mean(axis=1).reset_index().rename({0:'forest land'}, axis=1)

n_df = df1_mean.merge(df3_mean, on='Country Name')

plt.figure(figsize=(4,3))
sns.scatterplot(x=n_df['arable land'], y=n_df['forest land'])
plt.xlabel('Arable Land')
plt.ylabel('Forest Land')
plt.title('Arable vs Forest Land')
plt.show()

n_df.index = n_df.loc[:, 'Country Name']
n_df.sort_values(by='arable land', ascending=False)[:10].plot(kind='bar',figsize=(4,3))
plt.ylabel('frequency')
plt.title('Top 10 Countries with Max Arable Land')
plt.show()

n_df.sort_values(by='forest land', ascending=False)[:10].plot(kind='bar',figsize=(4,3))
plt.ylabel('frequency')
plt.title('Top 10 Countries with Max Forest Land')
plt.show()



df1, df2 = data_ingestion(df, 'Access to electricity (% of population)')
df3, df4 = data_ingestion(df, 'Electric power consumption (kWh per capita)')

df1_mean = df1.mean(axis=1).reset_index().rename({0:'Access to Electricity'}, axis=1)
df3_mean = df3.mean(axis=1).reset_index().rename({0:'Electric Power Consumption'}, axis=1)

n_df = df1_mean.merge(df3_mean, on='Country Name')

plt.figure(figsize=(4,3))
sns.scatterplot(x=n_df['Access to Electricity'], y=n_df['Electric Power Consumption'])
plt.xlabel('Access to Electricity')
plt.ylabel('Electric Power Consumption')
plt.title('Access to Electricity vs Electric Power Consumption')
plt.show()

n_df.index = n_df.loc[:, 'Country Name']
n_df.sort_values(by='Access to Electricity', ascending=False)[:10].plot(kind='bar',figsize=(4,3))
plt.ylabel('frequency')
plt.title('Top 10 Countries with Max Access to Electricity')
plt.show()

n_df.sort_values(by='Electric Power Consumption', ascending=False)[:10].plot(kind='bar',figsize=(4,3))
plt.ylabel('frequency')
plt.title('Top 10 Countries with Max Electric Power Consumption')
plt.show()



df1, df2 = data_ingestion(df, 'Energy use (kg of oil equivalent) per $1,000 GDP (constant 2017 PPP)')
df3, df4 = data_ingestion(df, 'CO2 emissions (kt)')

sns.scatterplot(x=df1.mean(axis=1), y=df3.mean(axis=1))
plt.xlabel('Energy use (kg of oil equivalent) per $1,000 GDP (constant 2017 PPP)')
plt.ylabel('CO2 emissions')
plt.title('Emergy use vs CO2 emissions')
plt.show()



df1 = df.groupby(['Country Name','Indicator Name'])['mean'].mean().unstack()

plt.figure(figsize=(10,7))
sns.heatmap(df1[selected_indicators].corr(), cmap='viridis', linewidths=.5, annot=True)


# Correlation Graph of some Countries for "Urban Population" Indicator
df_year, df_country = data_ingestion(df, 'Urban population')

plt.figure(figsize=(10,7))
sns.heatmap(df_country[countries].corr(), cmap='viridis', linewidths=.5, annot=True)


# ## Correlation Graph of Countries for "Arable Land" Indicator

df_year, df_country = data_ingestion(df, 'Arable land (% of land area)')

plt.figure(figsize=(10,7))
sns.heatmap(df_country[countries].corr(), cmap='viridis', linewidths=.5, annot=True)


# ## Correlation Graph of Countries for "Electric Power Consumption" Indicator


df_year, df_country = data_ingestion(df, 'Electric power consumption (kWh per capita)')

plt.figure(figsize=(10,7))
sns.heatmap(df_country[countries].corr(), cmap='viridis', linewidths=.5, annot=True)


print(df1[selected_indicators].describe())


# getting data ready. removing some Regions and join data to get only the countries data.

df2 = df1.merge(dfc1, left_on=df1.index, right_on='TableName', how='inner')
df2.index = df2['TableName']


for i in df2[selected_indicators]:
    sns.barplot(x='TableName', y=i, data=df2[i].sort_values(ascending=False)[:10].reset_index())
    plt.xticks(rotation=90)
    plt.xlabel(i)
    plt.title('Top 10 countries with respect to ' + str(i))
    plt.show()

