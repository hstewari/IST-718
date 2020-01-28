import pandas as pd
import numpy as np
import statsmodels.formula.api as smf 
#%%
flights = pd.read_pickle('flights.pkl') #read and load saved dataframe to continued analysis
print(flights.shape)
#%%
# Create dummy variables for analysis
dummies = pd.get_dummies(flights['carrier_name'])
df_dummies = pd.concat([flights,dummies], axis=1)
#df_dummies.columns
print(df_dummies[['MONTH', 'DAY_OF_WEEK']])
#%%
#+ CARRIER_DELAY + WEATHER_DELAY + NAS_DELAY + SECURITY_DELAY + LATE_AIRCRAFT_DELAY
my_model1 = str('CARRIER_DELAY ~  Q("Alaska Airlines Inc.") + Q("Allegiant Air") +Q("American Airlines Inc.") + Q("Delta Air Lines Inc.") + Q("Endeavor Air Inc.") + Q("Envoy Air") + Q("ExpressJet Airlines Inc.") + Q("Frontier Airlines Inc.") + Q("Hawaiian Airlines Inc.") + Q("JetBlue Airways") + Q("Mesa Airlines Inc.") + Q("PSA Airlines Inc.") + Q("Republic Airline") + Q("SkyWest Airlines Inc.") + Q("Southwest Airlines Co.") + Q("Spirit Air Lines") + Q("United Air Lines Inc.") + Q("Virgin America")')

train_model_fit = smf.ols(my_model1, data = df_dummies).fit()

print(train_model_fit.summary())
#%%
import calendar
df_dummies['MONTH'] = df_dummies['MONTH'].apply(lambda x: calendar.month_abbr[x])
month_dum = pd.get_dummies(df_dummies['MONTH'])
month_dummies = pd.concat([flights,month_dum], axis = 1)
month_dummies.columns
#%%
my_model2 = str('DEP_DELAY ~ Q("Apr") + Q("Aug") + Q("Dec") + Q("Feb") + Q("Jan") + Q("Jul") + Q("Jun") + Q("Mar") + Q("May") + Q("Nov") + Q("Oct") + Q("Sep") + CARRIER_DELAY + WEATHER_DELAY + NAS_DELAY + SECURITY_DELAY + LATE_AIRCRAFT_DELAY')
train_model_fit2 = smf.ols(my_model2, data = month_dummies).fit()

print(train_model_fit2.summary())
#%%
my_model3 = str('DEP_DELAY ~ CARRIER_DELAY + WEATHER_DELAY + NAS_DELAY + SECURITY_DELAY + LATE_AIRCRAFT_DELAY')
train_model_fit3 = smf.ols(my_model3, data = month_dummies).fit()
print(train_model_fit3.summary())
#%%
my_model1 = str('CARRIER_DELAY ~  Q("Alaska Airlines Inc.") + Q("Allegiant Air") +Q("American Airlines Inc.") + Q("Delta Air Lines Inc.") + Q("Endeavor Air Inc.") + Q("Envoy Air") + Q("ExpressJet Airlines Inc.") + Q("Frontier Airlines Inc.") + Q("Hawaiian Airlines Inc.") + Q("JetBlue Airways") + Q("Mesa Airlines Inc.") + Q("PSA Airlines Inc.") + Q("Republic Airline") + Q("SkyWest Airlines Inc.") + Q("Southwest Airlines Co.") + Q("Spirit Air Lines") + Q("United Air Lines Inc.") + Q("Virgin America")')

train_model_fit = smf.ols(my_model1, data = df_dummies).fit()

print(train_model_fit.summary())
#%%
my_model2 = str('WEATHER_DELAY ~  Q("Alaska Airlines Inc.") + Q("Allegiant Air") +Q("American Airlines Inc.") + Q("Delta Air Lines Inc.") + Q("Endeavor Air Inc.") + Q("Envoy Air") + Q("ExpressJet Airlines Inc.") + Q("Frontier Airlines Inc.") + Q("Hawaiian Airlines Inc.") + Q("JetBlue Airways") + Q("Mesa Airlines Inc.") + Q("PSA Airlines Inc.") + Q("Republic Airline") + Q("SkyWest Airlines Inc.") + Q("Southwest Airlines Co.") + Q("Spirit Air Lines") + Q("United Air Lines Inc.") + Q("Virgin America")')

train_model_fit2 = smf.ols(my_model2, data = df_dummies).fit()

print(train_model_fit2.summary())
#%%
my_model3 = str('NAS_DELAY ~  Q("Alaska Airlines Inc.") + Q("Allegiant Air") +Q("American Airlines Inc.") + Q("Delta Air Lines Inc.") + Q("Endeavor Air Inc.") + Q("Envoy Air") + Q("ExpressJet Airlines Inc.") + Q("Frontier Airlines Inc.") + Q("Hawaiian Airlines Inc.") + Q("JetBlue Airways") + Q("Mesa Airlines Inc.") + Q("PSA Airlines Inc.") + Q("Republic Airline") + Q("SkyWest Airlines Inc.") + Q("Southwest Airlines Co.") + Q("Spirit Air Lines") + Q("United Air Lines Inc.") + Q("Virgin America")')

train_model_fit3 = smf.ols(my_model3, data = df_dummies).fit()

print(train_model_fit3.summary())
#%%
my_model4 = str('SECURITY_DELAY ~  Q("Alaska Airlines Inc.") + Q("Allegiant Air") +Q("American Airlines Inc.") + Q("Delta Air Lines Inc.") + Q("Endeavor Air Inc.") + Q("Envoy Air") + Q("ExpressJet Airlines Inc.") + Q("Frontier Airlines Inc.") + Q("Hawaiian Airlines Inc.") + Q("JetBlue Airways") + Q("Mesa Airlines Inc.") + Q("PSA Airlines Inc.") + Q("Republic Airline") + Q("SkyWest Airlines Inc.") + Q("Southwest Airlines Co.") + Q("Spirit Air Lines") + Q("United Air Lines Inc.") + Q("Virgin America")')

train_model_fit4 = smf.ols(my_model4, data = df_dummies).fit()

print(train_model_fit4.summary())
#%%
my_model5 = str('LATE_AIRCRAFT_DELAY ~  Q("Alaska Airlines Inc.") + Q("Allegiant Air") +Q("American Airlines Inc.") + Q("Delta Air Lines Inc.") + Q("Endeavor Air Inc.") + Q("Envoy Air") + Q("ExpressJet Airlines Inc.") + Q("Frontier Airlines Inc.") + Q("Hawaiian Airlines Inc.") + Q("JetBlue Airways") + Q("Mesa Airlines Inc.") + Q("PSA Airlines Inc.") + Q("Republic Airline") + Q("SkyWest Airlines Inc.") + Q("Southwest Airlines Co.") + Q("Spirit Air Lines") + Q("United Air Lines Inc.") + Q("Virgin America")')

train_model_fit5 = smf.ols(my_model5, data = df_dummies).fit()

print(train_model_fit5.summary())
#%%
my_model6 = str('WEATHER_DELAY ~ Q("Apr") + Q("Aug") + Q("Dec") + Q("Feb") + Q("Jan") + Q("Jul") + Q("Jun") + Q("Mar") + Q("May") + Q("Nov") + Q("Oct") + Q("Sep")')
train_model_fit6 = smf.ols(my_model6, data = month_dummies).fit()

print(train_model_fit6.summary())
#%%
my_model7 = str('CARRIER_DELAY ~ Q("Apr") + Q("Aug") + Q("Dec") + Q("Feb") + Q("Jan") + Q("Jul") + Q("Jun") + Q("Mar") + Q("May") + Q("Nov") + Q("Oct") + Q("Sep")')
train_model_fit7 = smf.ols(my_model7, data = month_dummies).fit()

print(train_model_fit7.summary())
#%%
my_model8 = str('NAS_DELAY ~ Q("Apr") + Q("Aug") + Q("Dec") + Q("Feb") + Q("Jan") + Q("Jul") + Q("Jun") + Q("Mar") + Q("May") + Q("Nov") + Q("Oct") + Q("Sep")')
train_model_fit8 = smf.ols(my_model8, data = month_dummies).fit()

print(train_model_fit8.summary())
#%%
my_model9 = str('SECURITY_DELAY ~ Q("Apr") + Q("Aug") + Q("Dec") + Q("Feb") + Q("Jan") + Q("Jul") + Q("Jun") + Q("Mar") + Q("May") + Q("Nov") + Q("Oct") + Q("Sep")')
train_model_fit9 = smf.ols(my_model9, data = month_dummies).fit()

print(train_model_fit9.summary())
#%%
my_model10= str('LATE_AIRCRAFT_DELAY ~ Q("Apr") + Q("Aug") + Q("Dec") + Q("Feb") + Q("Jan") + Q("Jul") + Q("Jun") + Q("Mar") + Q("May") + Q("Nov") + Q("Oct") + Q("Sep")')
train_model_fit10 = smf.ols(my_model10, data = month_dummies).fit()

print(train_model_fit10.summary())
#%%
##Further models are made iteratively and repetitively to try and find
##something decent in the data