## Importing Packages
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import statsmodels.formula.api as smf # linear modeling
import matplotlib.pyplot as plt # plotting
import matplotlib.lines as mlines
import statsmodels.api as sm
import seaborn as sns
import warnings

# read in data from csv file
por = pd.read_csv("student-alcohol-consumption/student-mat.csv")
math = pd.read_csv("student-alcohol-consumption/student-por.csv")

# Combining 2 dataframe
df = pd.concat([por,math], sort= True, ignore_index=False)
# Make a copy of the dataframe for statistical analysis
df_stats = pd.concat([por,math], sort= True, ignore_index=False)
# Make a copy of the dataframe to print out nicely
df_nice = pd.concat([por,math], sort= True, ignore_index=False)

# Rename column in nice df to print out nicely
df_nice = df.rename({'Dalc':'Weekday Alc Consumption (1=low, 5=high)', 'Fedu':'Feather Education (0=none, 5=higher edu)', 'Fjob':'Father Job', 'G1':'Period 1 Grades (0-20 Scale)', 'G2':'Period 2 Grades (0-20 Scale)', 'G3':'Final Grade (0-20 Scale)', 'Medu':'Mother Education (0=none, 5=higher edu)', 'Mjob':'Mother Job', 'Pstatus':'Parents Living Together(T), Apart(A)','Walc':'Weekend Alc Consumption (1=low, 5=high)', 'absences':'Number of School Absences', 'activities':'Extra Curricular Activities', 'address':'Urban(U)/Rural(R) Location', 'age': 'Student Age', 'failures':'Number of Failures', 'famrel':'Family Relationship Quality (1=not good, 5=good)', 'famsize':'Family Size (LE3:<=3, GT3:>3', 'famsup':'Family Education Support', 'freetime':'Free Time (1=low, 5=high)','goout':'Go Out w/ Friends (1=low, 5=high)', 'guardian':'Guardian', 'health':'Current Health Status (1=bad, 5=good)', 'higher':'Wants to take Higher Education', 'internet':'Internet', 'nursery':'Attended Nursery School', 'paid':'Paid for Extra Classes', 'reason':'Reason to Choose this School', 'romantic':'In a Romantic Relationship', 'school':'Student School (GP=Gabriel Pereira, MS=Mousinho da Silveira)', 'schoolsup':'Extra Educational Support', 'sex':'Student Sex', 'studytime':'Weekly Studytime', 'traveltime':'Travel Time to School (1=<15 min, 2=15-30 min, 3=30 min-1 hour, 4=>1 hour)'}, axis='columns')

## Changing numeric variables to categorical variables
df_stats['internet'] = df_stats.internet.factorize( ['yes', 'no'] )[0]
df_stats['activities'] = df_stats.activities.factorize( ['yes', 'no'] )[0]
df_stats['romantic'] = df_stats.romantic.factorize( ['yes', 'no'] )[0]
df_stats["Dalc"] = df_stats["Dalc"].values
df_stats["goout"] = df_stats["goout"].values.astype(str).astype(int)

## Creating new variables "social index"
df_stats["index"] = df_stats['goout'] * 0.25 + df_stats['internet'] * 0.02 + df_stats['romantic'] * 0.03 + df_stats['activities'] * -0.01
df_nice["Social Index"] = df_stats["index"]

## Creating new variables "drinking index"
df_stats["drinking"] = (df_stats["Walc"] * 5 + df_stats["Dalc"] * 2) / 7 
df_nice["Drinking Index"] = df_stats["drinking"]

df_stats['Dalc'] = df_stats.Dalc.astype('category')
df_stats['Walc'] = df_stats.Walc.astype('category')
df_stats['health'] = df_stats.health.astype('category')
df_stats['goout'] = df_stats.goout.astype('category')
df_stats['freetime'] = df_stats.freetime.astype('category')
df_stats['famrel'] = df_stats.famrel.astype('category')
df_stats['studytime'] = df_stats.studytime.astype('category')
df_stats['traveltime'] = df_stats.traveltime.astype('category')
df_stats['Fedu'] = df_stats.Fedu.astype('category')
df_stats['Medu'] = df_stats.Medu.astype('category')
df_stats['internet'] = df_stats.internet.astype('category')
df_stats['romantic'] = df_stats.romantic.astype('category')
df_stats['activities'] = df_stats.activities.astype('category')

# creating and saving heat map
plt_heat_map = plt.figure(figsize=(16,9))
sns.heatmap(df_stats.corr().abs(), annot=True, cmap="Greens").set_title('Correlation Matrix')
plt.show()
plt_heat_map.savefig('img/plt_heat_map.png', dpi=plt_heat_map.dpi)

# First OLS regression model
regression = smf.ols(formula='drinking ~ school + sex + age + address + Pstatus + famsize +Medu + Fedu + Mjob + Fjob + reason + guardian + traveltime + studytime + failures + schoolsup + famsup + paid + activities + nursery + higher + internet + romantic + famrel + freetime + goout + health + absences + Course + G1 + G2 + G3 + index', data = df_stats).fit()
regression = smf.ols(formula='drinking ~ school + sex + age + address + Pstatus + famsize +Medu + Fedu + Mjob + Fjob + reason + guardian + traveltime + studytime + failures + schoolsup + famsup + paid + activities + nursery + higher + internet + romantic + famrel + freetime + goout + health + absences + Course + G1 + G2 + G3 + index', data = df_stats).fit()

# generating prediction using the model
df_nice['prediction'] = pd.DataFrame(regression.predict())

# Actual DWI vs. Predicted DWI based on regression prediction
plt_actual_pred_dwi_before = plt.figure(figsize=(16,9))
plt.scatter(df_nice["Drinking Index"], df_nice["prediction"], alpha=0.2)
plt.xlabel("Actual Drinking Index")
plt.ylabel("Predicted Drinking Index")
plt.title("Actual DWI vs. Predicted DWI")
plt.plot(df_nice["Drinking Index"], df_nice["Drinking Index"], c="r")
plt.xlim(0,6)
plt.xlim(0,6)
plt.show()
plt_actual_pred_dwi_before.savefig('img/plt_actual_pred_dwi_before.png', dpi=plt_actual_pred_dwi_before.dpi)

# OLS regression model after removing insignificant variables
regression_aft_removing = smf.ols(formula='drinking ~ sex + age + Medu + Fedu + Fjob + reason + guardian + traveltime + studytime+ paid + nursery + famrel + goout + health + absences + Course + index', data = df_stats).fit()

# generating prediction using the model
df_nice['full_prediction'] = pd.DataFrame(regression_aft_removing.predict())

# Actual DWI vs. Predicted DWI based on regression_aft_removing prediction
plt_actual_pred_dwi_aft = plt.figure(figsize=(16,9))
plt.scatter(df_nice["Drinking Index"], df_nice["full_prediction"], alpha=0.2)
plt.xlabel("Actual Drinking Index")
plt.ylabel("Predicted Drinking Index")
plt.title("Actual DWI vs. Predicted DWI (after removing insignificant variables)")
plt.plot(df_nice["Drinking Index"], df_nice["Drinking Index"], c="r")
plt.xlim(0,6)
plt.xlim(0,6)
plt.show()
plt_actual_pred_dwi_aft.savefig('img/plt_actual_pred_dwi_aft.png', dpi=plt_actual_pred_dwi_aft.dpi)

# residue of actual dwi vs. predicted dwi (AFTER)
diff = df_nice["Drinking Index"] - df_nice["full_prediction"]
plt_resid_actual_pred_dwi_aft = plt.figure(figsize=(16,9))
plt.scatter(df_nice["Drinking Index"], diff, alpha=0.3)
plt.axhline(0, c='r')
plt.xlabel("DWI")
plt.ylabel("Residuals")
plt.title('Residual Error of DWI')
plt.show()
plt_resid_actual_pred_dwi_aft.savefig('img/plt_resid_actual_pred_dwi_aft.png', dpi=plt_resid_actual_pred_dwi_aft.dpi)

# removing prediciton and full_prediction from nice dataframe
df_nice = df_nice.drop(['prediction'], axis=1)
df_nice = df_nice.drop(['full_prediction'], axis=1)
#df_nice = df_nice.drop(['Social Index'], axis=1)
#df_nice = df_nice.drop(['Drinking Index'], axis=1)

# histogram plot of social index (how social people are)
plt_hist_social_index = plt.figure(figsize=(16,9))
plt.hist(df_nice["Social Index"], width = 0.1)
plt.xlabel("Social Index")
plt.ylabel("Frequency")
plt.title("Histogram of the Social Index")
plt.show()
plt_hist_social_index.savefig('img/plt_hist_social_index.png', dpi=plt_hist_social_index.dpi)

# Scatter plot of social index vs. DWI
plt_scatter_social_dwi = plt.figure(figsize=(16,9))
plt.scatter(df_nice["Social Index"], df_nice["Drinking Index"], alpha=0.2)
plt.xlabel("Social Index")
plt.ylabel("Drinking Index")
plt.title("Scatter Plot of Social Index vs. DWI")
plt.plot()
plt_scatter_social_dwi.savefig('img/plt_scatter_social_dwi.png', dpi=plt_scatter_social_dwi.dpi)

