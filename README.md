# Final Project: Student Alcohol Consumption
## Eric Lin, Naveen Janarthanan, Estelle Jiang, Nuo Chen
### Dataset Source: https://www.kaggle.com/uciml/student-alcohol-consumption


### Project Description
The ultimate goal of our research is to determine the drinking rate of a range of students (ages 15 - 22) based on various personal and educational factors provided in the dataset. We are trying to determine how much a student’s statistics (such as education, relationships) and their parent’s statistics (such as education),  has an effect on the amount alcohol a student consume in a week. We found the student alcohol consumption dataset on Kaggle. It contains some interesting social, gender, and study factors about students. Some of the interesting features include the number of past class failures or whether a student is engage in a romantic relationship. To examine our hypothesis, first, we will conduct a multivariable linear regression test to determine whether the coefficients are statistically significant. (Feature learning) Then, we will utilize machine learning skills to predict the student’s alcohol consumption rate based on all features provided. Machine learning techniques such as the decision tree, the K-nearest neighbor, and the polynomial transformation will assist us to obtain a better prediction of the student alcohol consumption rate. Our main target audience are parents! There has been an increase in the number of students who drink, especially in their late-teen years to early adulthood. We want to provide a model for parents to determine what is the best way, in terms of student performance in and out of school to ensure that their children maintain safe amounts of alcoholic consumption. Parents should be able to notice the rates at which students consume alcohol, based on the correlation between their stats and their children’s stats.

### Technical Description
For our final web resource, we decided to use an HTML page (based on a rendered Markdown file) to present our research and developed model, as we had some experience doing so from the previous assignments. We are utilizing the dataset from Kaggle, which is a great website to discover and analyze open data. After looking through the dataset, we found out that there are numerous missing data points that we need to deal with before computing any machine learning on the data. Hence, we will being utilizing Python’s imputer to handle these missing values. Although, this method is imperfect, it will give us a more accurate model than simply dropping all rows with missing data.  In addition, we would need to set a domain for the dependent variables, as the value is bounded by various ranges. In addition to the algorithms we have been using throughout this class, we plan to run our data through the Bagging and Random Forest algorithm, which is one of the most powerful machine learning algorithms available. A major challenge that we will face is overfitting the data during the machine learning process. We will have to think extremely carefully on what variables to eliminate during the feature engineering process. In addition, another issue we may encounter is that our data is formatted in categories, which we will likely need to separate into dummy variables to ensure a better model.


### Literature:
Our research helps to identify the correlation between students’ drinking habit and other risk factors, including parent demographics, and academic success. Our research can be used to further infer causal relationships between students’ demographics/behavior and the risk of alcohol abuse.

[YOUTH AND ALCOHOL: A NATIONAL SURVEY DRINKING HABITS, ACCESS, ATTITUDES, AND KNOWLEDGE ](https://oig.hhs.gov/oei/reports/oei-09-91-00652.pdf)    

[Student drinking: culture, change and Cochrane](https://uk.cochrane.org/news/student-drinking-culture-change-and-cochrane)  
[USING DATA MINING TO PREDICT SECONDARY SCHOOL STUDENT ALCOHOL CONSUMPTION](https://www.researchgate.net/publication296695210_STUDENT_ALCOHOL_CONSUMPTION_presentation)  

**Key Findings:**

1. Fifty-one percent of junior and senior high school students have had at least one
drink within the past year and 8 million students drink weekly. 

2. More than 3 million students drink alone, more than 4 million drink when they
are upset, and less than 3 million drink because they are bored  

3. Parents, friends, and alcoholic beverage advetiements influences students’
attitudes about alcohol  

4. Students' drinking habit is heavily influenced by their surroundings and is likely 
impacted by the local student culture, and norms
