# Final Project: Student Alcohol Consumption
## Eric Lin, Naveen Janarthanan, Estelle Jiang, Nuo Chen
### Dataset Source: https://www.kaggle.com/uciml/student-alcohol-consumption

### Problem Domain
An issue that persists in modern day is **abusive alcohol consumption** by adolescents. These adolescents tend to start drinking at a very young age for various physical, emotional, and lifestyle changes. Puberty and learning how to live independently often contribute to the commence of alcohol consumption. However, due to the immature mindset that most adolescents have during these early ages, they tend to make bad decisions regarding anything they might term as "risky" or "cool", such as consuming large amounts of alcohol to get drunk. In fact, fifty-one percent of junior and senior high school students have had at least one drink within the past year and 8 million students drink weekly. More than 3 million students drink alone, more than 4 million drink when they are upset, and less than 3 million drink because they are bored. In addition, parents, friends, and alcoholic beverage advertisements influences students’
attitudes about alcohol. Students' drinking habit is often heavily influenced by their surroundings and is likely impacted by the local student culture and norms. As such, we wanted to analyze this issue in further detail by looking at all the possible variables that could potentially have an effect on student alcohol consumption, such as personal statistics and education values, and produce a model to help predict student drinking rates based on these features.

[Underage Drinking](https://pubs.niaaa.nih.gov/publications/AA67/AA67.htm)
[YOUTH AND ALCOHOL: A NATIONAL SURVEY DRINKING HABITS, ACCESS, ATTITUDES, AND KNOWLEDGE ](https://oig.hhs.gov/oei/reports/oei-09-91-00652.pdf)    
[STUDENT DRINKING: CULTURE, CHANGE, AND COCHRANE](https://uk.cochrane.org/news/student-drinking-culture-change-and-cochrane)


### Project Description
The ultimate goal of our research is to **determine the drinking rate** of a range of students (ages 15 - 22) based on various personal and educational factors provided in the dataset. We are trying to determine how much a student’s statistics (such as education, relationships) and their parent’s statistics (such as education),  has an effect on the amount alcohol a student consume in a week. We found the student alcohol consumption dataset on Kaggle. It contains some interesting social, gender, and study factors about students. Some of the interesting features include the number of past class failures or whether a student is engage in a romantic relationship. To examine our hypothesis, first, we will conduct a **multivariable linear regression test** to determine whether the coefficients are statistically significant. (Feature learning) Then, we will utilize machine learning skills to predict the student’s alcohol consumption rate based on all features provided. Machine learning techniques such as the **decision tree, the K-nearest neighbor, and the polynomial transformation** will assist us to obtain a better prediction of the student alcohol consumption rate. Our main target audience are parents! There has been an increase in the number of students who drink, especially in their late-teen years to early adulthood. We want to provide a model for parents to determine what is the best way, in terms of student performance in and out of school to ensure that their children maintain safe amounts of alcoholic consumption. Parents should be able to notice the rates at which students consume alcohol, based on the correlation between their stats and their children’s stats.

### Technical Description
For our final web resource, we decided to use an **HTML page** (based on a rendered Markdown file) to present our research and developed model, as we had some experience doing so from the previous assignments. We are utilizing the dataset from Kaggle, which is a great website to discover and analyze open data. After looking through the dataset, we found out that there are numerous missing data points that we need to deal with before computing any machine learning on the data. Hence, we will being utilizing Python’s imputer to handle these missing values. Although, this method is imperfect, it will give us a more accurate model than simply dropping all rows with missing data.  In addition, we would need to set a domain for the dependent variables, as the value is bounded by various ranges. In addition to the algorithms we have been using throughout this class, we plan to run our data through the Bagging and Random Forest algorithm, which is one of the most powerful machine learning algorithms available. A major challenge that we will face is **overfitting the data during the machine learning process**. We will have to think extremely carefully on what variables to eliminate during the feature engineering process. In addition, another issue we may encounter is that our data is formatted in categories, which we will likely need to separate into dummy variables to ensure a better model.
