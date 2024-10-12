#!/usr/bin/env python
# coding: utf-8

# In[353]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway


# In[354]:


df_campaign = pd.read_csv('campaign - campaign.csv')


# In[355]:


df_campaign.shape


#  - This indicates there are 2239 entries or records in the dataset.
#  - There are 27 columns in the dataset, which represent different features related to user sessions.

# In[356]:


df_campaign.head()


# In[357]:


df_campaign.info()


# In[358]:


# 1. Remove currency symbols and commas, and convert to numeric
df_campaign['Income'] = df_campaign['Income'].replace({'\$': '', ',': ''}, regex=True).astype(float)


# In[359]:


df_campaign.head()


# In[360]:


df_campaign.isnull().sum()


# In[361]:


df_campaign.describe()


# Insight:-
# - The high variability in product spending indicates distinct customer segments with different purchasing behaviors. Tailored marketing strategies might be more effective.
# - The higher average store purchases compared to web and catalog purchases may suggest the importance of in-store experiences or limitations in the online and catalog channels.
# - The low acceptance rates for campaigns imply there is room for improvement in targeting and campaign relevance. Analyzing customer preferences more deeply could help design more appealing campaigns.
# - The low complaint rate, combined with the variety in spending behavior, indicates that while most customers are content, there is potential to identify and resolve hidden dissatisfaction or improve customer engagement.  

# In[362]:


df_campaign.describe(include='object')


# Insight:-
# - General Insights and Recommendations:Customer Profiling: Knowing that the majority of customers have a graduation-level education and are married can help craft marketing strategies that appeal to educated and family-oriented individuals. Content, advertisements, and offers could be tailored accordingly.
# - Income Data Quality: The presence of missing or possibly incorrect income data needs addressing. It's essential to clean and validate this data to ensure accurate segmentation and analysis, which are critical for targeted marketing and understanding purchasing power.
# - Customer Acquisition Trends: The spike in enrollments on certain dates suggests that specific events or campaigns were highly effective. Analyzing what drove these peaks can provide valuable insights for future customer acquisition strategies.
# - Geographic Focus: Since Spain has the highest number of customers, it could be beneficial to focus marketing resources, customer service optimization, and product customization efforts in this region. However, understanding the characteristics and preferences of customers from other countries is also important to develop a balanced, global strategy.

# In[363]:


df_campaign['Income'].value_counts


# In[364]:


df_campaign.isnull().sum()


# In[365]:


# Calculate the median of the 'Income' column
median_income = df_campaign['Income'].median()
df_campaign['Income'].fillna(median_income, inplace=True)


# In[366]:


df_campaign.isnull().sum()


# ### Preprocess the data

# In[367]:


df_campaign['Dt_Customer'] = pd.to_datetime(df_campaign['Dt_Customer'])


# In[368]:


df_campaign['TotalSpent'] = df_campaign['MntWines'] + df_campaign['MntFruits'] + df_campaign['MntMeatProducts'] +                             df_campaign['MntFishProducts'] + df_campaign['MntSweetProducts'] + df_campaign['MntGoldProds']


# - Customer Segmentation: By creating a TotalSpent column, you can segment customers based on their spending habits. For example, high spenders vs. low spenders, or identify VIP customers.
# - Targeted Marketing: Knowing the total spending can help tailor marketing campaigns. High spenders might receive exclusive offers or premium product suggestions, while low spenders might be targeted with entry-level promotions to encourage more spending.
# - Predictive Analysis: TotalSpent can be a key feature in predictive modeling, such as predicting customer lifetime value (CLV), likelihood of churn, or response to future campaigns.
# - Customer Value Assessment: This feature allows for quick assessment of each customer's value to the company. It helps in understanding which customers are more valuable, thus informing retention strategies.
# - Insight into Product Preference: By analyzing correlations between TotalSpent and individual categories, you can gain insights into product preferences, helping to understand which product categories drive the most overall revenue.
# - Cross-sell and Upsell Opportunities: Understanding total spending helps in identifying opportunities to cross-sell (suggest complementary products) or upsell (encourage customers to buy more expensive items) based on spending habits.

# In[369]:


df_campaign['TotalAcceptedCampaigns'] = df_campaign['AcceptedCmp1'] + df_campaign['AcceptedCmp2'] +                                         df_campaign['AcceptedCmp3'] + df_campaign['AcceptedCmp4'] + df_campaign['AcceptedCmp5']


# Insight:-
# 
# - Customer Engagement: The TotalAcceptedCampaigns column is a direct measure of customer engagement with marketing efforts. A higher value indicates a customer is more receptive to marketing campaigns and promotions.
# - Identifying Potential Loyal Customers: Customers with a high number of accepted campaigns are likely more loyal and engaged. These customers can be targeted with loyalty programs or special offers to further enhance their experience and retention.
# - Predicting Future Campaign Success: By analyzing customers who have accepted multiple campaigns in the past, businesses can predict which customers are more likely to respond to future campaigns, optimizing marketing efforts and budget allocation.
# - Segmentation: This feature allows for segmentation of customers based on their responsiveness to marketing campaigns. For example, you can create segments like "Highly Engaged," "Moderately Engaged," and "Not Engaged" based on the TotalAcceptedCampaigns value.
# - Customer Lifetime Value (CLV): A customer who accepts more campaigns may have a higher lifetime value. This metric can be used as an input in CLV prediction models to estimate the long-term revenue potential of each customer.
# - Retention Strategies: Customers who accept campaigns are likely interested in the company's offerings. By understanding which campaigns were successful, you can tailor future campaigns to match the preferences and interests of these engaged customers, thus improving retention rates.
# - Personalized Marketing: Knowing the total number of accepted campaigns can help in personalizing future marketing communications. For example, sending a different type of message or offer to those who are frequent campaign responders compared to those who have never responded.

# In[370]:


df_campaign['LivingStatus'] = df_campaign['Marital_Status'].map(
    {'Married': 'In couple', 'Together': 'In couple', 'Divorced': 'Alone', 
     'Single': 'Alone', 'Absurd': 'Alone', 'Widow': 'Alone', 'YOLO': 'Alone'})


# Insight:-
# - The creation of the LivingStatus column is a strategic step towards simplifying customer segmentation and tailoring marketing efforts. 
# - By categorizing customers based on their living arrangements, we can better understand their needs, preferences, and behaviors. 
# - This insight can lead to more effective marketing strategies, enhanced customer engagement, and optimized product offerings, ultimately driving sales and customer satisfaction.   

# In[371]:


# Check unique values in the Income column
print("\nUnique values in Income column:")
print(df_campaign['Income'].unique())


# In[372]:


# Convert 'Income' to numeric, handling any non-numeric values
df_campaign['Income'] = pd.to_numeric(df_campaign['Income'].replace('[\$,]', '', regex=True), errors='coerce')


# In[373]:


# Calculate Age
current_year = 2024
df_campaign['Age'] = current_year - df_campaign['Year_Birth']


# In[374]:


# Remove any rows with non-numeric Age values
df_campaign = df_campaign[pd.to_numeric(df_campaign['Age'], errors='coerce').notnull()]


# In[375]:


# Convert Age to integer
df_campaign['Age'] = df_campaign['Age'].astype(int)


# In[376]:


# Print summary statistics after cleaning
print("\nSummary statistics after cleaning:")
print(df_campaign[['Age', 'Income']].describe())


# In[377]:


# Print value counts of Education column
print("\nEducation categories:")
print(df_campaign['Education'].value_counts())


# - Understanding the educational background of the customer base can provide valuable insights into tailoring marketing strategies, product offerings, and customer engagement efforts, ultimately leading to higher conversion rates and customer satisfaction.

# ### Univariate Analysis

# In[378]:


# Function to plot histogram for numerical columns
def plot_histograms(df, columns, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.histplot(df[col], ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()


# In[379]:


num_cols2 = df_campaign.select_dtypes(include=['int64']).columns
plot_histograms(df_campaign, num_cols2, 6, 4)


# Insight
# 
# 1. Customer Demographics:
#    - ID: Fairly uniform distribution, suggesting a good spread of unique customers.
#    - Year_Birth: Most customers were born between 1940 and 1980, with a peak around 1970-1975.
#    - Kidhome and Teenhome: Most households have 0 or 1 kid/teen, with very few having 2 or more.
# 
# 2. Customer Behavior:
#    - Recency: Relatively uniform distribution, indicating varied last purchase times.
#    - MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds: All show right-skewed distributions, suggesting most customers spend small amounts, with a few big spenders.
# 
# 3. Purchase Channels:
#    - NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases: Right-skewed distributions, indicating most customers make few purchases through each channel, with some making many.
#    - NumWebVisitsMonth: More normally distributed, peaking around 5-8 visits per month.
# 
# 4. Campaign Responses:
#    - AcceptedCmp1 through AcceptedCmp5: Binary distributions (0 or 1), with most customers not accepting campaigns (0). This suggests low campaign acceptance rates overall.
# 
# 5. Customer Satisfaction:
#    - Complain: Highly skewed towards 0, indicating very few complaints.
# 
# 6. Overall Customer Value:
#    - TotalSpent: Right-skewed distribution, suggesting most customers spend moderate amounts, with a few high-value customers.
#    - TotalAcceptedCampaigns: Most customers accept 0 or 1 campaign, with very few accepting multiple campaigns.
# 
# Key Inferences:
# 1. The customer base is primarily middle-aged to older adults, with small families.
# 2. There's a wide range in customer value, with a small segment of high-value customers across various product categories.
# 3. Web visits are common, but actual purchases across all channels are relatively low for most customers.
# 4. Marketing campaigns have low acceptance rates, suggesting potential for improvement in targeting or offer design.
# 5. Customer satisfaction seems high (low complaint rate), but this doesn't necessarily translate to high engagement with campaigns or purchases.
# 6. There's potential to increase customer value by improving campaign effectiveness and encouraging more frequent purchases across channels.
# 7. The business might benefit from strategies to convert occasional buyers into more frequent purchasers, and to increase acceptance of marketing campaigns.
# 

# ### Demographic analysis for Campaign data

# ### 1. Analyze customer demographics and their impact on spending

# In[380]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df_campaign['Age'], kde=True)
plt.title('Age Distribution')
plt.subplot(1, 2, 2)
sns.boxplot(x='Education', y='Income', data=df_campaign)
plt.title('Income Distribution by Education')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Insight:-
# 
# - Targeting by Age Group: The age distribution suggests that marketing strategies could be tailored according to different age segments to better address their specific needs and preferences.
# - Education-Based Income Insights: Understanding the relationship between education and income can help in segmenting the market. For instance, those with higher education and income may respond better to premium products or services, while those with lower income may prioritize value and affordability.
# - Further Analysis on Outliers: The outliers in the income distribution could represent unique customer segments that have distinct characteristics or purchasing behavior. Identifying these traits could lead to the development of targeted marketing campaigns.

# In[381]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x='Income', y='TotalSpent', data=df_campaign)
plt.title('Income vs Total Spent')
plt.subplot(1, 2, 2)
sns.boxplot(x='LivingStatus', y='TotalSpent', data=df_campaign)
plt.title('Total Spent by Living Status')
plt.tight_layout()
plt.show()


# Insight:-
# 
# 1. Income vs Total Spent:
#    - There is a clear positive correlation between income and total spent. As income increases, the total amount spent generally increases as well.
#    - The relationship appears to be roughly linear, with some scatter around the trend line.
#    - There is a wide range of incomes represented, from near 0 to about 160,000.
#    - Total spent ranges from near 0 to about 2,500 .
#    - There are some outliers, particularly at higher income levels, where spending is lower than the general trend would predict.
#    - The density of points is higher in the lower to middle income ranges, suggesting more customers in these brackets.
# 
# 2. Total Spent by Living Status:
#    - The box plot compares spending patterns between people living alone and those living in couples.
#    - Median spending (represented by the line in the middle of each box) appears to be slightly higher for those living in couples compared to those living alone.
#    - The interquartile range (the box itself) is larger for those living alone, indicating more variability in spending among single individuals.
#    - Both groups have numerous outliers on the high end of spending, represented by the points above the whiskers.
#    - The maximum spending amounts are similar for both groups, reaching just above 2,500.
#    - The lower quartile of spending seems to be higher for those living in couples, suggesting that couples have a higher minimum spending threshold.
# 
# Key Inferences:
# 1. Income is a strong predictor of spending, but it's not the only factor. Other variables (like living status) also play a role.
# 2. Couples tend to spend slightly more than singles on average, possibly due to shared expenses or lifestyle factors.
# 3. There's more consistency in spending patterns among couples compared to singles, as evidenced by the smaller interquartile range.
# 4. High-income outliers who spend less than expected could represent savers or individuals with different spending priorities.
# 5. The majority of customers fall in the lower to middle income brackets, which could inform marketing and product strategies.
# 6. Living status doesn't dramatically change the maximum amount spent, suggesting that other factors (like income) might be more influential for high spenders.
# 7. The data might be useful for personalized marketing strategies, tailoring approaches based on income levels and living status.
# 8. There may be opportunities to increase spending among higher-income individuals who are currently spending less than the trend would predict.
# 

# In[382]:


#Print summary statistics
print(df_campaign[['Age', 'Income']].describe())


# Insights:
# - Age Distribution: The age range is quite broad, from 28 to 131 years, with a mean age of about 55.2 years. This indicates a diverse age group, with a significant concentration in the middle range (around 47 to 65 years).
# - Income Distribution: The income data also shows considerable variation, with a mean income of around $52,000.
# - The income distribution has a large standard deviation, suggesting some outliers or a wide range of incomes. The median income ($51,373) is close to the mean, indicating a fairly symmetric distribution around this central value, though the large standard deviation suggests some high-income outliers.

# ### 2. Analyze campaign acceptance rates

# In[383]:


campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
acceptance_rates = df_campaign[campaign_cols].mean()

plt.figure(figsize=(10, 6))
acceptance_rates.plot(kind='bar')
plt.title('Campaign Acceptance Rates')
plt.ylabel('Acceptance Rate')
plt.show()


# In[384]:


# 3. Analyze factors influencing campaign success

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='TotalAcceptedCampaigns', y='Income', data=df_campaign)
plt.title('Income vs Total Accepted Campaigns')
plt.subplot(1, 2, 2)
sns.boxplot(x='TotalAcceptedCampaigns', y='Age', data=df_campaign)
plt.title('Age vs Total Accepted Campaigns')
plt.tight_layout()
plt.show()


# Insight:
# 
# 1. Income and Campaign Acceptance:
# There's a positive correlation between income and the number of accepted campaigns. As the number of total accepted campaigns increases from 0 to 4, the median income tends to rise. This suggests that higher-income individuals are more likely to accept multiple campaigns.
# 
# 2. Age and Campaign Acceptance:
# There doesn't appear to be a strong correlation between age and the number of accepted campaigns. The median age seems relatively consistent across different numbers of accepted campaigns, though there's a slight decrease in the age range for those accepting more campaigns.
# 
# 3. Campaign Acceptance Rates:
# The bar chart shows acceptance rates for different campaigns. Most campaigns have acceptance rates around 65-75%, except for "AcceptedCmp2" which has a notably lower acceptance rate of about 15%.
# 
# 4. Distribution of Accepted Campaigns:
# From the box plots, we can see that most people accept between 0 and 2 campaigns, with fewer people accepting 3 or 4 campaigns.
# 
# 5. Income Variability:
# There's considerable variability in income for each category of accepted campaigns, as shown by the large boxes and whiskers in the income plot.
# 
# 6. Age Range:
# The majority of participants seem to be between 40 and 80 years old, with some outliers extending to younger and older ages.
# 
# 7. Campaign Performance:
# Campaigns 3, 4, and 5 appear to have the highest acceptance rates, while Campaign 2 significantly underperforms compared to the others.
# 
# These insights suggest that while income may influence campaign acceptance, age is less of a factor. The data also indicates that certain campaigns are more successful than others in terms of acceptance rates, which could inform future marketing strategies.

# In[385]:


for col in ['Education', 'Marital_Status', 'Country']:
    plt.figure(figsize=(10, 5))
    df_campaign[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.show()


# In[386]:


# Customer behavior analysis
behavior_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df_campaign['TotalSpent'] = df_campaign[behavior_cols].sum(axis=1)

plt.figure(figsize=(12, 8))
sns.boxplot(data=df_campaign[behavior_cols])
plt.title('Spending Distribution by Product Category')
plt.xticks(rotation=45)
plt.show()


# Insight:-
# This image shows a box plot representing the spending distribution across different product categories. Here are some key observations and inferences:
# 
# 1. Categories: The plot displays 6 product categories: MiniWines, MfrFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, and MntGoldProds.
# 
# 2. Spending variation: There's significant variation in spending across categories, as evidenced by the different sizes of the boxes and whiskers.
# 
# 3. Highest spending: MiniWines appears to have the highest median spending and the widest range of spending, suggesting it's the most popular category with the most variable consumer behavior.
# 
# 4. Lowest spending: MfrFruits and MntSweetProducts seem to have the lowest median spending and smallest ranges, indicating these are less popular categories or have more consistent, lower spending patterns.
# 
# 5. Outliers: All categories show outliers (represented by dots above the whiskers), but MiniWines and MntMeatProducts have particularly extreme outliers, suggesting some customers spend much more than typical in these categories.
# 
# 6. Skewness: Most categories appear positively skewed, with longer upper whiskers and outliers stretching upwards, indicating some customers spend much more than the typical amount.
# 
# 7. Mid-range categories: MntMeatProducts and MntFishProducts fall in the middle in terms of median spending and variability.
# 
# 8. Consumer behavior: The wide ranges and presence of outliers across categories suggest diverse consumer preferences and spending habits among the customer base.
# 
# 9. Potential focus areas: Given the high spending and variability in MiniWines and MntMeatProducts, these could be key areas for marketing or promotional efforts to capitalize on high spenders or encourage more consistent purchasing.
# 
# This distribution provides valuable insights into customer spending patterns across product categories, which could inform inventory management, marketing strategies, and overall business decision-making. 

# In[387]:


# Correlation analysis for Campaign Data
plt.figure(figsize=(15, 12))
sns.heatmap(df_campaign.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap - Customer Data')
plt.show()


# ### 5. Hypothesis Testing

# In[388]:


# Hypothesis 1: Is income of customers dependent on their education?
education_groups = df_campaign.groupby('Education')['Income']
f_statistic, p_value = stats.f_oneway(*[group for name, group in education_groups])

print("\nHypothesis 1: Income dependence on Education")
print(f"F-statistic: {f_statistic}, p-value: {p_value}")
print("Conclusion: Income is" + (" " if p_value < 0.05 else " not ") + "significantly dependent on education.")


# In[389]:


print(df_campaign['Education'].value_counts())


# In[390]:


print(df_campaign['Education'].unique())


# In[391]:


print(df_campaign[['Education', 'Income']].isnull().sum())


# In[392]:


df_campaign['Income'].fillna(df_campaign['Income'].median(), inplace=True)


# In[393]:


education_groups = df_campaign.groupby('Education')['Income']
groups = [group.dropna() for name, group in education_groups]
if all(len(group) > 1 for group in groups):  # Ensure each group has more than one observation
    f_statistic, p_value = stats.f_oneway(*groups)
    print("\nHypothesis 1: Income dependence on Education")
    print(f"F-statistic: {f_statistic}, p-value: {p_value}")
    print("Conclusion: Income is" + (" " if p_value < 0.05 else " not ") + "significantly dependent on education.")
else:
    print("Some groups are empty or too small for ANOVA.")


# Insight:-
# - F-statistic: A high F-statistic suggests that there is a significant difference in income between at least some of the education groups.
# - p-value: The p-value is far below the typical significance level of 0.05, which means that the difference in income among education groups is statistically significant.
# - Conclusion:Based on the results, income is significantly dependent on education. This implies that education level has a notable impact on income, and there are likely meaningful differences in income across different education levels.

# In[394]:


# Hypothesis 2: Do higher income people spend more?
correlation = df_campaign['Income'].corr(df_campaign['TotalSpent'])
print("\nHypothesis 2: Correlation between Income and Total Spent")
print(f"Correlation coefficient: {correlation}")
print("Conclusion: There is a" + (" strong" if abs(correlation) > 0.5 else " weak") + 
      " correlation between income and total spending.")


# Insight:-
# - Correlation Coefficient: A value of 0.793 is close to 1, suggesting a strong positive linear relationship between income and total spending. This means that as income increases, total spending tends to increase as well.
# - Conclusion:There is a strong correlation between income and total spending. This implies that individuals with higher incomes generally tend to spend more, which could be useful for understanding consumer behavior and for making business decisions related to spending patterns.

# In[395]:


# Hypothesis 3: Do couples spend more or less money on wine than people living alone?
couple_wine = df_campaign[df_campaign['LivingStatus'] == 'In couple']['MntWines']
alone_wine = df_campaign[df_campaign['LivingStatus'] == 'Alone']['MntWines']
t_statistic, p_value = stats.ttest_ind(couple_wine, alone_wine)

print("\nHypothesis 3: Wine spending difference between couples and single people")
print(f"T-statistic: {t_statistic}, p-value: {p_value}")
print("Conclusion: There is" + (" " if p_value < 0.05 else " no ") + 
      "significant difference in wine spending between couples and single people.")


# Insight:
# - T-statistic: The value of -0.302 indicates that the observed difference in wine spending between couples and single people is very close to zero.
# - p-value: A p-value of 0.763 is much higher than the typical significance level of 0.05.
# - Conclusion:There is no significant difference in wine spending between couples and single people. This suggests that, based on your data, the spending on wine does not differ significantly between these two groups.

# In[396]:


# Hypothesis 4: Are people with lower income more attracted towards campaigns?
median_income = df_campaign['Income'].median()
low_income = df_campaign[df_campaign['Income'] <= median_income]['TotalAcceptedCampaigns']
high_income = df_campaign[df_campaign['Income'] > median_income]['TotalAcceptedCampaigns']
t_statistic, p_value = stats.ttest_ind(low_income, high_income)

print("\nHypothesis 4: Campaign acceptance difference between income brackets")
print(f"T-statistic: {t_statistic}, p-value: {p_value}")
print("Conclusion: There is" + (" " if p_value < 0.05 else " no ") + 
      "significant difference in campaign acceptance between income brackets.")


# Insight:-
# - T-statistic: A t-statistic of -13.512 indicates a very large difference between the income brackets in terms of campaign acceptance.
# - p-value: The p-value is extremely low, far below the common significance level of 0.05.
# - Conclusion:There is a significant difference in campaign acceptance between income brackets. This suggests that income level has a substantial effect on whether individuals accept a campaign, and there are likely clear variations in acceptance rates across different income groups.

# ### 6. Identify areas for optimization

# In[397]:


# Analyze customer segments by total spent and campaign acceptance
df_campaign['SpendingCategory'] = pd.qcut(df_campaign['TotalSpent'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

plt.figure(figsize=(12, 6))
sns.boxplot(x='SpendingCategory', y='TotalAcceptedCampaigns', data=df_campaign)
plt.title('Campaign Acceptance by Spending Category')
plt.show()


# Insights
# - Spending Correlation: There is a positive correlation between spending and campaign acceptance. Higher spending categories tend to have higher median acceptance rates.
# - Variability: As spending increases, the variability in campaign acceptance also increases. This suggests that while higher spending can lead to more successful campaigns, it also comes with higher risks.
# - Outliers: The presence of outliers in higher spending categories indicates that some campaigns perform exceptionally well, while others may not meet expectations.

# In[398]:


# Analyze the relationship between web visits and purchases
plt.figure(figsize=(10, 6))
sns.scatterplot(x='NumWebVisitsMonth', y='NumWebPurchases', hue='TotalAcceptedCampaigns', data=df_campaign)
plt.title('Web Visits vs Web Purchases')
plt.show()


# - Marketing Campaign Impact: The presence of color-coded clusters suggests that marketing campaigns may influence the relationship between web visits and purchases.
# - Campaign Success: Observing the clusters associated with higher accepted campaigns can provide insights into successful marketing strategies.
# - Outliers: Investigate the outliers to understand why certain cases deviate significantly from the general trend.

# In[399]:


# Analyze the effectiveness of different products in generating revenue
product_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
product_revenue = df_campaign[product_cols].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
product_revenue.plot(kind='bar')
plt.title('Revenue by Product Category')
plt.ylabel('Total Revenue')
plt.show()


# In[400]:


print("\nProduct Categories by Revenue:")
print(product_revenue)


# - Wines generate the highest revenue, significantly more than any other category, indicating a strong preference for this product among the customers.
# - Meat Products are the second highest, but their revenue is about half that of Wines, suggesting that while popular, they are not as dominant.
# - The remaining categories (Gold Products, Fish Products, Sweet Products, and Fruits) generate substantially less revenue compared to Wines and Meat Products. This could imply a niche market or lower demand for these items.

# ### Recommendations

# - Targeted Campaigns Based on Income:
# Although the T-test showed no significant difference in campaign acceptance between income brackets, it's still essential to segment customers and personalize campaigns based on other factors like purchase history or preferences to increase effectiveness.
# Continue collecting data and exploring other potential factors (e.g., age, education, marital status) that might influence campaign acceptance.
# 
# - Focus on High Revenue Products:
# Prioritize marketing efforts on MntWines and MntMeatProducts to capitalize on their popularity. These could include special promotions, loyalty programs, or targeted email campaigns.
# Develop new wine and meat product offerings or bundles to maintain and grow this customer segment.
# 
# - Boosting Lower Revenue Categories:
# Consider cross-promotional strategies that combine high-revenue products with lower-revenue ones. For example, offering discounts on MntSweetProducts and MntFruits when purchased with wines.
# Highlight the unique selling points of MntGoldProds, MntFishProducts, and other lower revenue categories to attract a different segment of the customer base.
# 
# - Data Quality Improvement:
# Continue to monitor and clean the data to ensure accuracy, especially for critical features like Income. Regularly update missing values and validate data entries.
# Invest in customer feedback mechanisms to better understand the factors influencing their purchase decisions, which can provide deeper insights into optimizing product offerings and campaigns.
# 
# - Explore Further Hypotheses:
# Consider examining other hypotheses, such as the impact of marital status, education level, or recency of the last purchase on campaign acceptance.
# Use clustering techniques to segment customers into different groups based on multiple features, and tailor campaigns specifically for each cluster.
# 
# - Experiment with New Campaign Strategies:
# Based on the insights gained, experiment with new types of campaigns, such as limited-time offers, exclusive product launches, or themed collections that resonate with customers' preferences (e.g., wine tasting events, holiday meat packages).
# 
# By focusing on these areas, the organization can optimize its marketing strategies, improve customer engagement, and potentially increase overall revenue. Let me know if you need more detailed analysis or have additional data to consider!

# ### Conclusion

# - The analysis of the df_campaign dataset provided key insights into customer behavior, campaign effectiveness, and product category preferences. It revealed that while income might not significantly impact campaign acceptance, other factors could play a crucial role. The data showed a strong preference for wine products, with wines generating the highest revenue, followed by meat products. Other categories like gold products, fish products, sweets, and fruits lag behind significantly in revenue generation.
# 
# - To enhance business performance, the company should focus on targeted campaigns, primarily leveraging the popularity of wines and meat products. Cross-promotional strategies could be employed to boost the visibility and sales of lower-revenue product categories. Additionally, maintaining data quality and exploring other potential factors influencing campaign acceptance will help refine marketing strategies and drive customer engagement.
# 
# By implementing these recommendations, the company can optimize its marketing efforts, cater more effectively to customer preferences, and ultimately drive growth in revenue and customer satisfaction.
