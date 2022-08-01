# ARMUT Service Recommendation using Assocation Rule Learning.

# Turkey's biggest online service platform 'ARMUT' brings service givers and service receivers together. They provide easy
# access to the services like house-keeping, cleaning, repairs, transportation via their online platform.
# In this project, our goal is to build a service recommendation system for the services that the company sells through using
# Asociation Rule Learning technique.

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime as dt
from mlxtend.frequent_patterns import apriori, association_rules

####################################################################
# 1. Data Preparation
####################################################################

df_ = pd.read_csv('../../../../Datasets/armut_data.csv')
df= df_.copy()

df.isnull().sum()
df.info()

# ServiceId represents a different feature in every CategoryID.
# We need to combine these two features with '_' in the middle.

df['service'] = df['ServiceId'].astype(str)+'_'+df['CategoryId'].astype(str)

# df['period'] = df[['ServiceId', 'CategoryId']].astype(str).agg('_'.join, axis=1) #alternative way

# Attention. As you can see from the dataset, we do not see any shopping bucket or bill description.
# Since we need to apply Association Rule, a bill explanation or bucket description need to be made.
# Here, we can describe the shopping bucket as ' services taken in one month by the customers'.
# For example, Userid 25446's bucket will be composed of 4_5, 48_5, 6_7, 47_7 for the 8th month.

df['CreateDate'] = df['CreateDate'].apply(pd.to_datetime)
df['New_Date'] = df['CreateDate'].dt.to_period('M')

df['SepetID'] = df['UserId'].astype(str) + '_' + df['New_Date'].astype(str)
df.head()
####################################################################
# 2. Creation of Association Rules and Providing Shopping Recommendations
###################################################################

#df.groupby(['SepetID','service']).agg({'service': 'count'}).unstack().iloc[0:20,0:20] #there are NaNS

#df.groupby(['SepetID','service']).agg({'service': 'count'}).unstack().fillna(0).iloc[0:20,0:20] #fill NANs with 0.

#new_df = df.groupby(['SepetID','service']).\
#                agg({'CategoryId': 'count'}).\
#                unstack().\
#                fillna(0).\
#               applymap(lambda x: 1 if x>0 else 0)

apr_re = df.groupby(['SepetID', 'service'])["service"].count().unstack().fillna(0).applymap(lambda x: 1 if  x>0 else 0)

frequent_itemsets = apriori(apr_re, min_support = 0.01, use_colnames = True,low_memory=True)
frequent_itemsets.sort_values('support', ascending = False)
rules =  association_rules(frequent_itemsets,
                   metric = 'support',
                   min_threshold=0.01)

rules.sort_values('support', ascending = False)

rules[(rules['support']>0.01) & (rules['lift']>0.05) & (rules['confidence']>0.05)].sort_values('confidence',ascending= False)

# id: 22492

product_id = 2_0

def arl_recommender(rules_df, service, rec_count=5):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, serv in enumerate(sorted_rules["antecedents"]):
        for j in list(serv):
            if j == service:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return print("Recommended service for the service you entered -->",recommendation_list[0:rec_count])

arl_recommender(rules,"2_0")




