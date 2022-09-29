# ARMUT-COMPANY-ARL-RECOMMENDATION

ARMUT Service Recommendation using Assocation Rule Learning. Turkey's biggest online service platform 'ARMUT' brings service givers and service receivers together. They provide easy access to the services like house-keeping, cleaning, repairs, transportation via their online platform.

THE GOAL : to build a service recommendation system for the services that the company sells through using Association Rule Learning technique.

The Dataset: It consists of user ids, serviceid for each element of each cateogiry, cateogryid, the date that the service purchased.

Actions: 
1. Create a feature that combines CATEGORY and SERVICEIDs. categoryid and serviceid is more useful together.
2. Create a feature that gives us the customer bucket. It means that we need to find out the monthly expenditure of the customer. I need to combine the user_id with the date.
3. Build a PIVOT TABLE WITH THESE TWO FEATURES. COLUMNS = CATEGORY_SERVICEID. ROWS = CUSTOMERID+DATE.
4. Build Association rules using Apriori Algorithm
5. Predict what a given customer can buy given that he bought one of your product.
