# Apriori
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# for Apriori algo
from apyori import apriori


# Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results] # change index if min_length and max_length are not 2
    rhs         = [tuple(result[2][0][1])[0] for result in results] # change index if min_length and max_length are not 2
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))


def apr():
    # import data: this dataset does NOT have columns -> each rows have different categorical variables
    # f column names are passed explicitly then the behavior is identical to header=None. (No columns)
    dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

    # print(dataset.head(2))
    '''
            0          1        2   ...               17       18         19
    0   shrimp    almonds  avocado  ...  frozen smoothie  spinach  olive oil
    1  burgers  meatballs     eggs  ...              NaN      NaN        NaN
    [7500 rows x 20 columns]
    '''

    # to chase each customer behaviors we need to split and store (string) the dataset
    transaction = [] # this array stores each customer info

    # for loop method
    # [str(dataset.values[i,j]) is to store each row's values as string
    # for j in range(len(dataset.columns)) is nested loop for traversing each column value
    for i in range(len(dataset)):
         transaction.append([str(dataset.values[i,j]) for j in range(len(dataset.columns))])

    # same results without loop
    # transaction = dataset.values.astype(str).tolist()


    # no need independent variables for Apriori

    # no need dependent variable for Apriori

    # no need splitting dataset

    # no need feature scaling as they are categorical variables


    # train/fit the models
    # apriori(dataset, min_support, min-confidence, min-lift)
    # min_support: we try to see 3 items per day => 21 items per week / all customers
    # min_confidence: default is 0.8, we can try smaller values because min_support is low
    # min_lift: between 3 and 9 is good choice
    # min_length is how many combinations/sets we need at least
    # max_length is how many combinations/sets we can have at most
    # min_length = max_length is only 2 combinations
    rules = apriori(transaction, min_support=float((3*7)/len(dataset)), min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

    # visualizing the rules
    result = list(rules) # put the rules into the list
    print(result)
    '''
    {'chicken', 'extra dark chocolate'}
    items_base=frozenset({'extra dark chocolate'} this means when people buy dark chocolate
    items_add=frozenset({'chicken'}) then, the people buy chicken
    confidence=0.23  -> 23% buy the chicken in the situations
    '''

    resultsinDataFrame = pd.DataFrame(inspect(result), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
    print(resultsinDataFrame)

    # resultsinDataFrame is object of data class

    # sorting by confidence top 10
    print('\n'+str(resultsinDataFrame.nlargest(n=10, columns='Confidence')))

if __name__ == '__main__':
    apr()

