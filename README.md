<h1>Python Surprise Predictions on Custom Dataset</h1>
This article is about using <a href=http://surpriselib.com/>surprise</a> library, Python's recommendation sci-kit, to predict user ratings for movies using a different database then the built-in one.<br><br>
There are many powerful recommendation algorithms within surprise. You can compare them to find the best model for your rating database, further enhance the model with parameter tuning, and use that model for predictions.<br><br>
<b>There is one important thing to mention here:</b> Throughout Python SciKits (SciPy Toolkits), you will usually encounter the same notation for using commands to analyze datasets. The common way of analyzing any dataset in python is seperate it into predictors (x) and targets (y). <b>Surprise, on the contrast, has its own way of working</b>. It only accepts databases built in a certain way and therefore you're expected to keep the dataset in one piece and change variable names. If you're used to the convenience of scikit-learn, this would cause you some discomfort. <b>Surprise also has distinct train_test_split, GridSearch, cross_validate and KFold methods</b>; these support its own working algorithm.
<h2>Methodology</h2>
We will use <b>a database that doesn't match identically</b> with the built-in <a href=https://grouplens.org/datasets/movielens/>MovieLens</a> database so that any other database can be fit into these models by following the same steps. We will split it, train the models on the bigger portion and and use the smaller portion for predictions; and finally, then compare predictions with actual ratings.<br><br>
<b>Disclaimer: Your results may vary with the example shown here; use your own results.</b>
<h2>Analysis</h2>
<h3>Importing Modules and Dataframe</h3>
Just like any other analysis, we start by importing the necessary modules:
<pre>
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
import math

from surprise import Dataset
from surprise import Reader
from surprise import SVD, NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline, CoClustering, BaselineOnly, NormalPredictor
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise.model_selection import GridSearchCV
</pre>
Notice that we're importing scikit-learn's train_test_split. It will come in handy.<br>
Datetime will be used for measuring model running time. Math will be used for a custom function.<br>
The long line imports models from surprise. Dataset and Reader are needed to turn a standard dataframe into a surprise dataframe.<br><br>
Next, it's time to load the dataset. You can download it from <a href=https://www.kaggle.com/rajmehra03/movielens100k/downloads/movielens100k.zip/2>this link</a>. It contains the same information found in MovieLens databases, but <b>variable names are different.</b> We'll use the 'ratings.csv' file within that zip, you can change the name to whatever you want to.
<pre>
df = pd.read_csv('C:/Users/Emir/Desktop/Movie Ratings.csv')
df.head()
</pre>
This is what our data looks like:
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>31</td>
      <td>2.5</td>
      <td>1260759144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1029</td>
      <td>3.0</td>
      <td>1260759179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1061</td>
      <td>3.0</td>
      <td>1260759182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1129</td>
      <td>2.0</td>
      <td>1260759185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1172</td>
      <td>4.0</td>
      <td>1260759205</td>
    </tr>
  </tbody>
</table>
<pre>
df.shape
>>>(100004, 4)
</pre>
We have a dataframe with 10.004 observations and 4 variables. The variables are the ID number for users, ID number for movies, rating (in a scale of 5) and a timestamp.
<h3>Removing Movies that are Rated Less than Others</h3>
<b>If a movie is rated only a small number of times, the ratings of that movie can skew prediction algorithms</b>; so we're going to get rid of some observations.
<pre>
df.groupby(['movieId']).size().max()/20
>>>17.05
</pre>
This code groups the observations by movieId and counts them, then takes the maximum number (the maximum times a movie is rated) and divides it by 20 to find a minimum rating count that is 5% of the maximum. You may choose any ratio you want.<br><br>
Let's create a new dataframe just with the rating counts:
<pre>
ratingcounter = pd.DataFrame({'Count' : df.groupby(['movieId']).size()}).reset_index()
ratingcounter.head()
</pre>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>247</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>107</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>59</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>56</td>
    </tr>
  </tbody>
</table>
<pre>
ratingcounter.shape
>>>(9066, 2)
</pre>
There are 9066 movies, and you can see how many times they are rated in the new ratingcounter dataframe.
<pre>
ratingcounter = ratingcounter.loc[ratingcounter['Count'] > 17]
</pre>
This code eliminates any movie with rating count less than 17.
<pre>
ratingcounter.shape
>>>(1428, 2)
</pre>
The number of movies is reduced to 1428. You may think that the number of discarded movies is too big (7638); but in a real life problem, you'll see that it becomes practically very hard to recommend any movie with good accuracy if there are too many movies to recommend. Also, a recommendation algorithm aims to push a customer to more popular choices because they are sold more often.  
<pre>
reducer = df.movieId.isin(ratingcounter.movieId)
df = df[reducer]
df.shape
>>>(71419, 4)
df.groupby(['movieId']).size().min()
>>>18
</pre>
After filtering the movies with the lesat ratings, our new dataset consists of 71.419 movies and the smallest number of ratings becomes 18.
<h3>Model Selection</h3>
<b>If you use the train_test_split of surprise, you can't use your train data with model selection.</b> So, we'll use the scikit-learn version of the train_test_split that we have already imported.
<pre>
df_train, df_test = train_test_split(df, test_size=0.2, random_state=7)
</pre>
Still, we're going to need some changes to make the our database work with surprise.
<pre>
reader = Reader()
dfsv = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
cv_train = Dataset.load_from_df(df_train[['userId', 'movieId', 'rating']], reader)
</pre>
Reader helps turning a standard dataset into surprise dataset.<br>
dfsv is the surprise version of our normal database; this will be used later.<br>
cv_train is our train data.<br><br>
Let's run some models on the cv_train to see model performances. We'll use a for loop for this.<br><br>
<pre>
CVResults = pd.DataFrame(columns = ['Model','RMSE','MAE','Timespan'])

classes = (SVD, NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline, CoClustering, BaselineOnly, NormalPredictor)

data = cv_train
kf = KFold(2, random_state=0)

for model in classes:
    
&nbsp;&nbsp;&nbsp;&nbsp;start = datetime.datetime.now()
&nbsp;&nbsp;&nbsp;&nbsp;out = cross_validate(model(), data, ['rmse', 'mae'], kf)
&nbsp;&nbsp;&nbsp;&nbsp;mean_rmse = '%.3f' % np.mean(out['test_rmse'])
&nbsp;&nbsp;&nbsp;&nbsp;mean_mae = '%.3f' % np.mean(out['test_mae'])
&nbsp;&nbsp;&nbsp;&nbsp;cv_time = str(datetime.datetime.now() - start)[:-3]

&nbsp;&nbsp;&nbsp;&nbsp;CVResults = CVResults.append({'Model': model.__name__, 'RMSE': mean_rmse, 'MAE': mean_mae, 'Timespan': cv_time}, ignore_index=True)

print('All models have run. Call the CVResults dataframe for results.')
</pre>
<pre>
>>>All models have run. Call the CVResults dataframe for results.
</pre>
CVResults is the database where all the results are stored. This dataframe is created at the beginning of this part of the code; in case something goes wrong and you need to run the code again, the dataframe will be recreated so that all previously appended rows would become void.<br><br>
The loop puts the train data into every model we've defined and runs it two fold; then calculates means of RMSE and MAE (root mean squared error and mean absolute error), measures the time to run all this, and apppends all this information into the CVResults dataframe. The printed line indicates that the code has stopped running successfully.
<pre>
CVResults
</pre>
Here are the results:
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>RMSE</th>
      <th>MAE</th>
      <th>Timespan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SVD</td>
      <td>0.887</td>
      <td>0.682</td>
      <td>0:00:03.355</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NMF</td>
      <td>0.942</td>
      <td>0.725</td>
      <td>0:00:03.407</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SlopeOne</td>
      <td>0.904</td>
      <td>0.694</td>
      <td>0:00:04.527</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KNNBasic</td>
      <td>0.947</td>
      <td>0.732</td>
      <td>0:00:02.920</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KNNWithMeans</td>
      <td>0.894</td>
      <td>0.684</td>
      <td>0:00:03.363</td>
    </tr>
    <tr>
      <th>5</th>
      <td>KNNBaseline</td>
      <td>0.884</td>
      <td>0.676</td>
      <td>0:00:04.518</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CoClustering</td>
      <td>0.944</td>
      <td>0.736</td>
      <td>0:00:01.908</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BaselineOnly</td>
      <td>0.882</td>
      <td>0.680</td>
      <td>0:00:00.575</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NormalPredictor</td>
      <td>1.389</td>
      <td>1.107</td>
      <td>0:00:00.764</td>
    </tr>
  </tbody>
</table>
<b>KNNBaseline has the lowest RMSE, and BaselineOnly has the lowest MAE.</b><br><br>
For the sake of this article, let's continue with both of the models and explore what's ahead.
<h3>Parameter Selection</h3>
Since we know which models performed best, it's time to see whether we can improve individual model performances or not.<br><br>
We're starting with BaselineOnly. <b>BaselineOnly model has a parameter space called bsl_options and the way to use this in grid search is different</b> than standard options. You need <a href=https://surprise.readthedocs.io/en/stable/getting_started.html#tune-algorithm-parameters-with-gridsearchcv>a particular treatment</a> where you define grid search parameters within another parenthesis.
<pre>
param_gridBO = {'bsl_options': {'method': ['als', 'sgd'],
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'reg': [1, 2],
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'learning_rate': [0.01, 0.05, 0.0025],
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'n_epochs': [5, 10, 15]}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}

gsBO = GridSearchCV(BaselineOnly, param_gridBO, measures=['rmse', 'mae'], cv=3)

gsBO.fit(data)

print('Best RMSE:', gsBO.best_score['rmse'], gsBO.best_params['rmse'])
print('Best MAE:', gsBO.best_score['mae'], gsBO.best_params['mae'])
</pre>
The method option allows to choose between alternating least squares and stochastic gradient descent.<br>
Reg denotes the regulation strength.<br>
Learning rate refers to <a href=https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/>the amount that the weights are updated during training</a>.<br>
Number of epochs specifies <a href=https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/>the number times that the learning algorithm will work through the entire training dataset</a>.<br>
After the grid search, best RMSE and MAE scores along with their respective parameters are printed.
<pre>
>>>Best RMSE: 0.8750872800496899 {'bsl_options': {'method': 'als', 'reg': 1, 'learning_rate': 0.01, 'n_epochs': 15}}
>>>Best MAE: 0.6736459751580958 {'bsl_options': {'method': 'als', 'reg': 1, 'learning_rate': 0.01, 'n_epochs': 15}}
</pre>
Since they both belong to the same parameter combination, we'll use that combination for predictions.<br><br>
Let's move on to the KNNBaseline model:
<pre>
param_gridKNNB = {'n_epochs': [5, 10, 15],
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'lr_all': [0.002, 0.005],
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'reg_all': [0.4, 0.6, 0.8]
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}

gsKNNB = GridSearchCV(KNNBaseline, param_gridKNNB, measures=['rmse', 'mae'], cv=3)

gsKNNB.fit(data)

print('Best RMSE:', gsKNNB.best_score['rmse'], gsKNNB.best_params['rmse'])
print('Best MAE:', gsKNNB.best_score['mae'], gsKNNB.best_params['mae'])
>>>Best RMSE: 0.8697179739351221 {'n_epochs': 5, 'lr_all': 0.002, 'reg_all': 0.4}
>>>Best MAE: 0.6658498081065872 {'n_epochs': 5, 'lr_all': 0.002, 'reg_all': 0.4}
</pre>
Again, both RMSE and MAE are achieved by the same parameter combination.
<h3>Predictions</h3>
Remember the dfsv database that we built at the beginning? Now it's time to use it.<br>
<b>Surprise doesn't accept the test split of the standard Pandas dataframe</b> even after it is converted to a surprise dataset. So, we're going to need another method to get our test dataset. For this, the train_test_split method of surprise needs to be called: Beware that <b>since you've already imported scikit-learn's splitter, this train_test_split needs another name</b> so that it doesn't replace the previous one.
<pre>
from surprise.model_selection import train_test_split as tts2
trainset, testset = tts2(dfsv, test_size=0.2, random_state=7)
</pre>
The new splitter is named as "tts2" and it splitted the converted dataframe with the same test size and random state as before.<br><br>
First, we'll get predictions from the BaselineOnly model:
<pre>
predictions = BaselineOnly(bsl_options=gsBO.best_params['rmse']).fit(trainset).test(testset)
ResultCatcher = pd.DataFrame(predictions, columns=['userId', 'movieId', 'Real_Rating', 'Estimated_Rating', 'details'])
ResultCatcher.drop(['details'], axis=1, inplace=True)
</pre>
The nature of our dataset requires some adjustments for predictions: Currently, predictions are float numbers with many decimal places but the actual ratings are multiples of 0.5. To match predictions to actual ratings, we need a custom Python function.
<pre>
def halfrounder(x):
&nbsp;&nbsp;&nbsp;&nbsp;frac, whole = math.modf(x)
&nbsp;&nbsp;&nbsp;&nbsp;if frac > 0.7499999:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a = 1
&nbsp;&nbsp;&nbsp;&nbsp;elif frac < 0.25:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a = 0
&nbsp;&nbsp;&nbsp;&nbsp;else:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a = 0.5
&nbsp;&nbsp;&nbsp;&nbsp;return whole + a
</pre>
This function takes a number, divides it into its whole and fraction, then assigns a new fraction depending on the old fraction and gives out the whole number + the new fraction. Let's apply it to the ResultCatcher dataframe and find the difference between actual ratings and predictions:
<pre>
ResultCatcher['Estimation_Rounded'] = ResultCatcher.apply(lambda row: halfrounder(row.Estimated_Rating), axis=1)
ResultCatcher['Error'] = abs(ResultCatcher['Real_Rating'] - ResultCatcher['Estimation_Rounded'])
ResultCatcher.head()
</pre>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>Real_Rating</th>
      <th>Estimated_Rating</th>
      <th>Estimation_Rounded</th>
      <th>Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>57</td>
      <td>1953</td>
      <td>5.0</td>
      <td>4.428659</td>
      <td>4.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>585</td>
      <td>3052</td>
      <td>4.0</td>
      <td>3.801224</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>205</td>
      <td>1210</td>
      <td>3.0</td>
      <td>3.824343</td>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>120</td>
      <td>1721</td>
      <td>3.5</td>
      <td>3.257743</td>
      <td>3.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>547</td>
      <td>4239</td>
      <td>3.0</td>
      <td>3.434260</td>
      <td>3.5</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
<pre>
ResultCatcher['Real_Rating'].sum()
>>>52024.0
ResultCatcher['Error'].sum()
>>>9240.0
ResultCatcher['Real_Rating'][ResultCatcher['Error'] == 0].count()
>>>3704
</pre>
You can see that <b>total summary of errors is less than 20% of actual ratings, but only around 7% of all predictions are exactly correct.</b> To take a look at how these errors are spread, we'll group real ratings and compare the group sizes with group prediction success:
<pre>
ResultComparison = pd.DataFrame({'Count': ResultCatcher.groupby(['Real_Rating']).size(),
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'Avg_Rounded_Est': ResultCatcher.groupby(['Real_Rating'])['Estimation_Rounded'].mean()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}).reset_index()
</pre>
This code creates a dataframe with ratings on the left, adds their counts to the next column as 'Count' and inserts another column for the averages of rounded estimations.
<pre>
ResultComparison
</pre>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Real_Rating</th>
      <th>Count</th>
      <th>Avg_Rounded_Est</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.5</td>
      <td>124</td>
      <td>3.173387</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>390</td>
      <td>3.097436</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.5</td>
      <td>181</td>
      <td>3.055249</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>881</td>
      <td>3.230988</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.5</td>
      <td>574</td>
      <td>3.178571</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.0</td>
      <td>2677</td>
      <td>3.457041</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.5</td>
      <td>1512</td>
      <td>3.554233</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.0</td>
      <td>4357</td>
      <td>3.751434</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.5</td>
      <td>1175</td>
      <td>3.921277</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5.0</td>
      <td>2413</td>
      <td>4.015955</td>
    </tr>
  </tbody>
</table>
The BaselineOnly model seems to <b>work best with bigger rating counts</b> as they have better average rounded estimations. Of course, this is somewhat expected: The more skew the observations have, the more skewed the predictions will be.
<pre>
ResultCatcher.plot.scatter(x='Real_Rating', y='Estimation_Rounded', alpha=0.002, s=150, figsize=(10,10))
</pre>
<img src='https://github.com/EmirKorkutUnal/Python-Surprise-Predictions-on-Custom-Dataset/blob/master/Images/BaselineOnly_Scatterplot.png'>
The model had its best performance when movies with ratings between 3 to 4.5 are predicted.<br><br>
Let's go through the same process for the KNNBaseline model:
<pre>
predictions2 = KNNBaseline(n_epochs=5, lr_all=0.002, reg_all=0.4).fit(trainset).test(testset)
ResultCatcher2 = pd.DataFrame(predictions2, columns=['userId', 'movieId', 'Real_Rating', 'Estimated_Rating', 'details'])
ResultCatcher2.drop(['details'], axis=1, inplace=True)
ResultCatcher2['Estimation_Rounded'] = ResultCatcher2.apply(lambda row: halfrounder(row.Estimated_Rating), axis=1)
ResultCatcher2['Error'] = abs(ResultCatcher2['Real_Rating'] - ResultCatcher2['Estimation_Rounded'])
ResultCatcher2.head()
</pre>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>Real_Rating</th>
      <th>Estimated_Rating</th>
      <th>Estimation_Rounded</th>
      <th>Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>57</td>
      <td>1953</td>
      <td>5.0</td>
      <td>4.509060</td>
      <td>4.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>585</td>
      <td>3052</td>
      <td>4.0</td>
      <td>3.950890</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>205</td>
      <td>1210</td>
      <td>3.0</td>
      <td>3.652286</td>
      <td>3.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>120</td>
      <td>1721</td>
      <td>3.5</td>
      <td>3.545518</td>
      <td>3.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>547</td>
      <td>4239</td>
      <td>3.0</td>
      <td>3.426906</td>
      <td>3.5</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
<pre>
ResultCatcher2['Error'].sum()
>>>9143.0
ResultCatcher2['Real_Rating'][ResultCatcher2['Error'] == 0].count()
>>>3725
ResultComparison2 = pd.DataFrame({'Count': ResultCatcher2.groupby(['Real_Rating']).size(),
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'Avg_Rounded_Est': ResultCatcher2.groupby(['Real_Rating'])['Estimation_Rounded'].mean()
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}).reset_index()
ResultComparison2
</pre>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Real_Rating</th>
      <th>Count</th>
      <th>Avg_Rounded_Est</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.5</td>
      <td>124</td>
      <td>3.173387</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>390</td>
      <td>3.097436</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.5</td>
      <td>181</td>
      <td>3.055249</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>881</td>
      <td>3.230988</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.5</td>
      <td>574</td>
      <td>3.178571</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.0</td>
      <td>2677</td>
      <td>3.457041</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.5</td>
      <td>1512</td>
      <td>3.554233</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.0</td>
      <td>4357</td>
      <td>3.751434</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.5</td>
      <td>1175</td>
      <td>3.921277</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5.0</td>
      <td>2413</td>
      <td>4.015955</td>
    </tr>
  </tbody>
</table>
<pre>
ResultCatcher.plot.scatter(x='Real_Rating', y='Estimation_Rounded', alpha=0.002, s=150, figsize=(10,10))
</pre>
<img src='https://github.com/EmirKorkutUnal/Python-Surprise-Predictions-on-Custom-Dataset/blob/master/Images/KNNBaseline_Scatterplot.png'>
Results are similar to the first model, but <b>KNNBaseline performed slightly better</b>.<br><br>
Just to spice things up, you can check the summary of errors for the original database where small number of rating counts are not ruled out. The code is not provided here, though the analysis is done through the same process, the only difference is that the cleaning at the beginning is skipped. When using BaselineOnly, the summary of errors for the whole database 

