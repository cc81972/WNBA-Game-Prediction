import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


schedule = pd.read_csv('reg_season.csv')

schedule = schedule.iloc[:, 1:-2]
 
advanced_stats = pd.read_csv('advanced_stats.csv')

advanced_stats = advanced_stats.dropna(axis=1, how='all')

advanced_stats = advanced_stats.iloc[:, 1:-1]


df = pd.merge(schedule, advanced_stats, left_on="Visitor/Neutral", right_on="Team")
df = pd.merge(df, advanced_stats, left_on="Home/Neutral", right_on="Team")
df = df.drop(['Team_x', 'Team_y'], axis = 1)

for index, row in df.iterrows():
    # Determine which team had more points
    if df.loc[index, 'PTS'] > df.loc[index, 'PTS.1']:
        # Place 0 for home loss
        df.loc[index, 'Home_Winner'] = 0
    else:
        # Place 1 for home win
        df.loc[index, 'Home_Winner'] = 1

remove_cols = ["Home/Neutral", "Visitor/Neutral", "Home_Winner", "PTS", "PTS.1"]


selected_cols = [x for x in df.columns if x not in remove_cols]


###SciKit Learn minmax import
scalar = MinMaxScaler()
df[selected_cols] = scalar.fit_transform(df[selected_cols])

rr = RidgeClassifier(alpha=1.0)

sfs = SequentialFeatureSelector(rr, n_features_to_select=10, direction='backward')
###Loops through data and determines the most impactful columns in deteremining the winners
sfs.fit(df[selected_cols], df['Home_Winner'])   
predictors = list(df[selected_cols].columns[sfs.get_support()])

def monte_carlo(n):
    accuracy = []
    for i in range(n):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(df[predictors], df['Home_Winner'], test_size=0.2)

        # Train a logistic regression model on the training data
        model = LogisticRegression()
        # Fit the model to our training data
        model.fit(X_train, y_train)

        # Predict the winners for the test data
        y_pred = model.predict(X_test)

        # Evaluate the accuracy of the model on the test data
        accuracy.append(accuracy_score(y_test, y_pred))

    # Get the average accuracy
    score = sum(accuracy) / len(accuracy)
    return score

score = monte_carlo(1000)
print(f"Accuracy: {score}")

non_aces_liberty_game = df
non_aces_liberty_game = non_aces_liberty_game.drop(non_aces_liberty_game[(non_aces_liberty_game['Home/Neutral'] == 'Las Vegas Aces') & (non_aces_liberty_game['Visitor/Neutral'] == 'New York Liberty')].index)
non_aces_liberty_game = non_aces_liberty_game.drop(non_aces_liberty_game[(non_aces_liberty_game['Home/Neutral'] == 'New York Liberty') & (non_aces_liberty_game['Visitor/Neutral'] == 'Las Vegas Aces')].index)
non_aces_liberty_game[(non_aces_liberty_game['Home/Neutral'] == 'Las Vegas Aces') & (non_aces_liberty_game['Visitor/Neutral'] == 'New York Liberty')]

final_matchup = df[(df['Home/Neutral'] == 'Las Vegas Aces') & (df['Visitor/Neutral'] == 'New York Liberty')][:1]
# Show the predictors we will use
final_matchup[predictors]

model = LogisticRegression()
model.fit(non_aces_liberty_game[predictors], non_aces_liberty_game['Home_Winner'])

# Predict the outcome of the final_matchup
y_pred = model.predict(final_matchup[predictors])
print(f"Prediction: {y_pred[0]}")
