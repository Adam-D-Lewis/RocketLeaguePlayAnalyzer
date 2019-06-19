# RocketLeaguePlayAnalyzer

## Go to https://rocket-league-play-analzyer.herokuapp.com/ to see a live demo of this app!

### Introduction
The Rocket League Play Analyzer is a consumer facing web app built using <b>Python, Flask, Pandas, Scikit-Learn, Random Forest Classifiers, and K-Nearest Neighbors</b> to provide personalized training recommendations to accelerate gamer rank advancement using <b>20 GB</b> of performance data collected using a <b>web API</b>.

### App Summary
Rocket League is a popular e-sports video game.  Gameplay is essentially virtual soccer in rocket propelled cars.  Rocket league boasts over <b>57 million</b> registered players, and due to its popularity an entire ecosystem has arisen all focused around helping players level up in the game ranking system from <u><a href="https://www.youtube.com/watch?v=IgI6tiYl2_A">YouTube videos</a></u>, to <u><a href="https://www.gamersensei.com/games/rocket-league-coaching">coaching services</a></u>, to <u><a href="https://www.badpanda.gg/membership">paid training content</a></u>.

The Rocket League Play Analyzer analyzes an individual player's game data, identifies which skills will help that player rank up in the in-game ranking system fastest, and then recommends specific training resources to help players do so.

### Getting Data
The data was collected from <u><a href="https://calculated.gg">https://calculated.gg</a></u>, a site which allows players to upload their saved replays, and then will analyze the players stats compared to other players.  They have a lot of great features on their site, but not a training resource recommender.

The calculated.gg <b>web api</b> was used to collect every 3v3 Standard ranked match played from January 1 to April 9, 2019.  The total data included over 200,000 matches and was over <b>20 GB</b> in size.

### Cleaning Data
The raw data was in the form of 2 sets of json objects.  One with the metadata for the games, and another with the performance statistics for the match.  The data contained much unneeded information for this application.  The unneeded information was discarded, two pandas dataframes were created and then manipulated with user defined functions. Finally, the dataframes were merged with an inner join to produce a dataframe of performance statistics per player per game.

After cleaning the data, the data size was reduced to about 500 MB.  Something that could easily fit in memory, and eventually be deployed in the heroku platform.

### Feature Importance w/ Random Forest Classifiers
For this project, the recommended skill to work on should be both:
                <ol>
                  <li>a skill which will help the player rank up
                  <li>a skill which the player is deficient in
                </ol>

This section deals with #1 in the list above.  Relative feature importance can be used with some caveats to determine which skills are most needed in order to increase in rank.

Unfortunately, calculating the variance inflation factor (VIF) showed a high degree of collinearity amongst the features of data.  Collinearity can affect the relative feature importance.  In order to mitigate this I iteratively removed the feature with the highest VIF until all VIFs were below 5.  This left me with 13 remaining features.

<p>Now that only features with an acceptable amount of multicollinearity remain, I can proceed.  Rocket League ranks go from 1-19.  It is entirely possible and likely that different skills are going to be critical if a player is going from rank 1 to rank 2 than if they are going from rank 18 to rank 19.  In order to determine which skills are most important based on what class a player is currently I trained 18 <b>random forest binary classifiers</b> to predict a player’s rank based on their game stats.  Each individual classifier was trained with only a subset of the data and predicts whether a player is in rank i or i+1 only where i is an integer from 1-18.</p>

### Recognizing Skill Deficiencies w/ K-Nearest Neighbors
For this project, the recommended skill to work on should be both:
                <ol>
                  <li>a skill which will help the player rank up
                  <li>a skill which the player is deficient in
                </ol>
                
<p>This section deals with #2 in the list above.  There are various ways one might determine which skill the player is deficient in relative to the players at the rank he is trying to achieve.  One way might be to compare his average stats to the average stats of the group he is trying to achieve.  There could be players who rank up because they are really good at scoring while others could rank up because they are exceptional at stealing the ball from their opponents, as a simplified example.  While improving in both of these stats are good, we’re looking for which are the best skills to practice.</p>
              <p>This app compare a player’s average stats not to the entire population at the next rank, but rather only the part of the population most similar to the where the player already is by looking at the <b>k-nearest neighbors</b>.</p>
              <p>While precomputing the nearest neighbors for every player in my dataset would unnecessarily waste storage space, I did need to normalize the data and then store the dataset in a ball tree in order to readily find the nearest neighbors for any potential user of the app.  The k-nearest neighbors can readily be found from the ball tree dataset.</p>
  
 ### Balancing Important Skills and Deficient Skills
 <p>We now have the relative feature importance and the relative deficiencies of the player, but how to average these two priorities?  I went with a geometric average, because if a skill was not very important I didn’t want to recommend it to the player even if they are very “deficient” in that statistic.  There are other logical ways to average these 2 quantities, but geometric averaging seems like a good bet.</p>
 
 ### Web Server, Front End, and Deployment
   <p>Flask was used for the backend for the web app.  The front end of the app was built with bootstrap 4.  The app is deployed under heroku’s free tier.</p>