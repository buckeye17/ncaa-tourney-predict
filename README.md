# Overview
This project is an initial attempt to predict a March Madness bracket based solely on past tournament data.  Specifically, the odds of the top seed winning a game were calculated for every pair of seeds that have played each other in the last ten years.  These odds are the only only consideration when generating a bracket.  

The data used for these predictions was obtained from Kaggle.  It provides plenty of additional data is available to improve the predictions.  Here's the link to the data:
https://www.kaggle.com/c/mens-march-mania-2022/data

# Generating a Single Bracket
This can be accomplished by running the `monte_carlo.py` script and setting line 343 to `n_samples = 1` and line 415 to `rand_bkt = generator(adj_arr, print_bracket=True, east_vs="south")`.  Note, apparently the four regions of the tournament will play against different regions in the Final Four each year.  In order to produce a valid bracket for a given year you will need to define which region the East Region will play against in the Final Four.  The command above indicates that the East Region winner will play the South Region winner.

# Generating Monte Carlo Similation
This can be accomplished by running the `monte_carlo.py` script and setting line 343 to `n_samples = 1000` or whatever sample size you desire.  Also set line 415 to `rand_bkt = generator(adj_arr, print_bracket=False, east_vs="south")` in order avoid generating overwhelming terminal output and to avoid slowing the simulation down.