import re

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

matches_data_file = "matches.csv"

# Function to calculate betting odds based on probability
def calculate_odds(probability):
    if probability == 0:
        return "N/A"
    return round(100 / probability, 4)

def clean_score(score_str):
    # Check if score_str is a valid string, otherwise return it as is
    if not isinstance(score_str, str):
        return score_str
    # Remove any content within parentheses
    cleaned_str = re.sub(r'\(.*?\)', '', score_str).strip()
    return cleaned_str

def parse_match_data(df):
    # Create new dataframe with required columns
    parsed_df = pd.DataFrame()
    
    # "League"
    parsed_df['League'] = df['league']
    
    # "Season_End_Year" (extracting last two digits of the season end year from "season")
    parsed_df['Season_End_Year'] = '20' + df['season'].str[2:4]
    parsed_df['Season_End_Year'] = parsed_df['Season_End_Year'].astype(int)
    
    
    # "Home" and "Away" (splitting the "game" column)
    parsed_df['Home'] = df['home_team']
    parsed_df['Away'] = df['away_team']
    
    print(df['score'].str.split('–').str[1])
    
    # Clean "score" column, removing parentheses and splitting at the dash
    df['cleaned_score'] = df['score'].apply(clean_score).str.split('–')
    
    # Extract HomeGoals and AwayGoals, converting to int if possible
    parsed_df['HomeGoals'] = df['cleaned_score'].str[0].apply(pd.to_numeric, errors='coerce')
    parsed_df['AwayGoals'] = df['cleaned_score'].str[1].apply(pd.to_numeric, errors='coerce')
    
    return parsed_df

# Function to plot primary statistics
def plot_primary_stats(season_stats):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].plot(season_stats.index, season_stats["AllGoals"], label="All Goals", color="blue")
    ax[0].set_title("Total Goals by Season")
    ax[0].set_xlabel("Season")
    ax[0].set_ylabel("Total Goals")
    ax[0].legend()

    ax[1].plot(season_stats.index, season_stats["AllConcededGoals"], label="Conceded Goals", color="red")
    ax[1].set_title("Total Conceded Goals by Season")
    ax[1].set_xlabel("Season")
    ax[1].set_ylabel("Total Conceded Goals")
    ax[1].legend()

    ax[2].plot(season_stats.index, season_stats["AllGoalsInvolved"], label="Average Goals Involved per Match", color="purple")
    ax[2].set_title("Average Goals Involved per Match by Season")
    ax[2].set_xlabel("Season")
    ax[2].set_ylabel("Average Goals per Match")
    ax[2].legend()

    st.pyplot(fig)

# Function to plot secondary statistics
def plot_secondary_stats(season_stats):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].plot(season_stats.index, season_stats["AvgGoals"], label="Avg Goals", color="blue")
    ax[0].plot(season_stats.index, season_stats["AvgConcededGoals"], label="Avg Conceded Goals", color="red")
    ax[0].set_title("Avg Goals vs Avg Conceded Goals by Season")
    ax[0].set_xlabel("Season")
    ax[0].set_ylabel("Average Goals")
    ax[0].legend()

    ax[1].plot(season_stats.index, season_stats["MinGoals"], label="Min Goals", color="green")
    ax[1].plot(season_stats.index, season_stats["MinConcededGoals"], label="Min Conceded Goals", color="orange")
    ax[1].set_title("Min Goals vs Min Conceded Goals by Season")
    ax[1].set_xlabel("Season")
    ax[1].set_ylabel("Minimum Goals")
    ax[1].legend()

    ax[2].plot(season_stats.index, season_stats["MaxGoals"], label="Max Goals", color="purple")
    ax[2].plot(season_stats.index, season_stats["MaxConcededGoals"], label="Max Conceded Goals", color="brown")
    ax[2].set_title("Max Goals vs Max Conceded Goals by Season")
    ax[2].set_xlabel("Season")
    ax[2].set_ylabel("Maximum Goals")
    ax[2].legend()

    st.pyplot(fig)

def calculate_season_stats(df, team_a, team_b, goal_threshold, team_goal_threshold):
    
    if team_a == None:
        matches = df[(df["Home"] == team_b) |
                    (df["Away"] == team_b)].copy()    
    elif team_b == None:
        matches = df[(df["Home"] == team_a) |
                    (df["Away"] == team_a)].copy()    
    else:
        matches = df[((df["Home"] == team_a) & (df["Away"] == team_b)) |
                    ((df["Home"] == team_b) & (df["Away"] == team_a))].copy()

    matches['TeamA'] = matches.apply(lambda row: team_a if row['Home'] == team_a or row['Away'] == team_a else team_b, axis=1)
    matches['TeamB'] = matches.apply(lambda row: team_b if row['TeamA'] == team_a else team_a, axis=1)
    matches['GoalsA'] = matches.apply(lambda row: row['HomeGoals'] if row['TeamA'] == row['Home'] else row['AwayGoals'], axis=1)
    matches['GoalsB'] = matches.apply(lambda row: row['AwayGoals'] if row['TeamA'] == row['Home'] else row['HomeGoals'], axis=1)
    matches['TotalGoals'] = matches['GoalsA'] + matches['GoalsB']
    
    matches_team_a_win = matches[matches['GoalsA'] > matches['GoalsB']]
    matches_team_b_win = matches[matches['GoalsA'] < matches['GoalsB']]
    matches_draw = matches[matches['GoalsA'] == matches['GoalsB']]

    st.write(matches)

    # Probability calculations
    total_matches = len(matches)
    
    if total_matches <= 0:
        return {}, pd.DataFrame()

    # Create conditions for cross-condition table
    conditions = {
        'TeamAWin': matches['GoalsA'] > matches['GoalsB'],
        'TeamBWin': matches['GoalsA'] < matches['GoalsB'],
        'Draw': matches['GoalsA'] == matches['GoalsB'],
        'TotalGoals > thresh': matches['TotalGoals'] > goal_threshold,
        'TotalGoals = thresh': matches['TotalGoals'] == goal_threshold,
        'TotalGoals < thresh': matches['TotalGoals'] < goal_threshold,
        'GoalsA > thresh': matches['GoalsA'] > team_goal_threshold,
        'GoalsA = thresh': matches['GoalsA'] == team_goal_threshold,
        'GoalsA < thresh': matches['GoalsA'] < team_goal_threshold,
        'GoalsB > thresh': matches['GoalsB'] > team_goal_threshold,
        'GoalsB = thresh': matches['GoalsB'] == team_goal_threshold,
        'GoalsB < thresh': matches['GoalsB'] < team_goal_threshold,
    }

    reversed_conditions = list(conditions.keys())[::-1]

    # Create an empty DataFrame to store the cross-condition probabilities
    prob_table = pd.DataFrame(index=reversed_conditions, columns=conditions.keys())

    # Calculate probabilities for each pair of conditions, but only for the upper triangle of the table
    for i, cond_a in enumerate(conditions.keys()):
        for j, cond_b in enumerate(conditions.keys()):
            if j > i:
                prob_table.loc[cond_a, cond_b] = ""
            else:
                combined_mask = conditions[cond_a] & conditions[cond_b]  # Match both conditions
                count_both_conditions = combined_mask.sum()

                # if cond_b == "TeamAWin" and cond_a == "GoalsA > thresh":
                #     print(df.loc[8536])
                #     print("count_both_conditions", combined_mask)
                
                probability = count_both_conditions / total_matches * 100  # Percentage of total matches
                prob_table.loc[cond_a, cond_b] = f"{calculate_odds(probability)} | {round(probability, 2)}%"

    # Return main statistics and the cross-condition table
    return {
        "total_matches": total_matches,
        
        "prob_team_a_win": len(matches_team_a_win) / total_matches * 100,
        "prob_team_b_win": len(matches_team_b_win) / total_matches * 100,
        "prob_draw": len(matches_draw) / total_matches * 100,

        "avg_total_goals_per_match": matches['TotalGoals'].mean(),
        "avg_team_a_goals_per_match": matches['GoalsA'].mean(),
        "avg_team_b_goals_per_match": matches['GoalsB'].mean(),

        "prob_goals_above_threshold": len(matches[matches['TotalGoals'] > goal_threshold]) / total_matches * 100,
        "prob_goals_equal_threshold": len(matches[matches['TotalGoals'] == goal_threshold]) / total_matches * 100,
        "prob_goals_below_threshold": len(matches[matches['TotalGoals'] < goal_threshold]) / total_matches * 100,

        "prob_team_a_goals_above_threshold": len(matches[matches['GoalsA'] > team_goal_threshold]) / total_matches * 100,
        "prob_team_a_goals_equal_threshold": len(matches[matches['GoalsA'] == team_goal_threshold]) / total_matches * 100,
        "prob_team_a_goals_below_threshold": len(matches[matches['GoalsA'] < team_goal_threshold]) / total_matches * 100,
        "prob_team_b_goals_above_threshold": len(matches[matches['GoalsB'] > team_goal_threshold]) / total_matches * 100,
        "prob_team_b_goals_equal_threshold": len(matches[matches['GoalsB'] == team_goal_threshold]) / total_matches * 100,
        "prob_team_b_goals_below_threshold": len(matches[matches['GoalsB'] < team_goal_threshold]) / total_matches * 100,
    }, prob_table


def display_statistics(statistics, prob_table):
    if len(statistics) <= 0:
        st.write("There are no matches.")
        return

    st.write("### Probability Statistics")
    
    st.write(f"**Total Matches:** {statistics['total_matches']}")

    st.write("#### Cross-Condition Probability Table")
    st.table(prob_table)

def calculate_most_frequent_results(df, team_a, team_b, top_n_results):
    matches = df[((df["Home"] == team_a) & (df["Away"] == team_b)) |
                 ((df["Home"] == team_b) & (df["Away"] == team_a))].copy()

    matches['TeamA'] = matches.apply(lambda row: team_a if row['Home'] == team_a or row['Away'] == team_a else team_b, axis=1)
    matches['TeamB'] = matches.apply(lambda row: team_b if row['TeamA'] == team_a else team_a, axis=1)
    matches['GoalsA'] = matches.apply(lambda row: row['HomeGoals'] if row['TeamA'] == row['Home'] else row['AwayGoals'], axis=1)
    matches['GoalsB'] = matches.apply(lambda row: row['AwayGoals'] if row['TeamA'] == row['Home'] else row['HomeGoals'], axis=1)
    matches['TotalGoals'] = matches['GoalsA'] + matches['GoalsB']
    
    # Create a result string (e.g., "2:1", "1:0") for each match
    matches["Result"] = matches.apply(lambda row: f"{row['GoalsA']}:{row['GoalsB']}", axis=1)

    # Count the most frequent results and return the top N results
    most_frequent_results = matches["Result"].value_counts().nlargest(top_n_results)

    # Return as a dictionary where the result is the key and frequency is the value
    return most_frequent_results.to_dict()


st.set_page_config(layout="wide")

# Main title
st.title("Head-to-Head Football Stats Calculator")

# Sidebar - Expected Value Calculator
st.sidebar.header("Expected Value Calculator")

# Input for win probability
win_rate = st.sidebar.number_input("Enter Win Rate (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

# Calculate odds based on the win rate
if win_rate > 0:
    odds = calculate_odds(win_rate)
    st.sidebar.write(f"For a win rate of {win_rate}%, the odds should be: {odds}")
else:
    st.sidebar.write("Please enter a valid win rate greater than 0.")

df = pd.read_csv(matches_data_file, dtype={'season': 'string'})
df = df.dropna(subset=['score'])
df = parse_match_data(df)

# Display available teams
st.subheader("Select Teams")
teams = sorted(pd.concat([df["Home"], df["Away"]]).unique())

team_a = st.selectbox("Select Team A", [None, *teams])
team_b = st.selectbox("Select Team B", [None, *teams])

# Select seasons range
st.subheader("Select Season Range")
min_season = int(df["Season_End_Year"].min())
max_season = int(df["Season_End_Year"].max())

season_range = st.slider("Select the season range", min_season, max_season, (min_season, max_season))

# Filter the dataframe based on the selected season range
df = df[(df["Season_End_Year"] >= season_range[0]) & (df["Season_End_Year"] <= season_range[1])]

# Input goal threshold
st.subheader("Set Parameters")
goal_threshold = st.number_input("Goal threshold", min_value=0.0, value=2.5, step=0.5)
team_goal_threshold = st.number_input("Team Goal threshold", min_value=0.0, value=1.5, step=0.5)
top_n_results = st.number_input("Top N results to display", min_value=1, value=10)


statistics, prob_table = calculate_season_stats(df, team_a, team_b, goal_threshold, team_goal_threshold)

# Dynamically display the statistics
display_statistics(statistics, prob_table)

# Display the most frequent results
st.subheader(f"Top {top_n_results} Most Frequent Results")
most_frequent_results = calculate_most_frequent_results(df, team_a, team_b, top_n_results)
for result, count in most_frequent_results.items():
    st.write(f"Result: {result} | Frequency: {count}")
