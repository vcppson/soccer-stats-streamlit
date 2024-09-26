import soccerdata as sd

def generate_seasons_list(start_season: str, end_season: str) -> list:
    # Convert start and end season to integers
    start_year = int(start_season[:2])
    end_year = int(end_season[:2])
    
    # List to store the seasons
    seasons = []

    # Loop to generate seasons list
    for year in range(start_year, end_year + 1):
        # Format year and the following year as a string (e.g., "0001")
        season = f'{year:02d}-{(year + 1) % 100:02d}'
        seasons.append(season)

    return seasons

start_season = "0001"
end_season = "2425"
seasons_list = generate_seasons_list(start_season, end_season)

leagues = ["Premier League", "Champions League", "EFL League One", "EFL League Two", "Serie A", "Serie B", "EFL Championship", "Europa-League", "Europa Conference League"]

# five38 = sd.FiveThirtyEight(leagues=leagues, seasons=seasons_list)

# df = five38.read_games()

fbref = sd.FBref(leagues=leagues, seasons=seasons_list)

df = fbref.read_schedule()

df.to_csv("matches.csv")
