import pandas as pd
import os

# ── Config ────────────────────────────────────────────────────────────────────

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBP_PATH = os.path.join(BASE_DIR, "data", "raw", "77", "nba_pbp_data_selenium_final.csv")
BOX_PATH = os.path.join(BASE_DIR, "data", "raw", "77", "nba_box_scores_selenium_final.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "games.csv")

SEASONS = [2020]  # TODO: fill in the seasons you want to keep e.g. [2020, 2021, 2022]
SEPARATOR = "[END OF GAME]"

# ── Step 1: Explore ───────────────────────────────────────────────────────────

def explore(pbp, box):
    # TODO: print columns, dtypes, and a few rows from each dataframe
    # figure out what the game ID column is called in each
    print("=== PBP ===")
    print(pbp.shape)        # (rows, columns) — tells you how big the dataset is
    print(pbp.columns)      # column names
    print(pbp.dtypes)       # data type of each column
    print(pbp.head(10))     # first 10 rows so you can see what the data looks like

    print("=== BOX ===")
    print(box.shape)
    print(box.columns)
    print(box.dtypes)
    print(box.head(10))


# ── Step 2: Filter by season ──────────────────────────────────────────────────

def filter_seasons(pbp, box):
    # TODO: keep only rows where season is in SEASONS list
    # return filtered pbp and box dataframes

    # If game ID looks like 199611010BOS
    pbp['year'] = pbp['game_id'].str[:4].astype(int)
    box['year'] = box['game_id'].str[:4].astype(int)

    # Then filter
    pbp = pbp[pbp['year'].isin(SEASONS)]
    box = box[box['year'].isin(SEASONS)]

    return pbp, box

    
# ── Step 3: Match PBP to box scores ──────────────────────────────────────────

def match_games(pbp, box):
    # TODO: group pbp by game ID
    shared_ids = set(pbp['game_id'].unique()) & set(box['game_id'].unique())
    pbp_matched = pbp[pbp['game_id'].isin(shared_ids)]
    box_matched = box[box['game_id'].isin(shared_ids)]
    return pbp_matched, box_matched

def format_box_row(row):
    return f"{row['Player']} | {row['PTS']}pts | {row['TRB']}reb | {row['AST']}ast | {row['FG']}-{row['FGA']} FG"

# ── Step 4: Format each game as a single text sequence ───────────────────────

def format_game(pbp_rows, box_rows):
    # TODO: join pbp events into a single string, one play per line
    # add SEPARATOR

    pbp_running_str = "\n".join(pbp_rows['event'])
    # join box score rows into a single string below the separator
    box_running_str = "\n".join(box_rows.apply(format_box_row, axis=1))
    # return the full sequence as one string
    return pbp_running_str + "\n" + SEPARATOR + "\n" + box_running_str



# ── Step 5: Save ──────────────────────────────────────────────────────────────

def save(games):
    # TODO: make sure output directory exists
    # save the list of formatted game strings to OUTPUT_PATH

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for game in games:
            f.write(game)
            f.write("\n[NEW GAME]\n") 

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # print("Loading sample for exploration...")
    pbp_sample = pd.read_csv(PBP_PATH, nrows=10000)
    box_sample = pd.read_csv(BOX_PATH, nrows=1000)
    explore(pbp_sample, box_sample)

    print("Loading full dataset...")
    pbp = pd.read_csv(PBP_PATH)
    box = pd.read_csv(BOX_PATH)

    print("Filtering seasons...")
    pbp, box = filter_seasons(pbp, box)

    print("Matching games...")
    pbp_matched, box_matched = match_games(pbp, box)
    

    print("Formatting games...")
    games = []

    print("Formatting games...")
    games = []
    for game_id, pbp_group in pbp_matched.groupby('game_id'):
        box_group = box_matched[box_matched['game_id'] == game_id]
        games.append(format_game(pbp_group, box_group))

    print(f"Total games formatted: {len(games)}")
    print(games[0])  

    print(f"Saving {len(games)} games...")
    save(games)
    print("Done.")

if __name__ == "__main__":
    main()