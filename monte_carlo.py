import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

PROJ_ROOT_DIR = r"C:\Users\adiad\Documents\GitHub\ncaa-tourney-predict"
PROJ_ROOT_DIR = PROJ_ROOT_DIR.replace("\\", "/") + "/"

def generator(
    odds_arr: np.ndarray,
    seed: int = None,
    print_bracket: bool = False,
    return_list: bool = True) -> list:
    '''This function will produce an entire tournament bracket.  A bracket can 
    be represented by printing to the terminal, as a python dictionary or in a nested-list
    structure.  The returned list structure mimics the input format required from a
    text file.

    Arguments:
    odds_arr -- an adjacency matrix providing the odds for the top seed winning with the

    Keyword arguments:
    top seed being the i-th row and the bottom seed being the j-th column
    seed -- the numpy seed used to generate random picks
    print_bracket -- whether the bracket should be written to the terminal
    return_list -- whether the returned object should be the nested-list structure or the 
    dictionary
    '''

    np.random.seed(seed)
    region_ls = ["Midwest", "East", "South", "West"]
    tourn_dict = {}
    tourn_ls = []
    final_four_dict = {}
    for r in region_ls:

        if print_bracket:
            print(f"Region {r} results:")

        round_dict = {}
        round_ls = ["First", "Second", "Sweet 16", "Elite 8"]
        round_size_ls = [64, 32, 16, 8]
        matchups_ls = [(ts,bs) for ts, bs in zip([1,8,5,4,6,3,7,2], [16,9,12,13,11,14,10,15])]
        for round, round_size in zip(round_ls, round_size_ls):

            round_str = f"{r[0]}{round_size}, "
            if print_bracket:
                print(f"{round} round")
                print("-------------------")

            victors_ls = []
            games_dict = {}
            for matchup in matchups_ls:
                ts = matchup[0]
                bs = matchup[1]
                result = np.random.rand()
                if result < odds_arr[ts-1,bs-1]:

                    if print_bracket:
                        print(f"Seed #{ts} beat #{bs}")

                    vs = ts
                    victors_ls.append(ts)

                else:
                    if print_bracket:
                        print(f"Seed #{ts} lost to #{bs}")

                    vs = bs
                    victors_ls.append(bs)
                
                games_dict[f"{ts} vs {bs}"] = vs
                games_dict["compact_str"] = f"{ts}v{bs}={vs}"
                round_str += f"{ts}v{bs}={vs}, "

            round_dict[f"{round} Round"] = games_dict
            tourn_ls += [round_str[:-2]]
            
            # rebuild the matchup list for the next round
            if round != round_ls[-1]:
                matchups_ls = []
                for i in range(0, len(victors_ls), 2):
                    if victors_ls[i] < victors_ls[i+1]:
                        matchups_ls.append((victors_ls[i], victors_ls[i+1]))
                    else:
                        matchups_ls.append((victors_ls[i+1], victors_ls[i]))
            
            if print_bracket:
                print("")

        final_four_dict[r] = victors_ls[0]
        tourn_dict[f"{r} Region"] = round_dict

    # Asses the East vs Midwest game
    round_str = "F4, "

    if print_bracket:        
        print("Final four round")
        print("-------------------")

    games_dict = {}
    east_final_game_dict_values = tourn_dict["East Region"]["Elite 8 Round"].values()
    east_seed = list(east_final_game_dict_values)[0]
    midwest_final_game_dict_values = tourn_dict["Midwest Region"]["Elite 8 Round"].values()
    midwest_seed = list(midwest_final_game_dict_values)[0]

    if east_seed == midwest_seed:
        ts = east_seed
        tr = "East"
        bs = midwest_seed
        br = "Midest"
        ts_odds = 0.5

    else:
        ts = min(east_seed, midwest_seed)
        tr = "East" if east_seed < midwest_seed else "Midwest"
        bs = max(east_seed, midwest_seed)
        br = "East" if east_seed > midwest_seed else "Midwest"
        ts_odds = odds_arr[ts-1,bs-1]

    result = np.random.rand()
    vs = ts if result <= ts_odds else bs
    vr = tr if result <= ts_odds else br

    if print_bracket:
        if result <= ts_odds:
            print(f"Seed #{ts} from the {tr} beat #{bs} from the {br}")
        
        else:
            print(f"Seed #{ts} from the {tr} lost to #{bs} from the {br}")

    games_dict[f"{tr} {ts} vs {br} {bs}"] = {
        "Victor Region": vr,
        "Victor Seed": vs,
        "compact_str": f"{tr[0]}{ts}v{br[0]}{bs}={vr[0]}{vs}"
    }

    round_str += f"{tr[0]}{ts}v{br[0]}{bs}={vr[0]}{vs}, "

    # Asses the West vs South game
    west_final_game_dict_values = tourn_dict["West Region"]["Elite 8 Round"].values()
    west_seed = list(west_final_game_dict_values)[0]
    south_final_game_dict_values = tourn_dict["South Region"]["Elite 8 Round"].values()
    south_seed = list(south_final_game_dict_values)[0]

    if west_seed == south_seed:
        ts = south_seed
        tr = "South"
        bs = west_seed
        br = "West"
        ts_odds = 0.5

    else:
        ts = min(west_seed, south_seed)
        tr = "West" if west_seed < south_seed else "South"
        bs = max(west_seed, south_seed)
        br = "West" if west_seed > south_seed else "South"
        ts_odds = odds_arr[ts-1,bs-1]

    result = np.random.rand()
    vs = ts if result <= ts_odds else bs
    vr = tr if result <= ts_odds else br

    if print_bracket:
        if result <= ts_odds:
            print(f"Seed #{ts} from the {tr} beat #{bs} from the {br}")
        
        else:
            print(f"Seed #{ts} from the {tr} lost to #{bs} from the {br}")

    games_dict[f"{tr} {ts} vs {br} {bs}"] = {
        "Victor Region": vr,
        "Victor Seed": vs,
        "compact_str": f"{tr[0]}{ts}v{br[0]}{bs}={vr[0]}{vs}"
    }

    round_str += f"{tr[0]}{ts}v{br[0]}{bs}={vr[0]}{vs}, "

    round_dict = {"Final Four Round": games_dict}
    tourn_ls += [round_str[:-2]]

    # Asses the championship game
    round_str = "F2, "

    if print_bracket:
        print()
        print("Championship round")
        print("-------------------")

    games_dict = {}
    east_midwest_seed = list(round_dict["Final Four Round"].values())[0]["Victor Seed"]
    east_midwest_region = list(round_dict["Final Four Round"].values())[0]["Victor Region"]
    west_south_seed = list(round_dict["Final Four Round"].values())[1]["Victor Seed"]
    west_south_region = list(round_dict["Final Four Round"].values())[1]["Victor Region"]

    if east_midwest_seed == west_south_seed:
        ts = east_midwest_seed
        tr = east_midwest_region
        bs = west_south_seed
        br = west_south_region
        ts_odds = 0.5

    else:
        ts = min(east_midwest_seed, west_south_seed)
        tr = "East" if east_midwest_seed < west_south_seed else "Midwest"
        bs = max(east_midwest_seed, west_south_seed)
        br = "East" if east_midwest_seed > west_south_seed else "Midwest"
        ts_odds = odds_arr[ts-1,bs-1]

    result = np.random.rand()
    vs = ts if result <= ts_odds else bs
    vr = tr if result <= ts_odds else br

    if print_bracket:
        if result <= ts_odds:
            print(f"Seed #{ts} from the {tr} beat #{bs} from the {br}")
        
        else:
            print(f"Seed #{ts} from the {tr} lost to #{bs} from the {br}")

    games_dict[f"{tr} {ts} vs {br} {bs}"] = {
        "Victor Region": vr,
        "Victor Seed": vs,
        "compact_str": f"{tr[0]}{ts}v{br[0]}{bs}={vr[0]}{vs}"
    }

    round_dict["Championship Round"] = games_dict
    tourn_dict["Multi-Region Playoff"] = round_dict
    round_str += f"{tr[0]}{ts}v{br[0]}{bs}={vr[0]}{vs}"
    tourn_ls += [round_str]

    if return_list:
        return tourn_ls
    else:
        return tourn_dict

def reader(fn: str) -> list:
    '''This function reads a csv file written in a specific convention to
    define an entire 64-team tournament.  See the example csv file for the
    expected conventions.
    '''

    f = open(fn, "r")
    Lines = f.readlines()

    region_ls = ["Midwest", "East", "South", "West", "Multi-Region Playoff"]
    region_lookup_ls = "MESWF"

    tourn_dict = {}
    tourn_ls = []
    for line in Lines:
        elements = line.strip().replace(" ", "").split(",") # Strips the newline character
        tourn_ls.append(elements)
        region_round_str = elements[0][0]
        region_ind = region_lookup_ls.index(region_round_str)
        region_str = region_ls[region_ind]

        games_ls = elements[1:]
        games_dict = {}
        for game_str in games_ls:

            seed_i, seed_j, seed_v = re.split("v|=", game_str)
            if region_str == "Multi-Region Playoff":
                region_i, seed_i = seed_i[0].upper(), int(seed_i[1:])
                region_ind = region_lookup_ls.index(region_i)
                region_i = region_ls[region_ind]
                region_j, seed_j = seed_j[0].upper(), int(seed_j[1:])
                region_ind = region_lookup_ls.index(region_j)
                region_j = region_ls[region_ind]

                vr_char, vs = seed_v[0].upper(), int(seed_v[1:])
                region_ind = region_lookup_ls.index(vr_char)
                vr = region_ls[region_ind]

                ts = min(seed_i, seed_j)
                bs = max(seed_i, seed_j)

                if ts == bs:
                    tr = region_i
                    br = region_j
                else:
                    seed_ls = [seed_i, seed_j]
                    tr = seed_ls.index(ts)
                    br = seed_ls.index(bs)

                games_dict[f"{tr} {ts} vs {br} {bs}"] = {
                    "Victor Region": vr,
                    "Victor Seed": vs
                }

            else:
                seed_i, seed_j, vs = int(seed_i), int(seed_j), int(seed_v)
                ts = min(seed_i, seed_j)
                bs = max(seed_i, seed_j)
                games_dict[f"{ts} vs {bs}"] = vs

    return tourn_ls

def scorer(true_bkt: list, pred_bkt: list, score_dict: dict) -> int:
    '''This function scores a predicted bracket against a true bracket
    according to the score dictionary.
    '''

    score = 0
    for true_line in true_bkt:
        round_str = true_line[0]
        round_pts_per_game = score_dict[int(round_str[1:])]
        for pred_line in pred_bkt:
            pred_line = pred_line.strip().replace(" ", "").split(",")
            if round_str == pred_line[0]:

                # get the victors from the game strings
                true_victors = set([game_str.split("=")[1] for game_str in true_line[1:]])
                pred_victors = set([game_str.split("=")[1] for game_str in pred_line[1:]])

                # calculate the score for the games in this predicted line
                n_correct = len(true_victors.intersection(pred_victors))
                score += n_correct*round_pts_per_game
    
    return score
    

def main():

    # set the number of samples to take in the monte carlo simulation
    n_samples = 1000
    
    # The dictionary needed for scoring a predicted bracket against
    # a true bracket. The key corresponds to the number of teams
    # participating in that round, the value corresponds to the
    # number of points given for a correct victor prediction in that
    # round.
    score_dict = {64: 1, 32: 2, 16: 4, 8: 8, 4: 16, 2: 32}

    ############################
    # CALCULATE THE ODDS OF TOP SEED WINNING FOR EACH SEED PAIR
    ############################

    # These files were downloaded from:
    # https://www.kaggle.com/c/mens-march-mania-2022/data
    tourn_df = pd.read_csv(PROJ_ROOT_DIR + "data/mens-march-mania-2022/MDataFiles_Stage1/MNCAATourneyCompactResults.csv")
    seed_df = pd.read_csv(PROJ_ROOT_DIR + "data/mens-march-mania-2022/MDataFiles_Stage1/MNCAATourneySeeds.csv")

    # drop play-in games
    tourn_df = tourn_df[tourn_df.DayNum >= 136]

    # limit data to last 10 years
    tourn_df = tourn_df[tourn_df.Season > 2011]

    # extract seeds as integers
    seed_df["SeedNum"] = seed_df.Seed.str[1:3].astype(int)

    # join winning seed numbers
    tourn_df = tourn_df.merge(seed_df, how="left", left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"])
    tourn_df = tourn_df.drop(["TeamID"], axis=1)
    tourn_df.rename(columns = {"SeedNum": "WSeedNum"}, inplace = True)

    # join losing seed numbers
    tourn_df = tourn_df.merge(seed_df.drop(["Seed"], axis=1), how="left", \
        left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"])
    tourn_df = tourn_df.drop(["TeamID"], axis=1)
    tourn_df.rename(columns = {"SeedNum": "LSeedNum"}, inplace = True)

    # drop games where seeds played against same seed from another region
    tourn_df = tourn_df[tourn_df.WSeedNum != tourn_df.LSeedNum]

    # add seed pair column
    tourn_df["SeedPair"] = tourn_df.apply(lambda row: \
        (min(row.WSeedNum, row.LSeedNum), max(row.WSeedNum, row.LSeedNum)), axis=1)

    # add top-seed-won boolean column
    tourn_df["TopSeedWon"] = tourn_df.WSeedNum < tourn_df.LSeedNum

    # calculate win percentage for top seed in all seed pairs
    top_seed_win_df = tourn_df[["SeedPair", "TopSeedWon"]].groupby("SeedPair").agg(["sum", "count"])
    top_seed_win_df.columns = top_seed_win_df.columns.get_level_values(1)
    top_seed_win_df["avg"] = top_seed_win_df["sum"] / top_seed_win_df["count"]

    # build an adjacency matrix
    adj_arr = 0.5*np.diag(np.diag(np.ones((16,16))))
    for r in range(16):
        for c in range(r + 1, 16):
            try:
                adj_arr[r, c] = top_seed_win_df.loc[top_seed_win_df.index == (r+1,c+1), "avg"].values[0]
            except:
                # this seed pair was never observed, so assume top seed always wins
                adj_arr[r, c] = 1

    # read ground truth bracket file
    true_bkt_fn = r"C:\Users\adiad\Documents\GitHub\ncaa-tourney-predict\2022_true_bracket.csv"
    true_bkt = reader(true_bkt_fn)

    score_ls = []
    # for _ in range(n_samples):
    for _ in tqdm(range(n_samples)):

        # generate a bracket using the odds in the adjacency matrix
        rand_bkt = generator(adj_arr)

        # score the predicted bracket
        score = scorer(true_bkt, rand_bkt, score_dict)
        score_ls.append(score)

    fig = go.Figure(
        data=[
            go.Histogram(
                x=score_ls,
                xbins={
                    "start": min(score_ls),
                    "end": max(score_ls),
                    "size": 1
                },
                histnorm="probability",
                cumulative_enabled=True
            )
        ],
        layout={
            "title": {
                "text": "Cumulative PDF for Predicted Bracket Scores",
                "x": 0.5
            },
            "xaxis": {"title": "Scores"},
            "yaxis": {"title": "Probability"},
        }
    )
    fig.show()

if __name__=="__main__":
    main()