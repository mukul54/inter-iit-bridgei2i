import pandas as pd
import numpy as np
import translation
import englishhinglish
from multiprocessing import Pool

import time

def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def translator(df):
    df["Translated"] = df.Tweet.apply(translation.engHin2Eng)
    return df

if __name__ == "__main__":
    PATH = "../Data/Evaluation_Data/articles_eval.csv"

    # df = pd.read_excel(PATH)
    df = pd.read_csv(PATH)
    print(df.columns)
    # print(translation.engHin2Eng("Mei gareeb hu !! Chill ha"))
    # df["Translated"] =df.Tweet.apply(translation.engHin2Eng)
    # df["Translated"] = df.Text.apply(translation.endHin2Eng)

    # df = parallelize_dataframe(df,translator)
    start_time = time.time()
    # df["Translated"] = df.Tweet.apply(translation.engHin2Eng)
    df["Translated"] = df.Text.apply(translation.engHin2Eng)
    end_time = time.time()
    print(end_time - start_time)
    df.to_csv("OUTPUT_articles.csv")


