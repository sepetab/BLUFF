import pandas as pd

datareader = pd.read_csv("Data/train.csv")
train_write = open("train.txt", "w")
train_write.close()
for index, row in datareader.iterrows():
    line_list = []
    line_list.append(str(row.get("is_sarcastic")))
    line_list.append(str(row.get("headline")))
    with open("train.txt", "a", encoding="utf-8") as train_write:
        train_write.write("__label__" + " ".join(line_list) + "\n")