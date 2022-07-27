import pandas as pd
#import fasttext

#Writing train to sarcasm.train
datareader = pd.read_csv("Data/train.csv")
train_write = open("sarcasm.train", "w")
train_write.close()
for index, row in datareader.iterrows():
    line_list = []
    line_list.append(str(row.get("is_sarcastic")))
    line_list.append(str(row.get("headline")))
    with open("sarcasm.train", "a", encoding="utf-8") as train_write:
        train_write.write("__label__" + " ".join(line_list) + "\n")

#Writing test to sarcasm.test
datareader = pd.read_csv("Data/test.csv")
test_write = open("sarcasm.test", "w")
test_write.close()
for index, row in datareader.iterrows():
    line_list = []
    line_list.append(str(row.get("is_sarcastic")))
    line_list.append(str(row.get("headline")))
    with open("sarcasm.test", "a", encoding="utf-8") as test_write:
        test_write.write("__label__" + " ".join(line_list) + "\n")

#model = fasttext.train_supervised(input="sarcasm.train")
#model.test("sarcasm.test")