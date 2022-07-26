from sklearn.model_selection import train_test_split
import pandas as pd



data = pd.read_json("./Data/Sarcasm_Headlines_Dataset_v2.json", lines=True).drop('article_link',axis=1)

train, test = train_test_split(data,test_size=0.25,stratify=data['is_sarcastic'],random_state=69)


print("\nBalanced for train set\n")
print(train.groupby('is_sarcastic').count())

print("---------------------------------------------------")

print("\nBalanced for test set\n")
print(test.groupby('is_sarcastic').count())

train.to_csv('./Data/train.csv',index=False)
test.to_csv('./Data/test.csv',index=False)




