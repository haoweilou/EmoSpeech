import pandas as pd 
dataframe = pd.read_csv("esd.csv",sep="\t")

random_seed = 42

# Create empty lists to store train and test data
train_list = []
test_list = []

# Group by speaker and emotion
grouped = dataframe.groupby(["speaker", "emotion"])

# Process each group
for (speaker, emotion), group in grouped:
    group = group.sample(frac=1, random_state=random_seed).reset_index(drop=True)  # Shuffle
    
    # Select 300 for train, 50 for test
    train_samples = group.iloc[:340]
    test_samples = group.iloc[340:350]

    # train_samples = group.iloc[:10]
    # test_samples = group.iloc[10:15]
    # Append to respective lists
    train_list.append(train_samples)
    test_list.append(test_samples)

# Concatenate the lists into DataFrames
train_df = pd.concat(train_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)

# Save the train and test sets
train_df.to_csv("train_set.csv", index=False)
test_df.to_csv("test_set.csv", index=False)

# Print dataset statistics
print(f"Train set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")