import json
from my_nltk_script import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NueroLine
from torch.multiprocessing import freeze_support

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []  # need to collect all the words
tags = []
xy = []  # will hold patterns and tags

# want to loop through all sentence in intents patterns
for intent in intents['intents']:  # with key intents
    tag = intent['tag']
    tags.append(tag)
    # now loop through patterns
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        # extend not append, because it is an array , don't want an array of arrays
        all_words.extend(w)
        # will know pattern and corresponding tag, add xy pair
        xy.append((w, tag))

# NOW after tokenization want to lower, stem , and remove punctuations
ignore_words = ['?', '!', '.', ',']


# stemming - all lower case and remove punctuations
all_words = [stem(w) for w in all_words if w not in ignore_words]
# sort the words we only want unique words, so we convert it to set, so it removes duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))  # all uniqeue labels , no duplicates

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# NOW we need to train the data - (bag of words)
# list of x-train data
X_train = []  # empty list , put bag of words
y_train = []

# loop throug xy array
for (pattern_sentence, tag) in xy:
    # already tokenized the pattern sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)  # will give us labels for all the tags
    y_train.append(label)  # CrossEntropyLoss,

X_train = np.array(X_train)
y_train = np.array(y_train)

# pytorch data set and create data loader.


# Hyper paramaters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])  # first bag of words
learning_rate = 0.001
num_epochs = 1000
print(input_size, output_size)


class ChatDataset(Dataset):
    def __init__(self):
        self.number_of_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing , such as dataset[i] can be used and we retrieve the i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.number_of_samples  # return size of data set


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                          shuffle=True, num_workers=0)  # num workers for multi threading , won't be doing on my computer, could set to 2 for multi threads!

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NueroLine(input_size, hidden_size, output_size)
model.to(device)

# loss and optimizer in pipeline
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# TRAIN THE MODEL
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device).long()

        # forwards pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward pass and optimizer step , empty gradients
        optimizer.zero_grad()
        loss.backward()  # calculate back propagation
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch +1}/{num_epochs}, loss={loss.item():.4f}')

# loss should decrease every epoch , loss succesfully decreased with each epoch , means our example patterns are not very complex , but our nueral net is very good for this purpouse.
print(f'final loss, loss= {loss.item():.4f}')


# Save and load model and implement the chat bot

# save data

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"  # for pytorch
torch.save(data, FILE)  # saves file in pkl format

print(f'training complete. file saved to {FILE}')
