import random
import json
import torch
from model import NueroLine
from my_nltk_script import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)

# all keys, info to create model
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]


model = NueroLine(input_size, hidden_size, output_size)
model.to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "NueroLine"
print("Let's chat friend! type 'quit' to exit")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break

    sentence = tokenize(sentence)  # tokenize sentence
    # gets tokenized sentence and all words we got from saved file
    X = bag_of_words(sentence, all_words)

    # one sample , shape zero as number of columns
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)  # bag of words returns numpy array

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]  # class label the number

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")

    else:
        print(f"{bot_name}: I do not understand.........")
