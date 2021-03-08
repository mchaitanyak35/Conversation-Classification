import os
import re
import json
import pickle
import contractions
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
# from IPython import embed

DIRECTORY = './data/'
MODEL_DIR = './models/'
mapping_file = './data/metadata/mapping_conv_topic.train.txt'


def fetch_classification_model_files():
    """
    Function to read and load the models, tokenizer, config files

    Args:
        question_no : the question no like 'q2'

    Output:
        tokenizer, model, config_data
    """

    # loading saved keras tokenizer
    with open(os.path.join(MODEL_DIR, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)

    # loading classification_model
    model = load_model(os.path.join(MODEL_DIR, 'classification'))

    # loading config json
    with open(os.path.join(MODEL_DIR, 'config.json')) as f:
        config_data = json.load(f)

    return tokenizer, model, config_data


def fetch_clean_conversation(text):
    '''
    Function to read individual txt files, read line by line and clean the text for further processing.

    Args:
        filepath (path) : path to the individual training txt file
    Return:
        convo (str) : clean string of the individual training txt file
    '''
    # fetch the text words after the person name, start time and end time in each line
    convo = [' '.join(line.split()[3:]) for line in text.split('\n') if line]
    # removing unwanted brace words which are [silence], [noise], [vocalized-noice], [laughter], <b_aside> and <e_aside>
    convo = [re.sub(r'\[silence\]|\[noise\]|\[vocalized\-noise\]|\[laughter\]|_1|<b_aside>|<e_aside>', '', word)
             for line in convo for word in line.split()]
    # cleaning the brace words which has first words as laughter. Removing braces and laughter- to
    # retain only the second word
    convo = [re.sub(r'laughter\-', '', word) if re.findall(r'\[laughter\-.*\]', word) else word
             for word in convo]
    # removing unwanted braces from remaining words
    convo = [re.sub(r'\[|\]', '', word) for word in convo]
    # replacing '-' with spaces and stitching the words
    convo = ' '.join(' '.join([contractions.fix(re.sub(r'\-', ' ', word)).lower() for word in convo]).split())
    # remoaving all the non alphabetic characters and stitching back all the remaining text
    convo = ''.join([char if char.isalpha() else ' ' for char in convo])

    return convo


def predict(text):
    tokenizer, model, config_data = fetch_classification_model_files()
    mapping = config_data['mapping']

    text = fetch_clean_conversation(text)

    word_vector = tokenizer.texts_to_sequences([text])
    word_vector = pad_sequences(word_vector, padding='post', maxlen=config_data['maxlen'])

    prediction = model.predict(word_vector)
    prediction = mapping[str(np.argmax(prediction))]
    print('prediction : ', prediction)

    return prediction


if __name__ == '__main__':
    mapping_dict = {line.split()[0]: line.split()[1].replace('"', '')
                    for line in open(mapping_file, "r").read().split('\n') if line}

    for file in os.listdir(DIRECTORY):
        filepath = os.path.join(DIRECTORY, file)
        if os.path.isdir(filepath):
            continue
        prediction = predict(fetch_clean_conversation(filepath))
        print(file, ' : ', mapping_dict[file.split('.')[1]], ' : ', prediction)
        # break
