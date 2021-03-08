import os
import requests
# from IPython import embed

DIRECTORY = './data/'
mapping_file = './data/metadata/mapping_conv_topic.train.txt'
url = 'http://127.0.0.1:5000/predict'


def main():
    mapping_dict = {line.split()[0]: line.split()[1].replace('"', '')
                    for line in open(mapping_file, "r").read().split('\n') if line}

    for file in os.listdir(DIRECTORY):
        filepath = os.path.join(DIRECTORY, file)
        if os.path.isdir(filepath):
            continue

        text = open(filepath, 'r').read()
        params = {'query': text}

        response = requests.get(url, params)
        prediction = response.json()['prediction']
        print(file, ' : ', mapping_dict[file.split('.')[1]], ' : ', prediction)
        # break


if __name__ == '__main__':
    main()
