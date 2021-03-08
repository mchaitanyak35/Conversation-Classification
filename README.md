## Conversation Classification

Model is build to predict the call conversation between Agent and Customer in to predefined classes.

Following are the main steps in the solution:

### Data Analysis:
	1. The training data is provided as txt files for each conversation. In each txt file, each line is a segment of call, either the spoken dialouges by agent or customer, silence, laughter or noise. Also the segment time data is provided. As this is a classification problem, we do not need to consider who said what, so the dialouges spoken by either people can be concatenated and considered as a single text.
	2. The data set is imbalanced, the majority class has 63 samples and the minority class has just 6 samples.
	3. In the conversation text we have brace words, not all of them are useful like "[noise]", "[silence]"" etc. But some of them have required informartion like "[laughter-going]", "is['it]".
	4. The average length of words in conversations before removal of stopwords are 1322, after removal it is 517.
	5. As it is a phone conversation we can see words like um, uh, hum, huh which are not useful for classificaiton model.

### Data Preprocessing:
	1. The text data has brace words, some of them which are not needed are removed, and remaining are retained after filtering the braces.
	2. Some unwanted strings like <b_aside> and <e_aside> are also removed as they do not add any information to the classification model training.
	3. All other unwanted characters are removed and they are restitched in lower case.
	4. NLTK stopwords are removed and other found phone call specific words like uh, hum are also removed.
	5. Data is split in to 80\% training and 20\% testing.

### Tokenization:
	1. I have used pretrained glove word embedding for converting text into word vectors.
	2. The vocabulary size is taken as 5000 as the total number of words found are ~7200.
	3. The max length of each conversation is taken as 500, as the average length of texts is 517 after removal of stopwords.
	4. Embedding dimension is taken as 128, so imported a glove word embedding file of 200d.

### Imbalance:
	1. As the data is imbalanced, I have used SMOTE and Randomundersampler to adjust the distribution to reduce the imabalnce if the dataset.
	2. Minority classes Budget and Bank are oversampled with SMOTE and brought to count 40.
	3. Majority class Family is undersampled to 54 to further reduce the imbalance of the dataset.

### Bidirectional LSTM:
	1. As the textual data is a conversation between Agent and Customer, it will have useful contextual information with in it. This information can be leveraged by using Bidirectional LSTMs in classifying the conversations.
	2. Additional Bidirectional LSTM layers or Dense layers are tried but the boost in the performance is negligible.

### Performance:
	1. As the dataset is imbalanced, the correct measurements are precision, recall and f1 scores of minority classes instead of overall accuracy numbers. Following accuracy numbers are achieved: 

			           precision    recall  f1-score   support

		           0       0.80      0.57      0.67         7
		           1       0.67      0.57      0.62         7
		           2       1.00      1.00      1.00         7
		           3       1.00      1.00      1.00         8
		           4       0.94      1.00      0.97        15
		           5       0.87      1.00      0.93        13

		    accuracy                           0.89        57
		   macro avg       0.88      0.86      0.86        57
		weighted avg       0.89      0.89      0.89        57

### API:
	1. Training code is written in jupyter-notebook which are uploaded here.
	2. Same preprocessing and cleaning steps used while training are used in prediction step aswell.
	3. Prediction code is written in .py files, which are made accessible by a simple Flask service.
	4. This can hosted by running the .py file flask_service.py. The url configured is localhost:5000/predict.
	5. API taked in text blob of the conversation and returns the prediction.
	Examples: CURL command: curl -X GET http://127.0.0.1:5000/predict -d query='''text to be passed for prediction- CALL CONVERSATION'''
	6. As an alternative testing.py can be configured and run all the .txt files. For that .txt files needs to be placed in "./data/" folder.
	7. Models needs to be placed in "./models/". Model files are shared over the email.
	8. The dependencies are mentioned in requirements.txt file and is suggested to be installed and ran in python virtual environment.




