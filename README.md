# socialmedia prediction
## [Course project](https://www.kaggle.com/c/ift6758-a20/overview)
__Authors: Wei Wei, Yinghan Gao, Milan Mao, Miao Feng__

### Task
The main task for the data challenge is to predict the number of 'likes' for given details about the simulated profiles of users on social media.

You have been provided with various attributes about the visual aspects of the users' social media profiles, the basic details, the profile pictures used in the simulation. With the is information, you need to predict how many 'likes' the users could've likely received.

There is a CSV file of various collected features that has been provided, in addition to images showing the profile pictures of the users (based on the simulation)

### Requirements
* Python 3 (code has been tested on Python 3.6.5)
* xgboost (tested on xgboost 0.90)
* tensorflow (2.0.0), keras(2.3.1) (tested with cuda 10.0)
* Python packages (not exhaustive): pandas, sklearn, numpy, mlxtend, joblib 

### Structure:

* ```data/```: dataset and prediction file
* ```cleaner/```: feature engineering, including models for processing profile images. File clean.py is used to clean data and building features. File loader.py is used to call clean.py. 
* ```models/```: different models for predicting target, including fine tuning tool. Supported model: xgboost, mlp, and rf. 
* ```savedModel/```: models saved for reproducing
* ```main.py```: main script
* ```util.py```: codes of utilities.


### Training and Testing
To train model, call:

```python main.py -t train``` (```-t, --task``` allows  you to train a model and saved as xgboost-current.joblib)

To test or predict by a pretrained model, call:
```python main.py -m xgboost-1``` (```-m, --model_name``` allows you to specify a pretrained model name and do the prediction. Predictions will be saved as ```submission.csv``` under ```data/```)

### Pretrained Models

* The fitted model by call training: xgboost-current.joblib
* The best model: xgbost-1.joblib