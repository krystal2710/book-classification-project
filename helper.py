from bs4 import BeautifulSoup
import requests
import numpy as np

HEADER = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15'}

def get_genres(url, headers = HEADER):
    r = requests.get(url, headers = headers)
    bs = BeautifulSoup(r.text, features="lxml")
    try: 
        genre_element = bs.find('div',{'data-testid':'genresList'}).find_all('a')
        genres_list = [element.text for element in genre_element]
    except: 
        genres_list = np.nan
    return genres_list


def get_weighted_genre(book_id, book_genres, headers = HEADER):
    
    if not isinstance(book_genres,list):
        return np.nan
    
    url = 'https://www.goodreads.com/work/shelves/' + book_id
    book_genres_dict = {genre.lower().replace(' ','-'):genre for genre in book_genres}

    r = requests.get(url, headers=headers)
    bs = BeautifulSoup(r.text, features="lxml")
    all_shelves = bs.find_all('div',{'class':'shelfStat'})

    current_book_dict = {}
    for shelf in all_shelves:
        shelf_text_list = shelf.text.split()
        genre = shelf_text_list[0]

        if genre in book_genres_dict:
            current_book_dict[genre] = shelf_text_list[1]
    return current_book_dict


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def max_frequency(L):
    '''Given a list returns the item with highest frequency'''
    count_dict = {}
    for label in L:
        if label in count_dict:
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    max_label = L[0]
    for label in count_dict:
        if count_dict[label] > count_dict[max_label]:
            max_label = label
    return max_label


    
def cross_val_baseline(X_train, y_train, cv = 10):
    skfolds = StratifiedKFold(n_splits = cv)
    accuracy_list = []
    f1_score_list = []
    for train_index, test_index in skfolds.split(X_train, y_train):
        y_test_fold = y_train[test_index]
        max_label = max_frequency(y_test_fold)
        y_pred = [max_label for i in range(len(y_test_fold))]
        accuracy = sum(y_pred == y_test_fold)/len(y_pred)
        accuracy_list.append(accuracy)
        f1_score_list.append(f1_score(y_test_fold, y_pred))
    return {'mean_accuracy': round(sum(accuracy_list)/len(accuracy_list),2), 'mean_f1': round(sum(f1_score_list)/len(accuracy_list),2)}

from tensorflow.keras import backend as K

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def run_nn(X_train, y_train, X_val, y_val, num_features):
    tf.random.set_seed(42) 

    model = Sequential()
    model.add(Dense(10, input_shape = (num_features,), activation = 'relu'))
    model.add(Dense(5, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', get_f1])
    
    early_stopping = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)

    #batch_size = all training samples
    history = model.fit(X_train,y_train,epochs = 5000, batch_size = len(X_train), validation_data=[X_val, y_val], verbose = 0, callbacks=[early_stopping])

    val_loss = round(history.history['val_loss'][-1],3)
    val_accuracy = round(history.history['val_accuracy'][-1],3)
    val_f1 = round(history.history['val_get_f1'][-1],3)
    num_epochs = len(history.history['val_accuracy'])

    return val_loss, val_accuracy, val_f1, num_epochs