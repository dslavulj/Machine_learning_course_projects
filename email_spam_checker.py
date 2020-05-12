import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC

def make_dictionary(train_dir):
    
    email = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in email:
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words
                    
    dictionary = Counter(all_words)
    
    list_to_remove = dictionary.keys()
    keys_to_remove = []
    for k in list_to_remove:
        if k.isalpha() == False:
            keys_to_remove.append(k)
        elif len(k) == 1:
            keys_to_remove.append(k)
    for item in keys_to_remove:
        del dictionary[item]
        
    dictonary = dictionary.most_common(3000)
    
    return dictionary


def extract_features(mail_dir):
    
    files = [os.path.join(mail_dir, file_descriptor) for file_descriptor in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    doc_id = 0
    for file_to_read in files:
        with open(file_to_read) as file_descriptor:
            for i, line in enumerate(file_descriptor):
                if i == 2:
                    words = line.split()
                    for word in words:
                        word_id = 0
                        for i, d in enumerate(dictionary):
                            if d[0] == word:
                                word_id = i
                                if i >= 3000:
                                    break
                                features_matrix[doc_id, word_id] = words.count(word)
            doc_id = doc_id + 1
    return features_matrix

print("process training ...")
train_dir = 'train-mail'
dictionary = make_dictionary(train_dir)

print("\nextracting...")
train_labels = np.zeros(702)
train_labels[351:701] = 1
train_matrix = extract_features(train_dir)

print("model training...")
multinomia_model = MultinomialNB()
svc_model = LinearSVC()
multinomia_model.fit(train_matrix, train_labels)
svc_model.fit(train_matrix, train_labels)

test_dir = 'test-mail'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1
result1 = multinomia_model.predict(test_matrix)
result2 = svc_model.predict(test_matrix)

print("0 = not spam, 1 = spam")
print(result1)
print(result2)
