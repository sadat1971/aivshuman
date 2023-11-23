import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import re
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score, precision_score
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import argparse


parser = argparse.ArgumentParser(description='Tell me what to print')

path = "/media2/sadat/Sadat/aivshuman/dataset/"
parser.add_argument("--LLM", type=str, default="13b", help="what is the LLM? chatgpt, 7b (llama2), 13b (llama)")
parser.add_argument("--version", type=str, default="none", help="what version do you want ?rephrase, expanded or summary_expanded ")
parser.add_argument("--dataset_path", type=str, default="/media2/sadat/Sadat/aivshuman/dataset/", help="path of the dataset?")
parser.add_argument("--result_path", type=str, default="/media2/sadat/Sadat/aivshuman/result/", help="path of the result?")

args = parser.parse_args()
## Machine Learning models:

class model:
    def __init__(self, train, test):
        ## Train and test should be a dictionary, where train["X"] should contain the features in numpy matrix format 
        ## mXn, where m is samples and n is features. the train["y"] should contain the labels in (m,) shape
        self.train = train
        self.test = test
        
    def performance_evaluation(self, GT, pred, modelname="Empty"):
        acc = accuracy_score(GT, pred)
        f1 = f1_score(GT, pred)
        precision = precision_score(GT, pred)
        recall = recall_score(GT, pred)
        print("----- modelname is {}------".format(modelname))
        print("accuracy is {:.4f} and f1 score is {:.4f}".format(acc, f1))
        print("precision is {:.4f} and recall is {:.4f}".format(precision, recall))
        return acc, f1
        
    def feature_importance(self, model):
        ## feat importance
        imp_acc = []
        for i in range(self.train["X"].shape[1]):
            feat = self.train["X"]
            feat = feat.transpose()
            np.random.shuffle(feat[i])
            feat = feat.transpose()
            train_pred = model.predict(feat)
            acc = accuracy_score(self.train["y"], train_pred)
            imp_acc.append(acc)
            if i%10==0:
                print("we are on ", str(i))
        df = pd.DataFrame()
        df["featname"] = ['f' + str(i) for i in range(self.train["X"].shape[1])]
        df["acc"] = imp_acc
        df = df.sort_values(by=['acc'], ascending=True)
        return df
        
    def naive_bayes(self, compute_feat_importance=False):
        df = pd.DataFrame()
        gnb = GaussianNB()
        y_pred = gnb.fit(self.train["X"], self.train["y"]).predict(self.test["X"])
        acc, f1 = self.performance_evaluation(self.test["y"], y_pred, "Gaussian NB")
        if compute_feat_importance:
            df = self.feature_importance(Optimized_model)
        return y_pred, df, acc, f1
    
    def SVM(self, cval_range=[-2,2,4], gammaval_range=[-2, 2, 4], tune=False, nfolds=1, 
            compute_feat_importance=False):
        df = pd.DataFrame()
        if tune==True:
            C = np.logspace(cval_range[0], cval_range[1], cval_range[2])
            gamma = np.logspace(gammaval_range[0], gammaval_range[1], gammaval_range[2])
            Param_tunable = {'C': C, 'gamma': gamma}
            Optimized_model = GridSearchCV(svm.SVC(kernel='rbf'), 
                                           Param_tunable, cv=nfolds, verbose = True, 
                                           n_jobs = -1)
        else:
            Optimized_model = svm.SVC(kernel='rbf')
        y_pred = Optimized_model.fit(self.train["X"], self.train["y"]).predict(self.test["X"])
        acc, f1 = self.performance_evaluation(self.test["y"], y_pred, "SVM")
        if compute_feat_importance:
            df = self.feature_importance(Optimized_model)
        return y_pred, df, acc, f1
    
    def random_forest(self, Estimators=[80, 100, 120], tune=False, nfolds=1, compute_feat_importance=False):
        df = pd.DataFrame()
        if tune==True:
            Param_tunable = {'n_estimators': Estimators}
            Optimized_model = GridSearchCV(RandomForestClassifier(), Param_tunable, 
                                           cv=nfolds, verbose = 1, n_jobs = -1)
            
        else:
            Optimized_model = RandomForestClassifier()
        y_pred = Optimized_model.fit(self.train["X"], self.train["y"]).predict(self.test["X"])
        acc, f1 = self.performance_evaluation(self.test["y"], y_pred, "Random Forest")
        if compute_feat_importance:
            df = self.feature_importance(Optimized_model)
        return y_pred, df, acc, f1
    
    def xgboost(self, compute_feat_importance=False):
        df = pd.DataFrame()
        model = XGBClassifier()
        y_pred = model.fit(self.train["X"], self.train["y"]).predict(self.test["X"])
        acc, f1 = self.performance_evaluation(self.test["y"], y_pred, "XGBoost")
        if compute_feat_importance:
            df = self.feature_importance(model)
        return y_pred, df, acc, f1
    
    def adaboost(self, Estimators=[80, 100, 120], tune=False, nfolds=1, compute_feat_importance=False):
        df = pd.DataFrame()
        if tune==True:
            Param_tunable = {'n_estimators': Estimators}
            Optimized_model = GridSearchCV(AdaBoostClassifier(random_state=42), Param_tunable, 
                                           cv=nfolds, verbose = 1, n_jobs = -1)
            
        else:
            Optimized_model = AdaBoostClassifier(random_state=42)
        y_pred = Optimized_model.fit(self.train["X"], self.train["y"]).predict(self.test["X"])
        acc, f1 = self.performance_evaluation(self.test["y"], y_pred, "Ada Boost")
        if compute_feat_importance:
            df = self.feature_importance(Optimized_model)
        return y_pred, df, acc, f1
    
    def logistic_regression(self, max_iter=100, compute_feat_importance=False):
        df = pd.DataFrame()
        model = LogisticRegression(random_state=0, max_iter=max_iter)
        y_pred = model.fit(self.train["X"], self.train["y"]).predict(self.test["X"])
        acc, f1= self.performance_evaluation(self.test["y"], y_pred, "loisitic regression")
        if compute_feat_importance:
            df = self.feature_importance(model)
        return y_pred, df, acc, f1



def prepare_data_with_doc2vec_vectors(path, split, LLM, version, train_or_test, model=None, 
                                      vector_size=100, window=5, min_count=2, workers=4, epochs=20):
    
    start = time.time()
    df = pd.read_pickle(path + train_or_test + "_split_" + str(split) + "_" + LLM + ".pkl") 
    df['human_tokenized'] = df['main'].apply(lambda x: word_tokenize(x.lower()))
    df['machine_tokenized'] = df[version].apply(lambda x: word_tokenize(x.lower()))
    print("\n... tokenization completed...\n")
    if train_or_test=="train":
        tagged_data_human = [TaggedDocument(words=row['human_tokenized'], tags=[str(i)]) for i, row in df.iterrows()]
        tagged_data_machine = [TaggedDocument(words=row['machine_tokenized'], tags=[str(i)]) for i, row in df.iterrows()]
        print("\n... tagged data building completed...\n")
        tagged_data = tagged_data_human + tagged_data_machine
        model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=20)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
        print("\n... model training completed...\n")

    organized_df = pd.DataFrame()
    organized_df["tokenized"] = df['human_tokenized'].tolist() + df["machine_tokenized"].tolist()
    organized_df["label"] = [0]*df.shape[0] + [1]*df.shape[0]
    if train_or_test=="train":
        organized_df = organized_df.sample(frac=1, random_state=42)
    organized_df["vectors"] = organized_df["tokenized"].apply(lambda x:model.infer_vector(x))
    print("\n... model inference completed...\n")
    organized_df.vectors = organized_df.vectors.apply(lambda x:x.reshape(1,vector_size))
    print(f"=====the time needed is {round(time.time()-start, 2)} seconds====")
    return organized_df, model



splits = [1,2,3]

SVM_acc = []
xgb_acc = []
logreg_acc = []
rf_acc = []

SVM_f1 = []
xgb_f1 = []
logreg_f1 = []
rf_f1 = []


for split in splits:
    trainset, trained_model = prepare_data_with_doc2vec_vectors(path=args.dataset_path, split=split, LLM=args.LLM, version=args.version,
                                                        train_or_test="train", model=None, vector_size=100, window=5, 
                                                        min_count=2, workers=4, epochs=20)
    testset, _ = prepare_data_with_doc2vec_vectors(path=args.dataset_path, split=split, LLM=args.LLM, version=args.version,
                                                        train_or_test="test", model=trained_model)
    trainset_X = np.vstack(trainset['vectors'].to_numpy())
    testset_X = np.vstack(testset['vectors'].to_numpy())

    ## Default
    train = dict()
    test = dict()
    train["X"] = trainset_X
    train["y"] = trainset.label.values
    test["X"] = testset_X
    test["y"] = testset.label.values
    ml = model(train, test)

    # Logreg
    _, _, acc, f1 = ml.logistic_regression(max_iter=100, compute_feat_importance=False)
    logreg_acc.append(acc)
    logreg_f1.append(f1)

    #SVM
    _, _, acc, f1 = ml.SVM()
    SVM_acc.append(acc)
    SVM_f1.append(f1)

    #XGBOOST
    _, _, acc, f1 = ml.xgboost(compute_feat_importance=False)
    xgb_acc.append(acc)
    xgb_f1.append(f1)

    ##RF
    _, _, acc, f1 = ml.random_forest()
    rf_acc.append(acc)
    rf_f1.append(f1)



df_result = pd.DataFrame()
df_result["splits"] = splits
df_result["logreg_acc"] = logreg_acc
df_result["logreg_f1"] = logreg_f1
df_result["SVM_acc"] = SVM_acc
df_result["SVM_f1"] = SVM_f1
df_result["xgb_acc"] = xgb_acc
df_result["xgb_f1"] = xgb_f1
df_result["rf_acc"] = rf_acc
df_result["rf_f1"] = rf_f1

df_result.to_csv(args.result_path + str(args.LLM) + "AI_" + args.version + "_vs_human_results.csv", index=False)




    
    

    
