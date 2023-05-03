from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

def random_forest_model_tuning_depth(X_train_smote, Y_train_smote,x_train,y_train,x_cv,y_cv,x_test,y_test):
    
    depth = [3,5,10,20,50,100]
    cv_f1_score = []
    train_f1_score = []
    for i in depth:
        model = RandomForestClassifier(max_depth = i,n_jobs = -1,n_estimators = 10)
        model.fit(X_train_smote, Y_train_smote)
        CV = CalibratedClassifierCV(model,method = 'sigmoid')
        CV.fit(X_train_smote, Y_train_smote)
        predicted = CV.predict(x_cv)
        train_predicted = CV.predict(x_train)
        cv_f1_score.append(f1_score(y_cv,predicted,average = 'macro'))
        train_f1_score.append(f1_score(y_train,train_predicted,average = 'macro'))
        print('depth {0} is finished'.format(i))
    for i in range(0,len(cv_f1_score)):
        print('f1 value score for depth =' + str(depth[i]) + ' is ' + str(cv_f1_score[i]))
    plt.plot(depth,cv_f1_score,c='r')
    plt.plot(depth,train_f1_score,c='b')
    plt.xlabel('depth(depth of the tree)')
    plt.ylabel('f1 score(train and test)')
    plt.title('depth vs df1 score')
    for i,score in enumerate(cv_f1_score):
        plt.annotate((depth[i],np.round(score,4)),(depth[i],np.round(cv_f1_score[i],4)))
    for i,score1 in enumerate(train_f1_score):
        plt.annotate((depth[i],np.round(score1,4)),(depth[i],np.round(train_f1_score[i],4)))
    index = cv_f1_score.index(max(cv_f1_score))
    best_dpt = depth[index]
    print('best max depth is ' + str(best_dpt))
    model = RandomForestClassifier(max_depth = best_dpt,n_jobs = -1)
    model.fit(X_train_smote, Y_train_smote)
    predict_train = model.predict(x_train)
    print('f1 score on train data ' + str(f1_score(y_train,predict_train,average = 'macro')))
    train_mat = confusion_matrix(y_train,predict_train)
    predict_cv = model.predict(x_cv)
    print('f1 score on cv data ' + str(f1_score(y_cv,predict_cv,average = 'macro')))
    cv_mat = confusion_matrix(y_cv,predict_cv)
    predict_test = model.predict(x_test)
    print('f1 score on test data ' + str(f1_score(y_test,predict_test,average = 'macro')))
    test_mat = confusion_matrix(y_test,predict_test)
    fig,ax = plt.subplots(1,3,figsize = (15,5))
    sns.heatmap(ax = ax[0],data = train_mat,annot=True,fmt='g',cmap="YlGnBu")
    ax[0].set_xlabel('predicted')
    ax[0].set_ylabel('actual')
    ax[0].title.set_text('confusion matrix for train data')
    sns.heatmap(ax = ax[1],data = cv_mat,annot=True,fmt='g')
    ax[1].set_xlabel('predicted')
    ax[1].set_ylabel('actual')
    ax[1].title.set_text('confusion matrix for CV data')
    sns.heatmap(ax = ax[2],data = test_mat,annot=True,fmt='g')
    ax[2].set_xlabel('predicted')
    ax[2].set_ylabel('actual')
    ax[2].title.set_text('confusion matrix for test data')


def random_forest_model_tuning_estimator(X_train_smote, Y_train_smote,x_train,y_train,x_cv,y_cv,x_test,y_test):
    estimators = [10,20,50,100,200]
    cv_f1_score = []
    train_f1_score = []
    for i in estimators:
        model = RandomForestClassifier(n_estimators = i,max_depth = 50,n_jobs = -1)
        model.fit(X_train_smote, Y_train_smote)
        CV = CalibratedClassifierCV(model,method = 'sigmoid')
        CV.fit(X_train_smote, Y_train_smote)
        predicted = CV.predict(x_cv)
        train_predicted = CV.predict(x_train)
        cv_f1_score.append(f1_score(y_cv,predicted,average = 'macro'))
        train_f1_score.append(f1_score(y_train,train_predicted,average = 'macro'))
        print('Estimator {0} is finished'.format(i))
    for i in range(0,len(cv_f1_score)):
        print('f1 value score n_estimators =' + str(estimators[i]) + ' is ' + str(cv_f1_score[i]))
    plt.plot(estimators,cv_f1_score,c='r')
    plt.plot(estimators,train_f1_score,c='b')
    plt.xlabel('estimators(n_estimators)')
    plt.ylabel('f1 score(train and test)')
    plt.title('estimators vs df1 score')
    for i,score in enumerate(cv_f1_score):
        plt.annotate((estimators[i],np.round(score,4)),(estimators[i],np.round(cv_f1_score[i],4)))
    for i,score1 in enumerate(train_f1_score):
        plt.annotate((estimators[i],np.round(score1,4)),(estimators[i],np.round(train_f1_score[i],4)))
    index = cv_f1_score.index(max(cv_f1_score))
    best_est = estimators[index]
    print('best estimator is ' + str(best_est))
    model = RandomForestClassifier(n_estimators = best_est,max_depth = 50,n_jobs = -1)
    model.fit(X_train_smote, Y_train_smote)
    predict_train = model.predict(X_train_smote)
    print('f1 score on train data ' + str(f1_score(Y_train_smote,predict_train,average = 'macro')))
    train_mat = confusion_matrix(Y_train_smote,predict_train)
    predict_cv = model.predict(x_cv)
    print('f1 score on cv data ' + str(f1_score(y_cv,predict_cv,average = 'macro')))
    cv_mat = confusion_matrix(y_cv,predict_cv)
    predict_test = model.predict(x_test)
    print('f1 score on test data ' + str(f1_score(y_test,predict_test,average = 'macro')))
    test_mat = confusion_matrix(y_test,predict_test)
    fig,ax = plt.subplots(1,3,figsize = (15,5))
    sns.heatmap(ax = ax[0],data = train_mat,annot=True,fmt='g',cmap="YlGnBu")
    ax[0].set_xlabel('predicted')
    ax[0].set_ylabel('actual')
    ax[0].title.set_text('confusion matrix for train data')
    sns.heatmap(ax = ax[1],data = cv_mat,annot=True,fmt='g')
    ax[1].set_xlabel('predicted')
    ax[1].set_ylabel('actual')
    ax[1].title.set_text('confusion matrix for CV data')
    sns.heatmap(ax = ax[2],data = test_mat,annot=True,fmt='g')
    ax[2].set_xlabel('predicted')
    ax[2].set_ylabel('actual')
    ax[2].title.set_text('confusion matrix for test data')


def best_random_forest_model(X_train_smote, Y_train_smote,x_train,y_train,x_cv,y_cv,x_test,y_test):
    best_rf_model = RandomForestClassifier(n_estimators=200,criterion='gini', max_depth=50,max_features = 'auto')
    best_rf_model.fit(X_train_smote, Y_train_smote)
    pred=best_rf_model.predict(x_test)
    print(classification_report(y_test, pred))
    pred_cv=best_rf_model.predict(x_cv)
    print(classification_report(y_cv, pred_cv))

def xgboost_model_tuning_depth(X_train_smote, Y_train_smote, x_train, y_train, x_cv, y_cv, x_test, y_test):
    depth = [3, 5, 10, 20, 50, 100]
    cv_f1_score = []
    train_f1_score = []
    for i in depth:
        model = XGBClassifier(max_depth=i, n_jobs=-1, n_estimators=10)
        model.fit(X_train_smote, Y_train_smote)
        CV = CalibratedClassifierCV(model, method='sigmoid')
        CV.fit(X_train_smote, Y_train_smote)
        predicted = CV.predict(x_cv)
        train_predicted = CV.predict(x_train)
        cv_f1_score.append(f1_score(y_cv, predicted, average='macro'))
        train_f1_score.append(f1_score(y_train, train_predicted, average='macro'))
        print('depth {0} is finished'.format(i))
    for i in range(0, len(cv_f1_score)):
        print('f1 value score for depth =' + str(depth[i]) + ' is ' + str(cv_f1_score[i]))
    plt.plot(depth, cv_f1_score, c='r')
    plt.plot(depth, train_f1_score, c='b')
    plt.xlabel('depth(depth of the tree)')
    plt.ylabel('f1 score(train and test)')
    plt.title('depth vs df1 score')
    for i, score in enumerate(cv_f1_score):
        plt.annotate((depth[i], np.round(score, 4)), (depth[i], np.round(cv_f1_score[i], 4)))
    for i, score1 in enumerate(train_f1_score):
        plt.annotate((depth[i], np.round(score1, 4)), (depth[i], np.round(train_f1_score[i], 4)))
    index = cv_f1_score.index(max(cv_f1_score))
    best_dpt = depth[index]
    print('best max depth is ' + str(best_dpt))
    model = XGBClassifier(max_depth=best_dpt, n_jobs=-1)
    model.fit(X_train_smote, Y_train_smote)
    predict_train = model.predict(x_train)
    print('f1 score on train data ' + str(f1_score(y_train, predict_train, average='macro')))
    train_mat = confusion_matrix(y_train, predict_train)
    predict_cv = model.predict(x_cv)
    print('f1 score on cv data ' + str(f1_score(y_cv, predict_cv, average='macro')))
    cv_mat = confusion_matrix(y_cv, predict_cv)
    predict_test = model.predict(x_test)
    print('f1 score on test data ' + str(f1_score(y_test, predict_test, average='macro')))
    test_mat = confusion_matrix(y_test, predict_test)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.heatmap(ax=ax[0], data=train_mat, annot=True, fmt='g', cmap="YlGnBu")
    ax[0].set_xlabel('predicted')
    ax[0].set_ylabel('actual')
    ax[0].title.set_text('confusion matrix for train data')
    sns.heatmap(ax=ax[1], data=cv_mat, annot=True, fmt='g')
    ax[1].set_xlabel('predicted')
    ax[1].set_ylabel('actual')
    ax[1].title.set_text('confusion matrix for CV data')
    sns.heatmap(ax=ax[2], data=test_mat, annot=True, fmt='g')
    ax[2].set_xlabel('predicted')
    ax[2].set_ylabel('actual')
    ax[2].title.set_text('confusion matrix for test data')


def xgboost_model_tuning_estimator(X_train_smote, Y_train_smote, x_train, y_train, x_cv, y_cv, x_test, y_test):
    estimators = [10, 20, 50, 100, 200]
    cv_f1_score = []
    train_f1_score = []
    for i in estimators:
        model = XGBClassifier(n_estimators=i, max_depth=50, n_jobs=-1)
        model.fit(X_train_smote, Y_train_smote)
        CV = CalibratedClassifierCV(model, method='sigmoid')
        CV.fit(X_train_smote, Y_train_smote)
        predicted = CV.predict(x_cv)
        train_predicted = CV.predict(x_train)
        cv_f1_score.append(f1_score(y_cv, predicted, average='macro'))
        train_f1_score.append(f1_score(y_train, train_predicted, average='macro'))
        print('Estimator {0} is finished'.format(i))
    for i in range(0, len(cv_f1_score)):
        print('f1 value score n_estimators =' + str(estimators[i]) + ' is ' + str(cv_f1_score[i]))
    plt.plot(estimators, cv_f1_score, c='r')
    plt.plot(estimators, train_f1_score, c='b')
    plt.xlabel('estimators(n_estimators)')
    plt.ylabel('f1 score(train and test)')
    plt.title('estimators vs df1 score')
    for i, score in enumerate(cv_f1_score):
        plt.annotate((estimators[i], np.round(score, 4)), (estimators[i], np.round(cv_f1_score[i], 4)))
    for i, score1 in enumerate(train_f1_score):
        plt.annotate((estimators[i], np.round(score1, 4)), (estimators[i], np.round(train_f1_score[i], 4)))
    index = cv_f1_score.index(max(cv_f1_score))
    best_est = estimators[index]
    print('best estimator is ' + str(best_est))
    model = XGBClassifier(n_estimators=best_est, max_depth=50, n_jobs=-1)
    model.fit(X_train_smote, Y_train_smote)
    predict_train = model.predict(X_train_smote)
    print('f1 score on train data ' + str(f1_score(Y_train_smote, predict_train, average='macro')))
    train_mat = confusion_matrix(Y_train_smote, predict_train)
    predict_cv = model.predict(x_cv)
    print('f1 score on cv data ' + str(f1_score(y_cv, predict_cv, average='macro')))
    cv_mat = confusion_matrix(y_cv, predict_cv)
    predict_test = model.predict(x_test)
    print('f1 score on test data ' + str(f1_score(y_test, predict_test, average='macro')))
    test_mat = confusion_matrix(y_test, predict_test)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.heatmap(ax=ax[0], data=train_mat, annot=True, fmt='g', cmap="YlGnBu")
    ax[0].set_xlabel('predicted')
    ax[0].set_ylabel('actual')
    ax[0].title.set_text('confusion matrix for train data')
    sns.heatmap(ax=ax[1], data=cv_mat, annot=True, fmt='g')
    ax[1].set_xlabel('predicted')
    ax[1].set_ylabel('actual')
    ax[1].title.set_text('confusion matrix for CV data')
    sns.heatmap(ax=ax[2], data=test_mat, annot=True, fmt='g')
    ax[2].set_xlabel('predicted')
    ax[2].set_ylabel('actual')
    ax[2].title.set_text('confusion matrix for test data')

def best_xgboost_model(X_train_smote, Y_train_smote,x_train,y_train,x_cv,y_cv,x_test,y_test):
    best_xgb_model = XGBClassifier(n_estimators=200,criterion='gini', max_depth=50,max_features = 'auto')
    best_xgb_model.fit(X_train_smote, Y_train_smote)
    pred=best_xgb_model.predict(x_test)
    print(classification_report(y_test, pred))
    pred_cv=best_xgb_model.predict(x_cv)
    print(classification_report(y_cv, pred_cv))
