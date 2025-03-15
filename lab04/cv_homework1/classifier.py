from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import joblib
import glob
import os
import time
import numpy as np


train_path = 'data/features/train/'
test_path = 'data/features/test/'

if __name__ == "__main__":
    t0 = time.time()
    clf_type = 'LIN_SVM'
    fds = []
    labels = []
    num = 0
    total = 0
    for feat_path in glob.glob(os.path.join(train_path, '*.feat')):
        data = joblib.load(feat_path)
        # 分别拿数据和标签
        fds.append(data[:-1])
        labels.append(data[-1])
    # for i, feature in enumerate(fds):
    #     if len(feature) != 324:
    #        print(f"Sample {i} has {len(feature)} features")

    if clf_type == 'LIN_SVM':
        # 定义分类器

        clf = LinearSVC(max_iter=500, C = 0.01, tol=1e-5)
        print("Training a Linear SVM Classifier......")



        # param_grid = {'C': [0.1, 0.11, 0.12, 0.13, 0.14]}
        # grid_search = GridSearchCV(clf, param_grid, cv=5)
        # grid_search.fit(fds, labels)
        # print(f"Best C value: {grid_search.best_params_['C']}")


    # 训练
    clf.fit(fds, labels)
    print('Training done!')
    print('Testing......')
    print(f"Actual number of iterations: {clf.n_iter_}")
    # 如果达到最大迭代次数，会与 max_iter 相同
    if clf.n_iter_ == clf.max_iter:
        print("Warning: Maximum number of iterations reached. Model may not have converged.")

    # joblib.dump(clf, 'trained_model.pkl')
    # print("Model saved as 'trained_model.pkl'.")

    # clf = joblib.load('trained_model.pkl')
    # 测试，计算精度
    for feat_path in glob.glob(os.path.join(test_path, '*.feat')):
        total += 1
        data_test = joblib.load(feat_path)
        temp = data_test[:-1]
        data_test_feat = temp.reshape((1, -1))
        result = clf.predict(data_test_feat)
        if int(result.item()) == int(data_test[-1].item()):
            num += 1
    rate = float(num) / total
    t1 = time.time()
    print('classification accuracy is :%f' % rate)
    print('total time is :%f' % (t1 - t0))
