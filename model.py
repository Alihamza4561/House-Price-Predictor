import numpy as np
class gdRegression:
    def __init__(self,learn_rate=0.01,epoch=100):
        self.intercept_=None
        self.coef_=None
        self.lr=learn_rate
        self.epoch=epoch

    def fit(self,X_train,y_train):
        X_train=np.array(X_train)
        y_train=np.array(y_train)
        self.intercept_=0
        n=X_train.shape[1]
        self.coef_=np.ones(n)

        for i in range(self.epoch):
            y_hat=np.dot(X_train,self.coef_)+self.intercept_
            inter_der=-2*np.mean(y_train-y_hat)
            self.intercept_=self.intercept_-(self.lr*inter_der)
            coef_der=-2*np.dot(X_train.T,(y_train-y_hat))/X_train.shape[0]
            self.coef_=self.coef_ -(self.lr*coef_der)

    def predict(self,X_test):
        return np.dot(X_test,self.coef_)+self.intercept_   