from abc import ABC, abstractmethod
from numpy.linalg import inv
from scipy.optimize import fsolve
from sklearn.base import is_classifier,is_regressor
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import fsolve
from drf import drf
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
import warnings
import copy

def element_squaredot(arr):
    return np.apply_along_axis(lambda x:np.outer(x,x),1,arr)

class MissingData(ABC):
    def __init__(self,data,y_col,x_col,model,K):
        self.data=copy.deepcopy(data)
        self.y_col=y_col
        self.x_cols=x_col
        self.z_cols=[]
        
        self.model=model
        self.second_model=LinearRegression()
        
        self.K=K
        self.type=''
        
        self.prepare_data()
        self.model_check()

    #find missing value, insert R col, regenerate dataset
    def prepare_data(self):
        missing_status=self.data.isna().any()
        missing_cols=missing_status[missing_status].index.tolist()

        if 'R' in missing_status.index.tolist():
            raise ValueError('We use R as missing indicator, please change data column name.') 

        if len(missing_cols)==0:
            raise ValueError('No value is missing, recommend to use standard linear model.') 
        elif len(missing_cols)==1:
            if missing_cols[0]==self.y_col:
                self.type='missing_y'
            else:
                self.type='missing_x'
                                
        elif len(missing_cols)>1:
            if self.y_col in missing_cols:
                raise ValueError('X and Y missing simultaneously is not supported.') 
            else:
                self.type='missing_x'

        self.data['R']=~self.data.isna().any(axis=1)
        self.data['R']=self.data['R'].astype('float')

        if self.type=='missing_x':
            self.z_cols=[z for z in self.x_cols if z not in missing_cols]
            self.x_cols=missing_cols

    def fit(self):
        if self.type=='missing_y':
            return self.missing_y()
        else:
            return self.missing_x()
    
    def __str__(self):
        missing_variables=self.y_col if self.type=='missing_y' else self.x_cols
        no_missing_x=self.x_cols if self.type=='missing_y' else self.z_cols
        missing_num=sum(1-self.data['R'])
        data_info = f'Dependent variable: {self.y_col}\n' \
                    f'Independent variables:{self.x_cols+self.z_cols}\n'\
                    f'Missing variable(s): {missing_variables}\n' \
                    f'Non-missing independent variable(s): {no_missing_x}\n' \
                    f'No. Observations: {self.data.shape[0]}\n'\
                    f'Missing num: {missing_num}\n'\
                    '-----------------------\n'\
                    f'Using model:{self.model.__str__()}'
        return data_info

    def model_check(self):    
        if is_classifier(self.model):
            self.method='predict_proba'
        else:
            self.method='predict'

    @abstractmethod
    def missing_y(self):
        pass
    
    @abstractmethod
    def missing_x(self):
        pass

class MissingDataLinear(MissingData):
    def __init__(self,data,y_col,x_col,model,K):
        super().__init__(data,y_col,x_col,model,K)
        if is_classifier(self.model):
            warnings.warn('classification model can only be used when missing data is binary variable')

    def __str__(self):
        return super().__str__()

    def missing_y(self):
        data=self.data
        form_x=self.x_cols
        form_y=self.y_col

        skf = StratifiedKFold(n_splits=self.K,shuffle=True)
        cvgroup=[]
        for train, test in skf.split(data, data['R']):
            cvgroup.append((train,test))
        
        lambda_dic={}
        mu_dic={}
        psi_a=pd.DataFrame(0,index=form_x,columns=form_x)
        psi_b=pd.Series(0,index=form_x)
        for i,(train,test) in enumerate(cvgroup):
            X=data.take(test)[form_x]
            Y=data.take(test)[form_y].fillna(0).values
            R=data.take(test)['R'].values

            #lambda=P(R=1|X)
            lambda_=data.take(train)['R'].mean()
            lambda_dic[str(i)]=lambda_#scalar

            #mu=E(Y|X,R=1)
            train_set=data.take(train).dropna()
            mu_model=self.model.fit(train_set[form_x],train_set[form_y].values)
            mu=mu_model.predict(X)
            mu_dic[str(i)]=mu#vector

            #psi_a=XX^T (the notion for X is reverse, i.e. X in python is X^T in paper)
            #psi_b=X[R(Y-mu)/lambda+mu]
            psi_a+=(X.T).dot(X)/len(test)
            psi_b+=X.T.dot(R*(Y-mu)/lambda_+mu)/len(test)

        coef=inv(psi_a).dot(psi_b)#main para to be estimated
        result=pd.DataFrame(coef,index=form_x,columns=['coef'])

        J0=psi_a/self.K
        PSI2=pd.DataFrame(0,index=form_x,columns=form_x)
        for i,(train,test) in enumerate(cvgroup):
            X=data.take(test)[form_x]
            Y=data.take(test)[form_y].fillna(0).values
            R=data.take(test)['R'].values

            lambda_=lambda_dic[str(i)]
            mu=mu_dic[str(i)]
            
            #notice the 'mul' here, not dot
            #right part is vector of shape (n_sample,)
            #left part is matrix of shape(form_x, n_sample)
            psi=X.T.mul(R*(Y-mu)/lambda_+mu-X.dot(coef))
            PSI2=PSI2+psi.dot(psi.T)/len(test)

        PSI2=PSI2/self.K
        sigma2=inv(J0).dot(PSI2).dot(inv(J0))/data.shape[0]
    
        #return (coef,np.diagonal(np.sqrt(sigma2)),sigma2)
        #sigma2 is a covariance matrix which contains negative value
        result['ste']=np.sqrt(np.diagonal(sigma2))
        return result
    
    def missing_x(self):
        data=self.data
        form_x=self.x_cols
        form_z=self.z_cols
        form_y=self.y_col

        form_yz=form_z+[form_y]

        K=5
        skf = StratifiedKFold(n_splits=K,shuffle=True)
        cvgroup=[]
        for train, test in skf.split(data, data['R']):
            cvgroup.append((train,test))
        
        lambda_dic={}
        mu1_dic={}
        mu2_dic={}
        psi_a=pd.DataFrame(0,index=form_x+form_z,columns=form_x+form_z)
        psi_b=pd.Series(0,index=form_x+form_z)
    
        for i,(train,test) in enumerate(cvgroup):
            X=data.take(test)[form_x].fillna(0).values.reshape(-1,len(form_x))
            Y=data.take(test)[form_y].values.reshape(-1,1)
            Z=data.take(test)[form_z].values.reshape(-1,len(form_z))
            R=data.take(test)['R'].values.reshape(-1,1)
            #lambda=P(R=1|Z,Y)
            lambda_=data.take(train)['R'].mean()
            lambda_dic[str(i)]=lambda_#scalar

            #mu1=E(X|Z,Y,R=1)
            #mu2=E(XX^T|Z,Y,R=1)
            train_set=data.take(train).dropna()
            mu1_model=self.model.fit(train_set[form_yz],train_set[form_x].values)
            mu1=mu1_model.predict(data.take(test)[form_yz])
            mu1_dic[str(i)]=mu1#vector

            XXT=element_squaredot(train_set[form_x].values)
            XXT=XXT.reshape(-1,len(form_x)**2)
            mu2_model = self.second_model.fit(train_set[form_yz], XXT)
            mu2=mu2_model.predict(data.take(test)[form_yz])
            mu2=mu2.reshape(-1,len(form_x),len(form_x))#vector
            mu2_dic[str(i)]=mu2

            left_up=pd.DataFrame(sum(np.expand_dims(R,axis=2)*(element_squaredot(X)-mu2)/lambda_+mu2),index=form_x,columns=form_x)
            left_down=pd.DataFrame(Z.T.dot(R*(X-mu1)/lambda_+mu1),index=form_z,columns=form_x)
            right_down=pd.DataFrame(Z.T.dot(Z),index=form_z,columns=form_z)
            psi_a=psi_a+pd.concat([pd.concat([left_up,left_down.T],axis=1),\
                                pd.concat([left_down,right_down],axis=1)],axis=0)/len(test)
            
            psi_b=psi_b+pd.concat([pd.Series(sum(Y*(R*(X-mu1)/lambda_+mu1)),index=form_x),pd.Series((Z.T.dot(Y)).flatten(),index=form_z)],axis=0)/len(test)
            
        coef=inv(psi_a).dot(psi_b)
        result=pd.DataFrame(coef,index=form_x+form_z,columns=['coef'])
        
        beta=result.loc[form_x]['coef'].values
        gamma=result.loc[form_z]['coef'].values

        J0=psi_a/K
        PSI2=pd.DataFrame(0,index=form_x+form_z,columns=form_x+form_z)

        for i,(train,test) in enumerate(cvgroup):
            X=data.take(test)[form_x].fillna(0).values
            Y=data.take(test)[form_y].values
            Z=data.take(test)[form_z].values.reshape(-1,len(form_z))
            R=data.take(test)['R'].values.reshape(-1,1)

            lambda_=lambda_dic[str(i)]
            mu1=mu1_dic[str(i)]
            mu2=mu2_dic[str(i)]

            up=(R*(X-mu1)/lambda_+mu1)*(Y-Z.dot(gamma)).reshape(-1,1)-(np.expand_dims(R,axis=2)*(element_squaredot(X)-mu2)/lambda_+mu2).dot(beta)
            down=Z.T*((Y-(R*(X-mu1)/lambda_+mu1).dot(beta))-Z.dot(gamma))
            up=pd.DataFrame(up,columns=form_x).T
            down=pd.DataFrame(down.T,columns=form_z).T
            psi=pd.concat([up,down],axis=0)
            PSI2=PSI2+psi.dot(psi.T)/len(test)

        PSI2=PSI2/K
        sigma2=inv(J0).dot(PSI2).dot(inv(J0))/data.shape[0]
    
        #return (coef,np.diagonal(np.sqrt(sigma2)),sigma2)
        result['ste']=np.sqrt(np.diagonal(sigma2))
        return result

class MissingDataLogistics(MissingData):
    def __init__(self,data,y_col,x_col,model,K):
        super().__init__(data,y_col,x_col,model,K)

    def __str__(self):
        return super().__str__()
        
    def missing_y(self):
        data=self.data
        form_x=self.x_cols
        form_y=self.y_col

        #support to use classification method
        skf = StratifiedKFold(n_splits=self.K,shuffle=True)
        cvgroup=[]
        for train, test in skf.split(data, data['R']):
            cvgroup.append((train,test))
        
        def score_function(coef):
            result=np.zeros(len(form_x))
            for i,(train,test) in enumerate(cvgroup):
                X=data.take(test)[form_x]
                Y=data.take(test)[form_y].fillna(0).values
                R=data.take(test)['R'].values

                #lambda=P(R=1|X)
                lambda_=data.take(train)['R'].mean()

                #mu=E(Y|X,R=1)
                train_set=data.take(train).dropna()
                mu_model=self.model.fit(train_set[form_x],train_set[form_y].values)
                if self.method=='predict_proba':
                    mu=mu_model.predict_proba(X)[:,1]
                else:
                    mu=mu_model.predict(X)
                result=result+X.T.dot(R*(Y-mu)/lambda_+mu-1/(1+np.exp(-X.dot(coef))))
            return result

        result=fsolve(score_function,np.ones(len(form_x)))
        return result
    
    def missing_x(self):
        data=self.data
        form_x=self.x_cols
        form_z=self.z_cols
        form_y=self.y_col
        model_ipw=self.model
        model_mu1=self.second_model

        form_yz=form_z+[form_y]

        K=5
        skf = StratifiedKFold(n_splits=K,shuffle=True)
        cvgroup=[]
        for train, test in skf.split(data, data['R']):
            cvgroup.append((train,test))
        
        lambda_dic={}
        eta_dic={}

        for i,(train,test) in enumerate(cvgroup):
            l=len(train)
            ipw_train=train[:l//2]#dat1
            nuis1_train=train[l//2:]#dat2
            #nuis2_train=train
            
            lambda_=data.take(train)['R'].mean()
            
            train_set=data.take(train).dropna()
            ipw_train_set=data.take(ipw_train).dropna()
            nuis1_train_set=data.take(nuis1_train).dropna()
            
            #model_ipw must be GLM model, have coef_
            model_ipw.fit(ipw_train_set[form_x+form_z],ipw_train_set[form_y])
            theta_ipw=model_ipw.coef_[0]
            theta_ipw_intercept=model_ipw.intercept_

            #regression model
            model_mu1.fit(train_set[form_yz],train_set[form_x])
            mu1=model_mu1.predict(data.take(test)[form_yz])
            mu1=pd.DataFrame(mu1,index=test)
            
            mu_dic={}
            for x_name in form_x:
                DRF = drf(min_node_size = 15, num_trees = 2000, splitting_rule = "FourierMMD") #those are the default values
                DRF.fit(nuis1_train_set[form_yz],nuis1_train_set[x_name])
                mu=DRF.predict(newdata = data.take(test)[form_yz]).weights
                mu_dic[x_name]=mu

            #index:test set; columns:nuis1 train set 
            len_x=len(form_x)
            part1=data.take(test)[form_z].dot(theta_ipw[len_x:])+theta_ipw_intercept#constant item
            #part1=pd.concat([part1]*mu.shape[1],axis=1)
            #part1.columns=nuis1_train_set.index
                
            expectation_part2=np.zeros(len(test))
            expectation_X=[]
            for x_i,x_name in enumerate(form_x):
                mu=mu_dic[x_name]
                
                part2=theta_ipw[x_i]*nuis1_train_set[x_name]
                part2=pd.concat([part2]*mu.shape[0],axis=1).T
                part2.index=test
                expectation_part2+=(part2*mu).sum(axis=1)

                X_matrix=pd.concat([nuis1_train_set[x_name]]*mu.shape[0],axis=1).T
                X_matrix.index=test
                expectation_X.append((X_matrix*mu).sum(axis=1))
                
            expectation_X=pd.concat(expectation_X,axis=1)
            #expectation_part1=part1
            
            expX=1/(1+np.exp(-part1-expectation_part2))
            mu2=expX
            mu3=expectation_X.apply(lambda x:x*expX)
            mu3=pd.DataFrame(mu3,columns=np.arange(len_x))
            #expX=1/(1+np.exp(-part1-part2))
            #X_matrix=pd.concat([nuis1_train_set[form_x]]*mu.shape[0],axis=1).T
            #X_matrix.index=test
            #XexpX=X_matrix*expX
            #mu2=(expX*mu).sum(axis=1)
            #mu3=(XexpX*mu).sum(axis=1)
            Y=data.take(test)[form_y].values
            Z=data.take(test)[form_z]
            
            eta=pd.concat([mu1.apply(lambda x:x*Y)-mu3,Z.apply(lambda x:x*(Y-mu2))],axis=1).T
            eta.index=form_x+form_z
            lambda_dic[str(i)]=lambda_
            eta_dic[str(i)]=eta

        def score_function(coef):
            result=np.zeros(len(form_x+form_z))
            for i,(train,test) in enumerate(cvgroup):
                XZ=data.take(test)[form_x+form_z].fillna(0)
                Y=data.take(test)[form_y].values
                R=data.take(test)['R'].values
                result=result+XZ.T.dot(R/lambda_dic[str(i)]*(Y-1/(1+np.exp(-XZ.dot(coef)))))\
                        -eta_dic[str(i)].dot(R/lambda_dic[str(i)]-1)
            return result

        result=fsolve(score_function,np.zeros(len(form_x+form_z)))
        return result