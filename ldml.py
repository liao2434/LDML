from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.base import is_classifier
from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd
from double_ml_data import DoubleMLData
import copy

def cross_fit_propensities(data, cvgroup, form_x, form_t, method_prop, trim=(0.01,0.99), trim_type='none', normalize=True, method='predict'):
    '''cross_val train and predict prop score, then process them'''
    prop=cross_val_predict(method_prop, data[form_x], data[form_t], cv=cvgroup,method=method)
    #adjust classifier model output
    if method == 'predict_proba':
        prop = prop[:, 1]

    if trim_type == 'drop':
        keep = np.logical_and(prop>trim[0],prop<trim[1])
        prop[~keep] = 0.5
    else:
        keep = np.full(data.shape[0],True) #otherwise,keep all rows  
        
    if trim_type == 'clip':
        prop=np.clip(prop,trim[0],trim[1])

    #we normalize propensity weights to have mean 1 within each treatment group
    #(In each group) P_i=P_i*MEAN(1/P_i),therefore 1/P_i would have mean 1
    if normalize:
        prop[keep] = data[form_t][keep]*prop[keep]*np.mean(data[form_t][keep]/prop[keep]) + \
                    (1.-data[form_t][keep])*(1.-(1.-prop[keep])*np.mean((1.-data[form_t][keep])/(1.-prop[keep])))
    
    return prop,keep

def solve_cumsum(vector,c):
    '''Return i for i such that summing vector[1:i] is closest to c'''
    return pd.Series(abs(np.cumsum(vector)-c)).idxmin()#keep index info

def density(X, w, x):
    '''Estimate the density of data in X at point(s) x with weights w'''
    X=np.array(X)
    if X.ndim==1:
        X=X.reshape(-1,1)
    x=np.array([x]).reshape(-1,1)
    kde = KernelDensity(kernel='gaussian', bandwidth="scott").fit(X=X,sample_weight=w)#fit whole curve
    return np.exp(kde.score_samples(x))#predict points

class LocalizedDML:
    def __init__(self,
                 gammas,
                 obj_dml_data,
                 n_folds,
                 method_ipw,
                 method_prop=None,
                 method_cdf=None,
                 trim_upperbound=0.99,
                 trim_lowerbound=0.01,
                 trim_type='none',
                 normalize=True,
                 avg_eqn=True,
                 semiadaptive=False):

        self.gammas=None
        if isinstance(gammas, list):
            if not False in [isinstance(g,float) for g in gammas]:
                if max(gammas)<1.0 and min(gammas)>0.0:
                    self.gammas=gammas

        if not self.gammas:
            raise TypeError('Gammas must be list of floats between 0.0 and 1.0. '
                        f'{str(gammas)} was passed.') 

        # check and pick up obj_dml_data
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError('The data must be of DoubleMLData type. '
                            f'{str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed.') 
        if len(obj_dml_data.d_cols)>1:
            raise TypeError('QTE estimation only accept signle treatment variable. '
                            f'obj_dml_data.d_cols {str(obj_dml_data.d_cols)} was passed.')
        self.dataset = obj_dml_data
        
        if not method_prop:
            method_prop=copy.deepcopy(method_ipw)
        if not method_cdf:
            method_cdf=copy.deepcopy(method_ipw)

        self.methods={'method_ipw':method_ipw,'method_prop':method_prop,'method_cdf':method_cdf}
        self.predict_methods={}
        
        for name,learner in self.methods.items():
            if not hasattr(learner, 'fit'):
                raise TypeError(f'{name}:{str(learner)} has no method .fit().')
            if is_classifier(learner):
                if not hasattr(learner, 'predict_proba'):
                    raise TypeError(f'{name}:{str(learner)} has no method .predict_proba().')
                self.predict_methods[name]='predict_proba'
            else:
                if not hasattr(learner, 'predict'):
                    raise TypeError(f'{name}:{str(learner)} has no method .predict().')
                self.predict_methods[name]='predict'
        
        #if treatment is binary, method ipw and method prop can be classification model
        if not self.dataset.binary_treats[0]:
            if is_classifier(method_ipw) or is_classifier(method_prop):
                raise ValueError('If treatment variable is not binary, then method_ipw and method_prop must be regressors.')

        # check resampling specifications
        if not isinstance(n_folds, int):
            raise TypeError('The number of folds must be of int type. '
                            f'{str(n_folds)} of type {str(type(n_folds))} was passed.')
        if n_folds < 1:
            raise ValueError('The number of folds must be positive. '
                             f'{str(n_folds)} was passed.')
        self.K = n_folds

        if not isinstance(trim_upperbound, float):
            raise TypeError('trim_upperbound must be of float type. '
                            f'{str(trim_upperbound)} of type {str(type(trim_upperbound))} was passed.')
        if not isinstance(trim_lowerbound, float):
            raise TypeError('trim_lowerbound must be of float type. '
                            f'{str(trim_lowerbound)} of type {str(type(trim_lowerbound))} was passed.')
        if trim_upperbound<=trim_lowerbound:
            raise ValueError('trim_upperbound must be greater than trim_lowerbound. Bounds should be floats between 0.0 and 1.0'
                             f'Got upper:{str(trim_upperbound)} and lower: {str(trim_lowerbound)}.')
        self.trim=(trim_lowerbound,trim_upperbound)

        if (not isinstance(trim_type, str)) | (trim_type not in ['none','drop','clip']):
            raise ValueError('trim_type must be "none","trim" or "drop". '
                             f'Got {str(trim_type)}.')
        self.trim_type = trim_type
        
        if not isinstance(normalize, bool):
            raise TypeError('normalize must be True or False. '
                            f'Got {str(normalize)}.')
        self.normalize = normalize

        if not isinstance(avg_eqn, bool):
            raise TypeError('avg_eqn must be True or False. '
                            f'Got {str(avg_eqn)}.')
        self.avg_eqn = avg_eqn

        if not isinstance(semiadaptive, bool):
            raise TypeError('avg_eqn must be True or False. '
                            f'Got {str(semiadaptive)}.')
        self.semiadaptive = semiadaptive

    def __str__(self):
        class_name = self.__class__.__name__
        header = f'================== {class_name} Object ==================\n'

        data_info = f'Outcome variable: {self.dataset.y_col}\n' \
                    f'Treatment variable(s): {self.dataset.d_cols}\n' \
                    f'Covariates: {self.dataset.x_cols}\n' \
                    f'Instrument variable(s): {self.dataset.z_cols}\n' \
                    f'No. Observations: {self.dataset.n_obs}\n'
        score_info = f'Use avg_eqn: {self.avg_eqn}\n'
        learner_info = ''
        for key, value in self.methods.items():
            learner_info += f'Learner {key}: {str(value)}\n'

        resampling_info = f'No. folds: {self.K}\n'

        res = header + \
            '\n------------------ Data summary      ------------------\n' + data_info + \
            '\n------------------ ldml type ------------------\n' + score_info + \
            '\n------------------ Machine learner   ------------------\n' + learner_info + \
            '\n------------------ Resampling        ------------------\n' + resampling_info
        return res
    
    def ipw_fit(self,dataset=None,K_ipw=None):
        if not dataset:
            dataset=self.dataset
        if not K_ipw:
            K_ipw=self.K
        
        data=dataset.data
        form_y=dataset.y_col
        form_x=dataset.x_cols
        form_t=dataset.d_cols[0]

        #prepare for solve_cumsum 
        data.sort_values(by=form_y,ascending=True,inplace=True)
        data.reset_index(drop=True,inplace=True)

        #分层cv
        skf = StratifiedKFold(n_splits=K_ipw,shuffle=True)
        cvgroup=[]
        for train, test in skf.split(data, data[form_t]):
            cvgroup.append((train,test))#train,test are list, see demo1
        
        prop,keep = cross_fit_propensities(data, cvgroup, form_x, form_t, self.methods['method_ipw'],\
                                        trim=self.trim, trim_type=self.trim_type, normalize=self.normalize,method=self.predict_methods['method_ipw'])

        w1=keep*data[form_t]/prop
        w0=keep*(1-data[form_t])/(1-prop)

        result=[]
        for i,gamma in enumerate(self.gammas):
            if self.avg_eqn:    
                q1=data[form_y][solve_cumsum(w1/sum(keep),gamma)]
                q0=data[form_y][solve_cumsum(w0/sum(keep),gamma)]
            else:
                q1_list=[]
                q0_list=[]
                for train,test in cvgroup:
                    # solve_cumsum keeps origin index, so data[form_y] doesn't need to slice
                    q1_list.append(data[form_y][solve_cumsum(w1[test]/sum(keep[test]),gamma)])
                    q0_list.append(data[form_y][solve_cumsum(w0[test]/sum(keep[test]),gamma)])
                q1=np.mean(q1_list)
                q0=np.mean(q0_list)
            
            # sample value of score function/J, J is estimated by IPW kde at q
            keep_t1_mask=np.logical_and(data[form_t]==1,keep)
            keep_t0_mask=np.logical_and(data[form_t]==0,keep)
            # density result is different from R, is bigger
            psi1=(w1[keep]*(data[form_y][keep]<=q1)-gamma)/density(data[form_y][keep_t1_mask], 1/prop[keep_t1_mask], q1)
            psi0=(w0[keep]*(data[form_y][keep]<=q0)-gamma)/density(data[form_y][keep_t0_mask], 1/(1-prop[keep_t0_mask]), q0)#???

            se1 = np.std(psi1,ddof=1) / np.sqrt(sum(keep))
            se0 = np.std(psi0,ddof=1) / np.sqrt(sum(keep))
            seqte = np.std(psi1-psi0,ddof=1) / np.sqrt(sum(keep))

            result.append(pd.DataFrame({'gamma':gamma,'q1':q1,'q0':q0,'qte':q1-q0,\
                                        'se1':se1,'se0':se0,'seqte':seqte},index=[i]))
        
        self.ipw_estimation=pd.concat(result,axis=0)
        return self.ipw_estimation

    def fit(self):
        data=self.dataset.data
        form_y=self.dataset.y_col
        form_x=self.dataset.x_cols
        form_t=self.dataset.d_cols[0]

        #prepare for solve_cumsum 
        data.sort_values(by=form_y,ascending=True,inplace=True)
        data.reset_index(drop=True,inplace=True)
        
        skf = StratifiedKFold(n_splits=self.K)
        cvgroup=[]
        split_cvgroup=[]
        for train, test in skf.split(data, data[form_t]):
            cvgroup.append((train,test))#train,test are list, see demo1
            if not self.semiadaptive:
                l=len(train)
                ipw_train=train[:l//2]
                nuis1_train=train[l//2:]
                split_cvgroup.append((ipw_train,nuis1_train,test))
            else:
                split_cvgroup.append((copy.deepcopy(train),copy.deepcopy(train),test))#need copy???

        #nuis2
        prop,keep = cross_fit_propensities(data, cvgroup, form_x, form_t, self.methods['method_prop'],\
                                        trim=self.trim, trim_type=self.trim_type, normalize=self.normalize,method=self.predict_methods['method_prop'])
        w1=keep*data[form_t]/prop
        w0=keep*(1-data[form_t])/(1-prop)

        K_ipw=int(np.ceil((self.K-1)/2))#向上取整
        
        result=[]
        for i,gamma in enumerate(self.gammas):
            cdf0,cdf1=np.zeros(data.shape[0]),np.zeros(data.shape[0])

            for ipw_train,nuis1_train,test in split_cvgroup:
                #use take method instead of iloc, for automatiaclly copy
                ipw_dataset=DoubleMLData(data.take(ipw_train),y_col=form_y,d_cols=form_t,x_cols=form_x)
                ipw_result = self.ipw_fit(ipw_dataset,K_ipw)

                def fit_predict(cdf,q):   
                    cdf_goal=(data[form_y]<=ipw_result[q][0])
                    cdf_fit=self.methods['method_cdf'].fit(data.take(nuis1_train),cdf_goal.take(nuis1_train))
                    cdf[test]=cdf_fit.predict(data.take(test))

                fit_predict(cdf1,'q1')
                fit_predict(cdf0,'q0')

            def ldml_formula(w,cdf,q_num,mask):
                if q_num==1:
                    p=(1.- data[form_t][mask]/prop[mask])
                elif q_num==0:
                    p=(1.-data[form_t][mask])/(1.-prop[mask])

                return data[form_y][solve_cumsum(w[mask]/sum(mask),gamma - np.mean(cdf[mask]*p))]
                
            if self.avg_eqn:
                q1=ldml_formula(w1,cdf1,1,keep)
                q0=ldml_formula(w0,cdf0,1,keep)
            else:
                q1_list,q0_list=[],[]
                for ipw_train,nuis1_train,test in split_cvgroup:
                    q1=ldml_formula(1,keep&test)
                    q0=ldml_formula(0,keep&test)
                q1,q0=np.mean(q1_list),np.mean(q0_list)
            
            def score_func(w,q,cdf,q_num,mask):
                if q_num==1:
                    p=(1.- data[form_t][mask]/prop[mask])
                elif q_num==0:
                    p=(1.-data[form_t][mask])/(1.-prop[mask])        
                return (w[keep] * (data[form_y][keep] <= q)-gamma-cdf[mask]*p)

            # sample value of score function/J, J is estimated by IPW kde at q
            keep_t1_mask=np.logical_and(data[form_t]==1,keep)
            keep_t0_mask=np.logical_and(data[form_t]==0,keep)
            psi1=score_func(w1,q1,cdf1,1,keep)/density(data[form_y][keep_t1_mask], 1/prop[keep_t1_mask], q1)
            psi0=score_func(w0,q0,cdf0,0,keep)/density(data[form_y][keep_t0_mask], 1/(1-prop[keep_t0_mask]), q0)

            se1 = np.std(psi1,ddof=1) / np.sqrt(sum(keep))
            se0 = np.std(psi0,ddof=1) / np.sqrt(sum(keep))
            seqte = np.std(psi1-psi0,ddof=1) / np.sqrt(sum(keep))
            result.append(pd.DataFrame({'gamma':gamma,'q1':q1,'q0':q0,'qte':q1-q0,\
                                        'se1':se1,'se0':se0,'seqte':seqte},index=[i]))
        self.ldml_estimation=pd.concat(result,axis=0)
        return self.ldml_estimation

if __name__=='__main__':
    pass