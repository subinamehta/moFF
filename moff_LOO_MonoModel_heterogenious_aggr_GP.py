import ConfigParser
import argparse
import ast
import bisect
import copy
import itertools
import logging
import os
import re
from itertools import chain, combinations

import GPy
import GPflow
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score ,mean_absolute_error
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.model_selection import KFold

from pyds import MassFunction


## filtering _outlier
def MahalanobisDist(x, y):
    covariance_xy = np.cov(x, y, rowvar=0)
    inv_covariance_xy = np.linalg.inv(covariance_xy)
    xy_mean = np.mean(x), np.mean(y)
    x_diff = np.array([x_i - xy_mean[0] for x_i in x])
    y_diff = np.array([y_i - xy_mean[1] for y_i in y])
    diff_xy = np.transpose([x_diff, y_diff])

    md = []
    for i in range(len(diff_xy)):
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]), inv_covariance_xy), diff_xy[i])))
    return md


## tookit function for search in asorted list
def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

def find_lt(a, x):
    'Find rightmost value less than x'
    i = bisect.bisect_left(a, x)
    if i:
        return a[i-1]
    raise ValueError

def find_le(a, x):
    'Find rightmost value less than or equal to x'
    i = bisect.bisect_right(a, x)
    if i:
        return a[i-1]
    raise ValueError

def find_gt(a, x):
    'Find leftmost value greater than x'
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return a[i]
    raise ValueError

def find_ge(a, x,total):
    'Find leftmost item greater than or equal to x'
    i = bisect.bisect_left(a, x)

    if (total[0,i] <  x) and (total[2,i] > x) :
        return a[i]
    else:
        if  (total[0,i-1] <  x) and (total[2,i-1] > x) :
            return a[i-1]
        else:
            return a[i +1]



## remove outlier
def MD_removeOutliers(x, y, width):
    MD = MahalanobisDist(x, y)
    threshold = np.mean(MD) * float(width)  # adjust 1.5 accordingly
    nx, ny, outliers = [], [], []
    for i in range(len(MD)):
        if MD[i] <= threshold:
            nx.append(x[i])
            ny.append(y[i])
        else:
            outliers.append(i)  # position of removed pair
    return (np.array(nx), np.array(ny), np.array(outliers))



def prediction_one_model_agreggated_gp(x, model,log):
   
    app= x['code_unique']
       
    #aa_l = len(x['peptide'])
    
    x = x.filter(regex=("rt_*"))
    
    #print x.isnull().sum()
    x['mean_rt']= x.mean()
    x['var_rt']=x.var()
    #print x
    #if  x.isnull().sum()  == 9:
     #   print x
    if app =='ESGIIQGDLIAK_1242.6819_2' : 
        print x
    #x.fillna(x.mean(),inplace=True)
    x_point_t = x[['mean_rt','var_rt']].values
    #x_point_t = np.append(x_point_t,   aa_l )
    #print x_point_t
    
    x_point_t= np.reshape( x_point_t,(1,x_point_t.shape[0]) )
    
    
    
    mean, var = model.predict_y(  x_point_t) 

    if app =='ESGIIQGDLIAK_1242.6819_2' : 
        print mean, var 

    #print mean 
    #print mean, mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0])
    
    return mean[0][0]
    


# combination of rt predicted by each single model
def prediction_one_model(x, model,log,flag):
   
    x = x.filter(regex=("rt_*"))

    
    x.fillna(x.mean(),inplace=True)
    #print x

    x = x.values.reshape(1 , x.shape[0] )
    if flag==2:
        pred ,var =  model.predict_y( x)
    else:
        pred = model.predict(x)
        # output not weight
    return  pred[0][0]

#
#retunr the half_width of the predicted rt
# (l_bound -u_bound) /2
#
def prediction_rt_unc_wind_one_model (x, model,log):
    x = x.values.reshape(1, x.shape[0])

    ress = model.predict_quantiles(x)
    return (abs(ress[0]-ress[1]) /2 )[0][0]


# check columns read  from  a properties file 
def check_columns_name(col_list, col_must_have):
    for c_name in col_must_have:
        if not (c_name in col_list):
            # fail
            return 1
    # succes
    return 0








def train_models_2_linear(data_A,data_B):
     ## Ridge Regression Skikit
    '''
    clf = linear_model.RidgeCV(alphas=np.power(2, np.linspace(-30, 30)), scoring='mean_absolute_error', cv =None)
    clf.fit(data_B, data_A)
    #print ( ' alpha of the CV ridge regression model %4.4f',clf.alpha_)
    clf_final = linear_model.Ridge(alpha=  clf.alpha_, solver='svd',random_state=1, fit_intercept=False)
    clf_final.fit(data_B, data_A)
     '''
    clf_final = linear_model.LinearRegression()
    clf_final.fit(data_B, data_A)

    return clf_final, clf_final.predict(data_B)


def train_models_2_ridge(data_A,data_B):
     ## Ridge Regression Skikit

    clf = linear_model.RidgeCV(alphas=np.power(2, np.linspace(-30, 30)), scoring='neg_mean_absolute_error', cv =None)
    clf.fit(data_B, data_A)
    #print ( ' alpha of the CV ridge regression model %4.4f',clf.alpha_)
    clf_final = linear_model.Ridge(alpha=  clf.alpha_, solver='svd',random_state=1, fit_intercept=True)
    clf_final.fit(data_B, data_A)



    return clf_final, clf_final.predict(data_B)

def train_local_gp(data_A,data_B,n_cluster_input):
    kmeans = KMeans(n_clusters=n_cluster_input, random_state=0).fit(data_B)
    models=[]
    for k_c in range(0,n_cluster_input):
        print 'cluster ', k_c, 'size', data_B[np.where(kmeans.labels_ == k_c)].shape
        print  'RT of the clustermax min :',data_A[np.where(kmeans.labels_ == k_c)].min(), data_A[np.where(kmeans.labels_ == k_c)].max()
        print  'RT Delta in the cluste r: ',data_A[np.where(kmeans.labels_ == k_c)].max() - data_A[np.where(kmeans.labels_ == k_c)].min()
        print '--'
        models.append( train_models_GP_flow(data_A[np.where(kmeans.labels_ == k_c)],data_B[np.where(kmeans.labels_ == k_c)] ))

    return models ,kmeans

def train_models_GP_flow(data_A,data_B):
    print np.min(data_B[:,0]),np.max(data_B[:,0])
    print np.min(data_B[:,1]),np.max(data_B[:,1])
    k = GPflow.kernels.RBF(input_dim=2)
    m = GPflow.gpr.GPR(data_B, data_A, kern=k)
    m.kern.variance.prior = GPflow.priors.Gamma(10,1)
    m.kern.lengthscales.prior = GPflow.priors.Gamma(10,0.1)
    m.optimize()
    ym_train_predicted, var = m.predict_y(  data_B)

    return m , ym_train_predicted

def train_models_3(data_A,data_B):

    ## random sampling

    #size_train= int (data_A.shape[0] * 1)
    #print size_train, 'over',  data_A.shape[0]
    #rows = random.sample(range(data_A.shape[0]), size_train )
    #data_A= data_A[rows,:]
    #data_B= data_B[rows,:]


    #rows = None
    #print 'size_train :', data_A.shape[0]
    ## GP by pyGP
    #kern = GPy.kern.Linear(data_B.shape[1])
    kern = GPy.kern.RBF(input_dim=10, lengthscale=0.25, variance=0.03)
    k_bias= GPy.kern.Bias(data_B.shape[1],variance=1.5)
    #kern = GPy.kern.Matern32(input_dim=1, lengthscale=0.05, variance=0.23)
    k_add = kern * k_bias
    #kern.set_prior(GPy.priors.Gaussian(2000,30))
    m = GPy.models.GPRegression(data_B, data_A,k_add)
    #gamma_prior = GPy.priors.Gamma.from_EV(, 2)
    #prior = GPy.priors.Gaussian(2000,30)
    #m.rbf.variances.set_prior(prior)
    #m.constrain_positive('*')
    m.optimize_restarts(num_restarts=1,num_processes=4)
    #m.optimize_restarts(num_restarts = 3)
    #print m
    ym_train_predicted, y_var = m.predict(data_B)


    return m, ym_train_predicted


  

def data_inputing(common):

    fill_value = pd.DataFrame({col: common.ix[:, :].mean(axis=1) for col in common.columns})
    # print  fill_value
    common.fillna(fill_value, inplace=True)


    return common

# combination of rt predicted by each single model
def combine_model(x, model, err, weight_flag,log):
    # x = x.values
    #print x
    #tot_err =  1- ( (np.array(err)[np.where(~np.isnan(x))]) / np.max(np.array(err)[np.where(~np.isnan(x))]))
    tot_err = np.sum(np.array(err)[np.where(~np.isnan(x))])
    #print tot_err
    #print x
    app_sum = 0
    app_sum_2 = 0
    for ii in range(0, len(x)):
        #print ii
        #print model[ii].params
        #log.info(' %i Input Rt  %s  Predicted: %s ',ii, x[ii],model[ii].predict(x[ii])[0])
        #print (' %i Input Rt  %s  Predicted: %s ',ii, x[ii],model[ii].predict(x[ii])[0])
        if ~  np.isnan(x[ii]):
            if int(weight_flag) == 0:
                app_sum = app_sum + (model[ii].predict(x[ii])[0])
            else:
                #print ii,model[ii].predict(x[ii])[0][0]
                w = (float(err[ii]) / float(tot_err))
                #w= tot_err[ii]
                #print ii ,'weighted', (model[ii].predict(x[ii])[0][0] * w ),w
                app_sum_2 = app_sum_2 + (model[ii].predict(x[ii])[0] * w )

    # " output weighted mean
    ##
    if int(weight_flag) == 1:
        return float(app_sum_2)
    else:
        # output not weight
        return float(app_sum) / float(np.where(~ np.isnan(x))[0].shape[0])


def train_single_model_linear(A,B,log,index):
    common = pd.merge(A, B, on=['code_unique'], how='inner')

    if int(args.out_flag) == 1:
        filt_x, filt_y, pos_out = MD_removeOutliers(common['rt_y'].values, common['rt_x'].values,args.w_filt)
        data_B = filt_x
        data_A = filt_y
        data_B = np.reshape(data_B, [filt_x.shape[0], 1])
        data_A = np.reshape(data_A, [filt_y.shape[0], 1])
        log.info('Outlier founded %i  w.r.t %i', pos_out.shape[0], common['rt_y'].shape[0])
        print ('Outlier founded %i  w.r.t %i', pos_out.shape[0], common['rt_y'].shape[0])
        #cc = pd.DataFrame({'rt_y':filt_y, 'rt_x':filt_x})
        #cc.to_csv("D:\\workspace\\Lukas_data\\"+ str(i[0]) +"_vs_" + str(i[1]) + "_filt_"+  args.w_filt +".txt",sep='\t',index=False)
    else:
        data_B = common['rt_y'].values
        data_A = common['rt_x'].values
        #if i[1]==1:
        #common.to_csv("D:\\workspace\RT_data_benchmark\\NOTEBOOK-GP_test_stuff_doc\\"+ str(i[0]) +"_vs_" + str(i[1]) + ".txt",sep='\t',index=False)
        data_B = np.reshape(data_B, [common.shape[0], 1])
        data_A = np.reshape(data_A, [common.shape[0], 1])
    log.info(' Size training shared peptide , %i %i ', data_A.shape[0], data_B.shape[0])

    # version with confidence interval
    #model , predicted_train=  train_models_1(data_A,data_B)
    #model , predicted_train = train_models_1_robust(data_A,data_B)
    #model , predicted_train = train_models_1_robustLinearModel(data_A,data_B)

    # version with Ridge regression
    model , predicted_train =  train_models_2_ridge(data_A,data_B)

    err = mean_absolute_error(data_A,predicted_train )
    #print index,data_A.shape, err

    return model,err


## run the mbr in LOO mode  moFF : input  ms2 identified peptide   output csv file with the matched peptides added
def run_mbr(args):
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    if not (os.path.isdir(args.loc_in)):
        exit(str(args.loc_in) + '-->  input folder does not exist ! ')

    filt_outlier = args.out_flag

    if str(args.loc_in) == '':
        output_dir = 'mbr_output'
    else:
        if '/' in str(args.loc_in):

            ## lukas data
            output_dir = "../RT_data_benchmark/"
                      #   "#MonoModel_missing_fillrowmean_RR\\"


            #output_dir ="D:\\bench_mark_dataset\\saved_result_ev"
                        #saved_result_ev"
            #output_dir = str(args.loc_in) + '\\mbr_output'
        else:
            exit(str(args.loc_in) + ' EXIT input folder path not well specified --> / missing ')

    if not (os.path.isdir(output_dir)):
        print "Created MBR output folder in ", output_dir
        os.makedirs(output_dir)
    else:
        print "MBR Output folder in :", output_dir

    config = ConfigParser.RawConfigParser()
    # it s always placed in same folder of moff_mbr.py
    config.read('moff_setting.properties')
    ## name of the input file
    exp_set = []
    # list of the input dataframe
    exp_t = []
    # list of the output dataframe
    exp_out = []
    # list of input dataframe used as help
    exp_subset = []
    for root, dirs, files in os.walk(args.loc_in):
        for f in files:
            if f.endswith('.' + args.ext):
                exp_set.append(os.path.join(root, f))
    if not ((args.sample) == None):
        print re.search(args.sample, exp_set[0])
        exp_set_app = copy.deepcopy(exp_set)
        for a in exp_set:
            if (re.search(args.sample, a) == None):
                exp_set_app.remove(a)
        exp_set = exp_set_app
    if (exp_set == []) or (len(exp_set) == 1):
        print exp_set
        exit(
            'ERROR input files not found or just one input file selected . check the folder or the extension given in input')
    min_RT= 100
    max_RT= 0
    for nn,a in enumerate(exp_set):

        print 'Reading file.... ', a
        exp_subset.append(a)
        data_moff = pd.read_csv(a, sep="\t", header=0)
        list_name = data_moff.columns.values.tolist()
        if check_columns_name(list_name, ast.literal_eval(config.get('moFF', 'col_must_have_x'))) == 1:
            exit('ERROR minimal field requested are missing or wrong')
        data_moff['matched'] = 0
        ## pay attantion on this conversion
        data_moff['mass'] = data_moff['mass'].map('{:4.4f}'.format)
        data_moff['prot'] = data_moff['prot'].astype(str)
        #data_moff['prot']=  data_moff['prot'].apply(lambda x:  x.split('|')[1] )
        data_moff['charge'] = data_moff['charge'].astype(int)
        data_moff['code_unique'] = data_moff['peptide'].astype(str) + '_' + data_moff['mass_th'].astype(str) +'_' + data_moff['charge'].astype(str)
        #data_moff['code_unique'] = data_moff['omegamodpep'].astype(str) + '_' + data_moff['mass'].astype(str) + '_' + data_moff['charge'].astype(str)
        data_moff['code_share'] = data_moff['peptide'].astype(str) + '_' + data_moff['mass_th'].astype(str) +'_' + data_moff['charge'].astype(str) + '_' + data_moff['prot'].astype(str)

        # sort in pandas 1.7.1
        data_moff = data_moff.sort(columns='rt')

        exp_t.append(data_moff)
        exp_out.append(data_moff)
        ''' 
        # Discretization of the RT space: get the max and the min valued
        if data_moff['rt'].min() <= min_RT:
            min_RT = data_moff['rt'].min()
        if data_moff['rt'].max() >= max_RT:
            max_RT = data_moff['rt'].max()
        #print data_moff['rt'].min(), data_moff['rt'].max()
        '''

    intersect_share = reduce( np.union1d, ([x['code_unique'].unique() for x in exp_t]) )


    print intersect_share.shape
    kf = KFold(n_splits=3)
    kf.get_n_splits(intersect_share)
    print (kf)
    #for train_index, test_index in kf.split(intersect_share):
      #  print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
        #X_train, X_test = intersect_share[train_index], intersect_share[test_index]
        #y_train, y_test = y[train_index], y[test_index]


    #exit('--debug')
    

    ### ---- fine checking
    print 'Read input --> done '
    n_replicates =  len(exp_t)
    exp_set = exp_subset
    aa = range(0, n_replicates)
    out = list(itertools.product(aa, repeat=2))


    log_mbr = logging.getLogger('MBR module')
    log_mbr.setLevel(logging.INFO)
    w_mbr = logging.FileHandler(args.loc_in + '/' + args.log_label + '_' + '_monoModel_mbr_.log', mode='w')
    w_mbr.setLevel(logging.INFO)
    log_mbr.addHandler(w_mbr)

    log_mbr.info('Filtering is %s : ', 'active' if args.out_flag == 1 else 'not active')
    log_mbr.info('Number of replicates %i,', n_replicates)
    log_mbr.info('Pairwise model computation ----')
    ## This is not always general

    c_data= copy.deepcopy(exp_t)

    # it contains the fix_rep values
    #fix_rep=9

    #print 'MATCHING between RUN for  ', exp_set[fix_rep]
    ## list_inter: contains the  size of the interval
    ## pep_out : feature leaved out
    for fix_rep in [0]:
        fold_c= 0
        for train_index, test_index in kf.split(intersect_share):
        
            print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
            X_train, X_test = intersect_share[train_index], intersect_share[test_index]
            out_df =pd.DataFrame(columns=['prot', 'peptide', 'charge', 'mz', 'rt_0', 'rt_1', 'rt_2','rt_3', 'rt_4', 'rt_5', 'rt_6', 'rt_7', 'rt_8', 'rt_9','time_base',  'rt'])
            model_save=[]
            model_err=[]

            print 'MATCHING between RUN for  ', exp_set[fix_rep], ' fold', fold_c
    
        
            #print '-- ', pep_out ,'-- '
            ## binning  of the RT space in
            list_inter = 130

            ## fix_rep input della procedura
            exp_t[fix_rep ]= c_data[fix_rep]
            #exp_t= c_data
            exp_t[fix_rep] = exp_t[fix_rep][~exp_t[fix_rep].code_unique.isin(X_test)]
            print c_data[fix_rep].shape, exp_t[fix_rep].shape, X_test.shape

            list_name.append('matched')
            for jj in [fix_rep]:
                first = True
                #temp_Data=[]
                temp_data_target=[]
                temp_data_input=[]
                for i in out:
                    if i[0] == fix_rep and i[1] != jj:
                        #log_mbr.info('  Matching  %s peptide in   searching in %s ', exp_set[i[0]], exp_set[i[1]])
                        #print ( exp_set[i[0]], exp_set[i[1]])
                        list_pep_repA = exp_t[i[0]]['code_unique'].unique()
                        list_pep_repB = exp_t[i[1]]['code_unique'].unique()
                        log_mbr.info(' Peptide unique (mass + sequence) %i , %i ', list_pep_repA.shape[0],
                                     list_pep_repB.shape[0])
                        #set_dif_s_in_1 = np.setdiff1d(list_pep_repB, list_pep_repA)
                        #add_pep_frame = exp_t[i[1]][exp_t[i[1]]['code_unique'].isin(set_dif_s_in_1)].copy()
                        pep_shared = np.intersect1d(list_pep_repA, list_pep_repB)
                        #log_mbr.info('  Peptide (mass + sequence)  added size  %i ', add_pep_frame.shape[0])
                        log_mbr.info('  Peptide (mass + sequence) )shared  %i ', pep_shared.shape[0])
                        comA = exp_t[i[0]][exp_t[i[0]]['code_unique'].isin(pep_shared)][
                            ['code_unique', 'peptide', 'prot', 'rt']]
                        comB = exp_t[i[1]][exp_t[i[1]]['code_unique'].isin(pep_shared)][
                            ['code_unique', 'peptide', 'prot', 'rt']]

                        print  i[1] ,comB[comB['code_unique']=='ESGIIQGDLIAK_1242.6819_2']
                            

                        ## check the varience for each code unique and gilter only the highest 99
                        ##IMPORTant
                        flag_var_filt = True
                        if flag_var_filt :
                            dd = comA.groupby('code_unique', as_index=False)
                            top_res = dd.agg(['std','mean','count'])
                            #print np.nanpercentile(top_res['rt']['std'].values,[5,10,20,30,50,60,80,90,95,97,99,100])
                            th = np.nanpercentile(top_res['rt']['std'].values,90)
                            comA = comA[~ comA['code_unique'].isin(top_res[top_res['rt']['std'] > th].index)]
                            # data B '
                            dd = comB.groupby('code_unique', as_index=False)
                            
                            top_res = dd.agg(['std','mean','count'])
                            print comB.shape
                            print np.nanpercentile(top_res['rt']['std'].values,[5,10,20,30,50,60,80,90,95,97,99,100])
                            th = np.nanpercentile(top_res['rt']['std'].values,90)

                            comB = comB[~ comB['code_unique'].isin(top_res[top_res['rt']['std'] > th].index)]
                            print 'after cariance filter',comB.shape
                        ##end variance filtering
                        comA = comA.groupby('code_unique', as_index=False).mean()
                        comB = comB.groupby('code_unique', as_index=False).mean()

                        m, err= train_single_model_linear (comA,comB,log_mbr,i[1])
                        model_save.append( m )
                        model_err.append(err)
                        #common = pd.merge(comA, comB, on=['code_unique'], how='inner')
                        temp_data_target.append(comA[['code_unique', 'rt']])
                        temp_data_input.append(comB[['code_unique', 'rt']])

            print len(temp_data_input)
            
            '''
            # taking the intersect for the trainning set no missing 
            tt = reduce(np.intersect1d,([x.code_unique.values for x in temp_data_target]))
            df=pd.DataFrame(index= tt)
            for i in range(0,len(temp_data_input)):
                df[str(i)]= temp_data_input[i][ temp_data_input[i]['code_unique'].isin(tt)]['rt'].values
            X = df.ix[:,0:df.shape[0]].values
            Y = temp_data_target[i][ temp_data_target[i]['code_unique'].isin(tt)]['rt'].values
            df.to_csv( 'fold_0_rep0Target_intersectdataset.data', sep='\t', index=True )
            '''
            ## union with missing values
            tt = reduce(np.intersect1d, ([x.code_unique.values for x in temp_data_target]))
            #print tt[0:1]
            aa_len =  [ len(x.split('_')[0]) for x in tt ]
            
            df=pd.DataFrame(index= tt)
            df2 =pd.DataFrame(index= tt)
            df2['Y'] = np.nan
            for i in range(0,len(temp_data_input)):
                df[str(i)]= np.nan
                df.ix[np.intersect1d(tt,temp_data_input[i]['code_unique']),i ] = temp_data_input[i][ temp_data_input[i]['code_unique'].isin(tt)]['rt'].values
                df2.ix[ np.intersect1d(tt,temp_data_target[i]['code_unique']),0 ] = temp_data_target[i][ temp_data_target[i]['code_unique'].isin(tt)]['rt'].values


            #option for data export
            #df['Y'] = df2['Y']
            #df.to_csv('fold_0_rep0Target_uniondataset.data', sep='\t', index=True)
            #
            
            df['mean_rt']= df.mean(axis=1)
            df['var_rt']=df.var(axis=1)

            ## check variance of training set
            print 'variance aggregation trainnin set'
            print np.nanpercentile(df['var_rt'].values,[5,10,20,30,50,60,80,90,95,97,99,100])

            #df['aa_len']=aa_len
            df.to_csv('fold_0_rep0Target_uniondataset.data', sep='\t', index=True)
            
            ## filling Na value
            #df= data_inputing(df)
            
            X=df[['mean_rt','var_rt']].values
            Y = df2['Y'].values
            
            del df2

        
            X = np.reshape(X, [X.shape[0], X.shape[1]])
            Y = np.reshape(Y, [Y.shape[0], 1])
            
            
            print X.shape,Y.shape
            
            
            ## train model
            #model ,predicted_train  = train_models_2_ridge(Y,X)
            #selec_model=1
            #print model.intercept_, model.coef_
            #print X.shape, mean_absolute_error(Y,predicted_train )
            #'''
            # original model with GPy
            #model, predicted_train = train_models_3(Y, X)
            #GP flow
            model,predicted_train = train_models_GP_flow(Y,X)
            print predicted_train.shape
            
            # local GP version GPflow
            #models, cluster_model = train_local_gp(Y,X,7)
            # not used in local GP
            selec_model=2
            print 'TRainning error ' , X.shape, mean_absolute_error( Y, np.reshape(predicted_train, [predicted_train.shape[0], 1]) )
            #'''

            ## from here we have to deal with one one model



            ## test part id here
            diff_field = np.setdiff1d(exp_t[0].columns, ['matched', 'peptide', 'mass', 'mz', 'charge', 'prot', 'rt'])
            for jj in [fix_rep]:
                #c_pred += 1
                #if c_pred == 2:
                #    exit()
                pre_pep_save = []
                c_rt = 0
                for i in out :
                    if i[0] == fix_rep and i[1] != jj:
                        log_mbr.info('Matching peptides found in  %s ', exp_set[i[1]])
                        add_pep_frame = exp_t[i[1]][exp_t[i[1]]['code_unique'].isin(X_test).copy()]
                        ## custom case
                        # 'mz'
                        add_pep_frame = add_pep_frame[['peptide',  'charge','mass_th','prot', 'rt']]
                        add_pep_frame = add_pep_frame.groupby(['peptide','charge','mass_th','prot'],as_index=False).mean()
                        print add_pep_frame.shape

                        ## normal case
                        #add_pep_frame = add_pep_frame[['peptide', 'mass', 'mz', 'charge', 'prot', 'rt']]
                        # I do not need for this data
                        #add_pep_frame['charge']=add_pep_frame['charge'].astype(int)
                        #add_pep_frame['code_unique'] = add_pep_frame['peptide'] + '_' + add_pep_frame['prot'] + '_' +  add_pep_frame['mass'].astype(str) + '_' + add_pep_frame['charge'].astype( str)

                        ## withput ptotein in the key
                        #add_pep_frame['code_unique'] = add_pep_frame['omegamodpep'] + '_' + add_pep_frame['mass'].astype(str) + '_' + add_pep_frame['charge'].astype( str)
                        add_pep_frame['code_unique'] = add_pep_frame['peptide'] + '_'  + add_pep_frame['mass_th'].astype( str) + '_' + add_pep_frame[
                            'charge'].astype(str)


                        #add_pep_frame = add_pep_frame.groupby('code_unique', as_index=False)['peptide', 'mass', 'charge', 'prot', 'rt'].aggregate(max)
                        # 'mz'
                        add_pep_frame = add_pep_frame[['code_unique','peptide',  'charge', 'prot', 'rt']]
                        list_name = add_pep_frame.columns.tolist()
                        list_name = [w.replace('rt', 'rt_' + str(c_rt)) for w in list_name]
                        add_pep_frame.columns = list_name
                        pre_pep_save.append(add_pep_frame)

                        c_rt += 1
           # print 'input columns',pre_pep_save[0].shape
            if n_replicates == 2:
                test = pre_pep_save[0]
            else:
                #'mz',

                    test = reduce(lambda left, right: pd.merge(left, right, on=['code_unique','peptide',  'charge', 'prot'], how='outer'),pre_pep_save)



            if test.empty:
                print 'something wrong !!! --skip with ',pep_out
                continue
            #if test.shape[0]>1:
            #    print 'nooo'

            #test['time_base']= 100
           
            test['time_base'] = test[['rt_0','rt_1','rt_2','rt_3','rt_4','rt_5','rt_6','rt_7','rt_8','rt_9']].apply( lambda x: combine_model(x, model_save,model_err, args.w_comb ,log_mbr) ,axis=1)
            #test['time_base'] = test.ix[:,:].apply( lambda x: prediction_one_model(x, model,log_mbr,selec_model) ,axis=1)
            #  local GP
            test['time_aggr'] = test.ix[:,:].apply( lambda x: prediction_one_model_agreggated_gp(x, model,log_mbr) ,axis=1)
            #test['rt_pred_width'] = test.ix[:, 4: (4 + (n_replicates - 1))].apply( lambda x: prediction_rt_unc_wind_one_model(x, model, log_mbr), axis=1)
            #test['mixed_model'] =test.ix[:,:].apply( lambda x: prediction_mixed_model(x, model,model_save,model_err, args.w_comb ,log_mbr) ,axis=1)
            test['matched'] = 1
            # test= test[['peptide','mass','mz','charge','prot','rt']]
            #print 'original feature ', c_data[fix_rep][c_data[fix_rep].code_unique == pep_out][['prot','rt'] ].shape
            test = test.merge(c_data[fix_rep][c_data[fix_rep].code_unique.isin(X_test)][['prot','rt','code_unique'] ],on=['code_unique','prot'],how='inner')
            out_df= pd.concat([out_df,test], join='outer', axis=0)
            
    ## print the entire file
    ## the file contains only the shared peptide LOO founded
            fold_c += 1 
    #print 'final dataframe',out_df.shape
            out_df.to_csv(path_or_buf=  'rep_0_fold_' + str(fold_c) + '_intersect_unionDATA_2Aggrfeature__monoModel_match.txt',sep='\t',index=False)
    #print 'rimetto i pep tolti per LO0'
            exp_t[fix_rep ]= c_data[fix_rep]
    exit('debug')
    #log_mbr.info('Before adding %s contains %i ', exp_set[jj], exp_t[jj].shape[0])
    #exp_out[jj] = pd.concat([exp_t[jj], test], join='outer', axis=0)
    #log_mbr.info('After MBR %s contains:  %i  peptides', exp_set[jj], exp_out[jj].shape[0])
    log_mbr.info('----------------------------------------------')
    #print 'added LOO feature ',out_df.shape, 'for '
        #exp_out[jj].to_csv(
         #   path_or_buf=output_dir + '/' + str(os.path.split(exp_set[jj])[1].split('.')[0]) + '_match.txt', sep='\t',
         #   index=False)
        # print os.path.split(exp_set[0])[1].split('.')[0]
    if out_df.shape[0] > 0:
        return 1
    else:
        return -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='moFF match between run input parameter')

    parser.add_argument('--inputF', dest='loc_in', action='store',
                        help='specify the folder of the input MS2 peptide files  REQUIRED]', required=True)

    parser.add_argument('--sample', dest='sample', action='store',
                        help='specify which replicate files are used fot mbr [regular expr. are valid] ',
                        required=False)

    parser.add_argument('--ext', dest='ext', action='store', default='txt',
                        help='specify the extension of the input file (txt as default value) ', required=False)

    parser.add_argument('--log_file_name', dest='log_label', default='moFF', action='store',
                        help='a label name for the log file (moFF_mbr.log as default log file name) ', required=False)

    parser.add_argument('--filt_width', dest='w_filt', action='store', default=2,
                        help='width value of the filter (k * mean(Dist_Malahobi , k = 2 as default) ', required=False)

    parser.add_argument('--out_filt', dest='out_flag', action='store', default=1,
                        help='filter outlier in each rt time allignment (active as default)', required=False)

    parser.add_argument('--weight_comb', dest='w_comb', action='store', default=0,
                        help='weights for model combination combination : 0 for no weight (default) 1 weighted devised by model errors.',
                        required=False)

    args = parser.parse_args()

    check = run_mbr(args)
    #print check
    exit()