import ConfigParser
import argparse
import ast
import copy
import itertools
import logging
import os
import re

import GPy
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error


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



def combine_model_GP(x, model, err, weight_flag,log):
    # x = x.values
    #tot_err =  1- ( (np.array(err)[np.where(~np.isnan(x))]) / np.max(np.array(err)[np.where(~np.isnan(x))]))
    tot_err = np.sum(np.array(err)[np.where(~np.isnan(x))])
    #print tot_err
    #print x
    app_sum = 0
    app_sum_2 = 0
    for ii in range(0, len(x)):
        pred , var= model[ii].predict(x[ii].reshape(1,1))
        log.info(' %i Input Rt  %r  Predicted: %r ',ii,x[ii] , pred)
        if ~  np.isnan(x[ii]):
            if int(weight_flag) == 0:

                app_sum = app_sum + (pred)
            else:
                #print ii,model[ii].predict(x[ii])[0][0]
                w = (float(err[ii]) / float(tot_err))
                #w= tot_err[ii]
                #print ii ,'weighted', (model[ii].predict(x[ii])[0][0] * w ),w
                app_sum_2 = app_sum_2 + (pred * w )

    # " output weighted mean
    ##
    if int(weight_flag) == 1:
        return float(app_sum_2)
    else:
        # output not weight
        return float(app_sum) / float(np.where(~ np.isnan(x))[0].shape[0])



# combination of rt predicted by each single model
def prediction_model(x, model,log):

    app_sum = 0
    for ii in range(0, len(x)):
        log.info(' %i Input Rt  %s  Predicted: %s ',ii, x[ii],model.predict(x[ii])[0])
        app_sum = app_sum + (model.predict(x[ii])[0])

    return float(app_sum) / float(np.where(~ np.isnan(x))[0].shape[0])


# check columns read  from  a properties file 
def check_columns_name(col_list, col_must_have):
    for c_name in col_must_have:
        if not (c_name in col_list):
            # fail
            return 1
    # succes
    return 0

def train_models_1_robustLinearModel(data_A,data_B):
     ## base version  with stat model.
     # weighting model lin regression regression
     ## WARNING paramert are in inverse order

    clf_final = sm.RLM(data_A,data_B,M=sm.robust.norms.AndrewWave()).fit()
    #print clf_final
    y = clf_final.predict(data_B)
    #print y.shape
    y= np.reshape(y, [y.shape[0], 1])
    #print y.shape
    return clf_final,y


def train_models_1_robust(data_A,data_B):
     ## base version  with stat model.
     # weighting model lin regression regression
     ## WARNING paramert are in inverse order

    clf_final = sm.WLS(data_A,data_B,weights=list(range(1,data_B.shape[0]+1))).fit()
    #print clf_final
    y = clf_final.predict(data_B)
    #print y.shape
    y= np.reshape(y, [y.shape[0], 1])
    #print y.shape
    return clf_final,y


def train_models_1(data_A,data_B):
     ## base version  with stat model.
     # basic lin regression regression
     ## WARNING paramert are in inverse order
    clf_final = sm.OLS(data_A,data_B ).fit()
    y = clf_final.predict(data_B)
    y= np.reshape(y, [y.shape[0], 1])
    return clf_final,y


def train_models_2(data_A,data_B):
     ## Ridge Regression Skikit
    clf = linear_model.RidgeCV(alphas=np.power(2, np.linspace(-30, 30)), scoring='mean_absolute_error')
    clf.fit(data_B, data_A)
    # log_mbr.info( ' alpha of the CV ridge regression model %4.4f',clf.alpha_)
    clf_final = linear_model.Ridge(alpha=clf.alpha_)
    clf_final.fit(data_B, data_A)

    return clf_final, clf_final.predict(data_B)


def train_models_3(data_A,data_B):
    ''' to get the confidence interval!
    bound_l  =  ym_train_predicted - 1.9600 * np.sqrt(y_var )
    bound_u =  ym_train_predicted + 1.9600 * np.sqrt(y_var)

    bound_l_80  =  ym_train_predicted - 1.2800 * np.sqrt(y_var )
    bound_u_80 =  ym_train_predicted + 1.2800 * np.sqrt(y_var)
    '''
     ## GP by pyGP
    kern = GPy.kern.Linear(1)
    m = GPy.models.GPRegression(data_B, data_A,kern)
    #gamma_prior = GPy.priors.Gamma.from_EV(3, 2)
    #m.linear.variances.set_prior(gamma_prior)
    #m.Gaussian_noise.variance.constrain_bounded(0.5, 10)
    m.optimize()
    ym_train_predicted, y_var = m.predict(data_B)


    return m, ym_train_predicted




## run the mbr in LOO mode  moFF : input  ms2 identified peptide   output csv file with the matched peptides added
def run_mbr(args):
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    if not (os.path.isdir(args.loc_in)):
        exit(str(args.loc_in) + '-->  input folder does not exist ! ')

    filt_outlier = args.out_flag

    if str(args.loc_in) == '':
        output_dir = 'mbr_output'
    else:
        if '\\' in str(args.loc_in):
            output_dir ="D:\\bench_mark_dataset\\result_LOO_OneModel"
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
        data_moff['mass'] = data_moff['mass'].map('{:.4f}'.format)
        data_moff['prot']=  data_moff['prot'].apply(lambda x:  x.split('|')[1] )
        data_moff['charge'] = data_moff['charge'].astype(int)
        data_moff['code_unique'] = data_moff['peptide'].astype(str) + '_' + data_moff['mass'].astype(str) + '_' + data_moff['charge'].astype(str)
        data_moff['code_share'] = data_moff['peptide'].astype(str) + '_' + data_moff['mass'].astype(str) + '_' + data_moff['charge'].astype(str) + '_' + data_moff['prot'].astype(str)

        # sort in pandas 1.7.1
        data_moff = data_moff.sort(columns='rt')
        if nn ==0:
            intersect_share =  data_moff[data_moff['prot']=='16128008']['code_share'].unique()
        else:
            intersect_share = np.intersect1d( data_moff[data_moff['prot']=='16128008']['code_share'],intersect_share)
        exp_t.append(data_moff)
        exp_out.append(data_moff)
        # Discretization of the RT space: get the max and the min valued
        if data_moff['rt'].min() <= min_RT:
            min_RT = data_moff['rt'].min()
        if data_moff['rt'].max() >= max_RT:
            max_RT = data_moff['rt'].max()
        #print data_moff['rt'].min(), data_moff['rt'].max()


    print intersect_share.shape
    print intersect_share
    # most abundant protein :-> 16128008
    ##----------- for checking how the pep in the spec. proteins look likes / haow many
    #for kk in range(1):
    #    #print kk, exp_t[kk][ ( exp_t[kk]['prot']=='16128008' )&  (exp_t[kk]['code_unique'].isin(intersect_share[0:1]) )].shape
    #    print kk, exp_t[kk][  (exp_t[kk]['code_share'].isin(intersect_share) )]

    ### ---- fine checking
    print 'Read input --> done '
    n_replicates =  len(exp_t)
    exp_set = exp_subset
    aa = range(0, n_replicates)
    out = list(itertools.product(aa, repeat=2))


    log_mbr = logging.getLogger('MBR module')
    log_mbr.setLevel(logging.INFO)
    w_mbr = logging.FileHandler(args.loc_in + '/' + args.log_label + '_' + 'mbr_.log', mode='w')
    w_mbr.setLevel(logging.INFO)
    log_mbr.addHandler(w_mbr)

    log_mbr.info('Filtering is %s : ', 'active' if args.out_flag == 1 else 'not active')
    log_mbr.info('Number of replicates %i,', n_replicates)
    log_mbr.info('Pairwise model computation ----')
    ## This is not always general
    out_df =pd.DataFrame(columns=['prot', 'peptide', 'charge', 'mz', 'rt_0', 'rt_1', 'rt_2',
       'rt_3', 'rt_4', 'rt_5', 'rt_6', 'rt_7', 'rt_8', 'rt_9',
       'time_base', 'matched', 'rt'])
    c_data= copy.deepcopy(exp_t)

    # it contains the fix_rep values
    fix_rep=9

    print 'MATCHING between RUN for  ', exp_set[fix_rep]
    ## list_inter: contains the  size of the interval
    ## pep_out : feature leaved out
    for list_inter in [180]:
        for pep_out in intersect_share[0:1]   :

            exp_t[fix_rep ]= c_data[fix_rep]
            exp_t[fix_rep] = exp_t[fix_rep][exp_t[fix_rep].code_share != pep_out]
            #print 'after', exp_t[fix_rep].shape
            ##  palce here the inizialization
            ##
            # add matched columns
            list_name.append('matched')
            train_data =pd.DataFrame(columns=['rt_x', 'rt_y','code_unique','run_id'])
            for jj in [fix_rep]:
                for i in out:
                    #print i
                    if i[0] == fix_rep and i[1] != jj:
                        log_mbr.info('  Matching  %s peptide in   searching in %s ', exp_set[i[0]], exp_set[i[1]])
                        list_pep_repA = exp_t[i[0]]['code_unique'].unique()
                        list_pep_repB = exp_t[i[1]]['code_unique'].unique()
                        log_mbr.info(' Peptide unique (mass + sequence) %i , %i ', list_pep_repA.shape[0],
                                     list_pep_repB.shape[0])
                        set_dif_s_in_1 = np.setdiff1d(list_pep_repB, list_pep_repA)
                        add_pep_frame = exp_t[i[1]][exp_t[i[1]]['code_unique'].isin(set_dif_s_in_1)].copy()
                        pep_shared = np.intersect1d(list_pep_repA, list_pep_repB)
                        log_mbr.info('  Peptide (mass + sequence)  added size  %i ', add_pep_frame.shape[0])
                        log_mbr.info('  Peptide (mass + sequence) )shared  %i ', pep_shared.shape[0])
                        comA = exp_t[i[0]][exp_t[i[0]]['code_unique'].isin(pep_shared)][
                            ['code_unique', 'peptide', 'prot', 'rt']]
                        comB = exp_t[i[1]][exp_t[i[1]]['code_unique'].isin(pep_shared)][
                            ['code_unique', 'peptide', 'prot', 'rt']]
                        ## check the varience for each code unique and gilter only the highest 99
                        ##IMPORTant
                        flag_var_filt = True
                        if flag_var_filt :
                            dd = comA.groupby('code_unique', as_index=False)
                            top_res = dd.agg(['std','mean','count'])
                            #print np.nanpercentile(top_res['rt']['std'].values,[5,10,20,30,50,60,80,90,95,97,99,100])
                            th = np.nanpercentile(top_res['rt']['std'].values,99)
                            comA = comA[~ comA['code_unique'].isin(top_res[top_res['rt']['std'] > th].index)]
                            # data B '
                            dd = comB.groupby('code_unique', as_index=False)
                            top_res = dd.agg(['std','mean','count'])
                            th = np.nanpercentile(top_res['rt']['std'].values,99)
                            comB = comB[~ comB['code_unique'].isin(top_res[top_res['rt']['std'] > th].index)]
                        ##end variance filtering
                        comA = comA.groupby('code_unique', as_index=False).mean()
                        comB = comB.groupby('code_unique', as_index=False).mean()
                        common = pd.merge(comA, comB, on=['code_unique'], how='inner')
                        ##
                        common['run_id']= int(i[1])
                        train_data= pd.concat([train_data,common], join='outer', axis=0)




            train_data.to_csv('../export_OneModel_9vsOther.txt',sep='\t',index=False)
            exit('Debug')
            ## learning is done here
            if int(args.out_flag) == 1:
                filt_x, filt_y, pos_out = MD_removeOutliers(train_data['rt_y'].values, train_data['rt_x'].values, args.w_filt)
                data_B = filt_x
                data_A = filt_y
                data_B = np.reshape(data_B, [filt_x.shape[0], 1])
                data_A = np.reshape(data_A, [filt_y.shape[0], 1])
                log_mbr.info('Outlier founded %i  w.r.t %i', pos_out.shape[0], common['rt_y'].shape[0])
            else:
                data_B = train_data['rt_y'].values
                data_A =  train_data['rt_x'].values
                data_B = np.reshape(data_B, [train_data.shape[0], 1])
                data_A = np.reshape(data_A, [train_data.shape[0], 1])
            # training step
            # version with confidence interval
            #model , predicted_train=  train_models_1(data_A,data_B)
            #model , predicted_train = train_models_1_robust(data_A,data_B)
            #model , predicted_train = train_models_1_robustLinearModel(data_A,data_B)

            # version with Ridge regression
            model , predicted_train=  train_models_2(data_A,data_B)

            # GP process test
            #print 'GP test'
            #model , predicted_train=  train_models_3(data_A,data_B)
            log_mbr.info('Model Trained   --------')

            log_mbr.info(' Mean absolute error training : %4.4f sec', mean_absolute_error(data_A, predicted_train))
            diff_field = np.setdiff1d(exp_t[0].columns, ['matched', 'peptide', 'mass', 'mz', 'charge', 'prot', 'rt'])
            #c_pred = 0
            for jj in [fix_rep]:
                #c_pred += 1
                #if c_pred == 2:
                #    exit()
                pre_pep_save = []
                c_rt = 0
                for i in out:
                    if i[0] == fix_rep and i[1] != jj:
                        log_mbr.info('Matching peptides found  in  %s ', exp_set[i[1]])
                        add_pep_frame = exp_t[i[1]][exp_t[i[1]]['code_share']== pep_out  ].copy()
                        ## custom case
                        # 'mz'
                        add_pep_frame = add_pep_frame[['peptide', 'mass',  'charge', 'prot', 'rt']]
                        add_pep_frame = add_pep_frame.groupby(['peptide','mass','prot'],as_index=False).mean()

                        ## normal case
                        #add_pep_frame = add_pep_frame[['peptide', 'mass', 'mz', 'charge', 'prot', 'rt']]
                        # I do not need for this data
                        #add_pep_frame['charge']=add_pep_frame['charge'].astype(int)
                        #add_pep_frame['code_unique'] = add_pep_frame['peptide'] + '_' + add_pep_frame['prot'] + '_' +  add_pep_frame['mass'].astype(str) + '_' + add_pep_frame['charge'].astype( str)

                        ## withput ptotein in the key
                        add_pep_frame['code_unique'] = add_pep_frame['peptide'] + '_' + add_pep_frame['mass'].astype(str) + '_' + add_pep_frame['charge'].astype( str)

                        #add_pep_frame = add_pep_frame.groupby('code_unique', as_index=False)['peptide', 'mass', 'charge', 'prot', 'rt'].aggregate(max)
                        # 'mz'
                        add_pep_frame = add_pep_frame[['code_unique','peptide', 'mass',  'charge', 'prot', 'rt']]
                        list_name = add_pep_frame.columns.tolist()
                        list_name = [w.replace('rt', 'rt_' + str(c_rt)) for w in list_name]
                        add_pep_frame.columns = list_name
                        pre_pep_save.append(add_pep_frame)

                        c_rt += 1
            # print 'input columns',pre_pep_save[0].columns

            if n_replicates == 2:
                test = pre_pep_save[0]
            else:
                #'mz',
                test = reduce(
                    lambda left, right: pd.merge(left, right, on=['code_unique','peptide', 'mass',  'charge', 'prot'], how='inner'),
                    pre_pep_save)

            if test.shape[0]>1:
                print 'nooo'
                print test


            #test['time_ev']= test.ix[:, 5: (5 + (n_replicates - 1))].apply(lambda x : mass_assignment_consonat_bf(log_mbr, x, model_save,
            #                         model_err, args.w_comb,interval, 0.5,(l_int /2) ),axis=1)


            test['time_base'] = test.ix[:, 5: (5 + (n_replicates - 1))].apply(
                 lambda x: prediction_model( x,model,log_mbr),axis=1)
            # GP computation
            #test['time_base'] = test.ix[:, 5: (5 + (n_replicates - 1))].apply(
            #    lambda x: combine_model_GP(x, model_save,model_err, args.w_comb ,log_mbr) ,axis=1)

            test['matched'] = 1
            test['interval_number']= list_inter


            # if test[test['time_pred'] <= 0].shape[0] >= 1  :
            #	log_mbr.info(' -- Predicted negative RT : those peptide will be deleted')
            #	test= test[test['time_pred'] > 0]

            #list_name = test.columns.tolist()
            #list_name = [w.replace('time_ev', 'rt_ex') for w in list_name]
            #list_name = [w.replace('time_base', 'rt_base') for w in list_name]


            # test= test[['peptide','mass','mz','charge','prot','rt']]
            #print 'original feature ', c_data[fix_rep][c_data[fix_rep].code_unique == pep_out][['prot','rt'] ].shape
            test = test.merge(c_data[fix_rep][c_data[fix_rep].code_share == pep_out][['prot','rt','code_share'] ],on='prot',how='inner')
            out_df= pd.concat([out_df,test], join='outer', axis=0)
    ## print the entire file
    ## the file contains only the shared peptide LOO founded
    out_df.to_csv(path_or_buf= output_dir + '/' + str(os.path.split(exp_set[jj])[1].split('.')[0]) + '_match.txt',sep='\t',index=False)
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