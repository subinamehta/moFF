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

##  crete ate the power set
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
## create dictionary for possible set name .
def create_dict_frame(pos_index):
    let= np.array( list(map(chr, range(97, 97 +len(pos_index)))) )
    map_inter = {}
    map_set2_inter = {}
    for i in powerset( list(range(0,len(pos_index)) )):
        if ( len(i) == 1 ):
            map_inter[pos_index[i]] = let[i]
            map_set2_inter [ let[i] ] = pos_index[i]
        else:
            map_inter[  ''.join(str(pos_index[list(i)].tolist()))  ]= ''.join(let[list(i)].tolist())
            map_set2_inter [ ''.join(let[list(i)].tolist())  ] = ''.join(str(pos_index[list(i)].tolist()))
    return map_inter,map_set2_inter

## methods for n expert in input
## Union of the focal set
def focal_set_union (bba_set):
    app=  bba_set[0].core()
    for ii in range(1,len(bba_set)):
            app = app.intersection(bba_set[ii].core())
    return app

## methods for find a subsett of element in  the same set
def focal_set_get_union (bba_set):
    app=  bba_set[0].core()
    index_exp= [0]
    #print 'init',app
    for ii in range(1,len(bba_set)):
            #print 'current',ii,bba_set[ii].core()
            if   ( app.intersection(bba_set[ii].core()) )  :
                app = app.intersection(bba_set[ii].core())
                index_exp.append(ii)
            #print 'current output ', app
    return app, index_exp


def conj_bba_set(bba_set):
    app=  bba_set[0]
    for ii in range(1,len(bba_set)):
            app = app.combine_conjunctive(bba_set[ii])
    return app


def disj_bba_set(bba_set):
    app=  bba_set[0]
    for ii in range(1,len(bba_set)):
            app = app.combine_disjunctive(bba_set[ii])
    return app

## union of the frame
def define_frame(x, model ,intervall,delta):
     #//----   union of the set

    if  x[np.isnan(x)].shape[0] == 0:
        # init  no one nan values
        pos= bisect.bisect(intervall[1,:].tolist(),model[0].predict(x[0])[0]) - 1
        if pos < delta:
               init_union = np.arange(pos-pos,pos+pos)
        else:
               init_union = np.arange(pos-delta,pos+delta)

        offset = 1
    else:
        if np.isnan(x[0]) :
            first_index = np.where(~np.isnan(x))[0][0]
            ## caso fisrt element is nana
            pos = bisect.bisect(intervall[1, :].tolist(), model[first_index].predict(x[first_index])[0]) - 1

            init_union = np.arange(pos-delta,pos+delta)
            offset = np.where(~np.isnan(x))[0][1]
        else:
            ## caso first element !=0
            pos= bisect.bisect(intervall[1,:].tolist(),model[0].predict(x[0])[0]) - 1

            init_union = np.arange(pos-delta,pos+delta)
            offset = np.where(~np.isnan(x))[0][1]
    for ii in range(offset, len(x)):
        if  ~  np.isnan(x[ii]):
            pos= bisect.bisect(intervall[1,:].tolist(),model[ii].predict(x[ii])[0]) - 1
            if pos < delta:
                pos_app = np.arange(pos-pos,pos+pos)
            else:

                pos_app = np.arange(pos-delta,pos+delta)
            # check if pos could flow out of mmax number of interval
            pos_app= pos_app[pos_app < intervall.shape[1]]
            init_union = np.union1d(pos_app,init_union  )
         #print 'frame', ii, pos_app
    return init_union


def define_frame_GP(x, model, intervall, delta):
    # //----   union of the set

    if x[np.isnan(x)].shape[0] == 0:
        # init  no one nan values
        pos = bisect.bisect(intervall[1, :].tolist(), model[0].predict(x[0].reshape(1,1))[0]) - 1
        if pos < delta:
            init_union = np.arange(pos - pos, pos + pos)
        else:
            init_union = np.arange(pos - delta, pos + delta)

        offset = 1
    else:
        if np.isnan(x[0]):
            first_index = np.where(~np.isnan(x))[0][0]
            ## caso fisrt element is nana
            pos = bisect.bisect(intervall[1, :].tolist(), model[first_index].predict(x[first_index].reshape(1,1))[0]) - 1
            init_union = np.arange(pos - delta, pos + delta)
            offset = np.where(~np.isnan(x))[0][1]
        else:
            ## caso first element !=0
            pos = bisect.bisect(intervall[1, :].tolist(), model[0].predict(x[0].reshape(1,1))[0]) - 1

            init_union = np.arange(pos - delta, pos + delta)
            offset = np.where(~np.isnan(x))[0][1]
    for ii in range(offset, len(x)):
        if ~  np.isnan(x[ii]):
            pos = bisect.bisect(intervall[1, :].tolist(), model[ii].predict(x[ii].reshape(1,1))[0]) - 1
            if pos < delta:
                pos_app = np.arange(pos - pos, pos + pos)
            else:

                pos_app = np.arange(pos - delta, pos + delta)
            # check if pos could flow out of mmax number of interval
            pos_app = pos_app[pos_app < intervall.shape[1]]
            init_union = np.union1d(pos_app, init_union)
            # print 'frame', ii, pos_app
    return init_union




def conj_combination (bba, log, intervall,pos_inex_union,radius):
    #print bba
    m_comb=  conj_bba_set(bba)
    log.info('Dempster combination rule  = %r', m_comb )
    log.info( 'conflict %r',  m_comb.local_conflict() )
    log.info('pig_trans %r', m_comb.pignistic())
    max_set=0
    set_res= ''
    for s in  m_comb.pignistic().keys():
        #log.info('Pig.prob %.4f',  m_comb.pignistic()[s])
        if m_comb.pignistic()[s] > max_set:
            max_set= m_comb.pignistic()[s]
            set_res = s
            #print set_res
    log.info( 'choosen interval %s', list(set_res))
    log.info( 'max prob %.4f', max_set)
    # integer  as hypothesis in the frame
    output =  intervall[1,int(list(set_res)[0])]

    #log.info( 'All intervall  %r',intervall[:,pos_inex_union])
    log.info( 'final value %.4f | %.4f  %.4f ',output, output - radius, output + radius)
    return output


def disj_combination (bba, log, intervall, out_map_set,pos_inex_union):
    m_comb= disj_bba_set(bba)
    log.info(' Disjuntive combination rule for m_1 and m_2 = %r', m_comb )
    log.info( 'conflict %r' ,  m_comb.local_conflict() )
    log.info( 'pig_trans %r', m_comb.pignistic())
    max_set=  max(m_comb.pignistic().values())
    set_res= []
    for s in  m_comb.pignistic().keys():
        #log.info('Pig.prob %.4f ,  %r',  m_comb.pignistic()[s],s )
        if m_comb.pignistic()[s] == max_set:
            max_set= m_comb.pignistic()[s]
            set_res.append(s)
    #log.info( 'choosen interval %s', list(set_res))
    #log.info( 'max prob %.4f', max_set)
    ss =0
    for  gg in set_res:
            ss += intervall[1,int(out_map_set[list(gg)[0]])]
    output = ss/ len(set_res)
    #output =  intervall[1,int(out_map_set[list(set_res)[0]])]
    log.info( 'All intervall  %r',intervall[:,pos_inex_union])
    log.info( 'final value %.4f',output)
    return output

## use a kmean to assign the belief from  the distance distribution
def mass_assignment_kmeans(log,x, model, err, weight_flag,intervall,k,r):
    x = x.values
    if x[~ np.isnan(x)].shape[0] == 1:
        # run normal combination.
    ##    # it does not make sense to combine with just one expert
        return combine_model(x, model, err, weight_flag)
    x_out = [  model[ii].predict(x[ii])[0][0] for ii in range(len(x)) ]
    err = np.array(err)
    bba_input= []
    log.info( '-----  ----- ----')
    log.info('input values : %r', x)
    log.info('predict values : %r', x_out)
    log.info('error_model : %r ', np.array(err) )
    norm_err =  ( np.array(err)[np.where(~np.isnan(x))]  )  /(np.sum(np.array(err)[np.where(~np.isnan(x))]))
    log.info('norm_error : %r ', np.array(norm_err) )
    log.info( 'radius in min %.4f # interval %r  ', r,intervall.shape)
    pos_inex_union= define_frame(x, model, intervall )
    log.info( 'frame final %r', pos_inex_union)
    out_map,out_map_set = create_dict_frame(pos_inex_union )
    #//---- end frame computation
    for ii in range(0, len(x)):
        if ~  np.isnan(x[ii]):
            pos= bisect.bisect(intervall[1,:].tolist(),model[ii].predict(x[ii])[0][0]) - 1
            log.info('interval %i %i', ii, pos)
            val = model[ii].predict(x[ii])[0][0]
            all_dist=[]
            for aa in pos_inex_union:
                all_dist.append(np.exp( - k *  abs( val - intervall[1,aa])) )
            log.info('Distance for all')
            log.info( ' %r', all_dist)
            all_dist= np.array(all_dist,dtype=np.float_)
            all_dist_norm = all_dist / all_dist.sum()
            all_dist_norm =all_dist_norm.reshape(all_dist_norm.shape[0],1)
            log.info('Distance norm for all')
            log.info( ' %r', all_dist_norm)
            max_sil= 0
            k_sel=0
            save_kmean = np.zeros(all_dist_norm.shape[0])
            for k in [2,3,4]:
                outkmean =KMeans(n_clusters=k, random_state=3).fit_predict(all_dist_norm)
                #print outkmean
                silhouette_avg = silhouette_score(all_dist_norm, outkmean)
                #print silhouette_avg
                if  silhouette_avg > max_sil:
                    max_sil = silhouette_avg
                    k_sel = k
                    save_kmean= outkmean
            log.info( 'k choosen: %r  Avg_sil_coef : %4f.f', k_sel, max_sil)

            m1 = MassFunction()
            for ss in range(k_sel):
                if np.where(save_kmean == ss)[0].shape[0] > 1:
                    m1[out_map[''.join(str(list(pos_inex_union[np.where(save_kmean ==ss)])))]]= all_dist_norm[np.where(save_kmean == ss),0].sum()
                else:
                    m1[out_map[pos_inex_union[np.where(save_kmean ==ss)][0]]]= all_dist_norm[np.where(save_kmean == ss),0]

            #print m1
            #for ff in m1.focal():
            #    bba_input.append(m1)


    for jj in range(len(bba_input)):
        log.info( 'Exp %i : %r  %r',jj, bba_input[jj],bba_input[jj].local_conflict() )

    #log.info('union_focal element %r',  focal_set_union(bba_input) )
    if focal_set_union(bba_input)  :
        output = conj_combination (bba_input,log, intervall, out_map_set,pos_inex_union )
    else:
        app,ii_index  = focal_set_get_union(bba_input)
        #print ii_index
        if len(ii_index) >= 2:
             # uso la ConJ rule if two or more have  same common intervall
             log.info('Subset of expert combined with Conj Rule %r', ii_index )
             bba_input = [bba_input[i] for i in ii_index]
             output = conj_combination (bba_input,log, intervall, out_map_set,pos_inex_union )
    log.info( '----- ----- -----')

    # " output basic check control
    if  output  > 0:
        return output
    else:
        # output not weight
        return -1

def combine_model_GP_consonant_bf(x, model, err, weight_flag, log,intervall,k,r):
        bba_input = []
        debug__mode= True

        log.info ('radius in min %.4f # interval %r  ', r, intervall.shape)
        pos_inex_union = define_frame_GP(x, model, intervall, 9)
        #print pos_inex_union
        log.info('frame final Theta %r %r', pos_inex_union, pos_inex_union.shape)
        for ii in range(0, len(x)):
            if ~  np.isnan(x[ii]):
                pred, var = model[ii].predict(x[ii].reshape(1, 1),include_likelihood=False)

                #iv_l = float(pred - (2 * np.sqrt(var)))
                #iv_u = float(pred + (2 * np.sqrt(var)))
                ress = model[ii].predict_quantiles( x[ii].reshape(1,1) )
                iv_l = ress[0]
                iv_u = ress[1]

                #print pred - np.sqrt(var)[0]
                log.info( '%i Input Rt  %4.4f  Predicted: %4.4f Var %4.4f  Interval at 95 %4.4f %4.4f ' % (ii, x[ii] ,float(pred), float(var), float(iv_l) ,float(iv_u )))

                log.info( 'Error mean.abs %4.4f' % float(err[ii]))
                #pos = bisect.bisect(intervall[1, :].tolist(), pred) -1
                pos = index(intervall[1, :], find_ge(intervall[1, :], pred,intervall))

                ## -1 because all(val < x for val in a[lo:i]) see bisect doc.
                pos_l = index( intervall[1,:], find_ge(intervall[1,:], iv_l,intervall))

                pos_u = index( intervall[1,:], find_ge(intervall[1,:], iv_u,intervall))

                log.info('interval %i  #index highest %i', ii, pos)
                val = pred
                pos_index = np.arange(pos_l, pos_u+1)
                '''
                if pos_index.shape[0]==1:
                    print pos_index
                    print pos_l,pos,pos_u
                    print iv_l,pred,iv_u
                    print abs(pos_u-pos_l),r
                    print intervall[:,pos -1 :pos+1]
                    exit('-- DEBUG ')
                '''
                log.info('interval %i  %r', ii, pos_index)
                dist_min = 0
                if debug__mode:
                    all_dist = []
                    for aa in pos_inex_union:
                        cur_val = np.exp(- k * abs(pred - intervall[1, aa]))
                        log.info('#interval %r %4.4f ' , aa,float(cur_val[0]))

                        # cur_val =  abs( val - intervall[1,aa])
                        if cur_val > dist_min:
                            dist_min = cur_val
                            pos = aa

                        all_dist.append(cur_val[0])

                dist_norm = pd.Series(all_dist, index=pos_inex_union)

                # normalization
                #dist_norm= dist_norm[:]/ dist_norm.sum()
                #print dist_norm
                ## assignment

                #print dist_norm[pos_index]
                #print dist_norm[pos_index].sum()
                #print  dist_norm.max(),dist_norm.argmax()

                ''''''
                # assigment 1
                m1 = MassFunction()

                m1[[str(dist_norm.argmax())]] =   dist_norm.max()
                left_belief = 1 -  dist_norm.max()

                m1[[str(a) for a in pos_index.tolist()]] = left_belief
                #print m1
                #print ('---    ----')
                ''''''
                bba_input.append(m1)
        combination=False
        #print len(bba_input)
        for jj in range(len(bba_input)):
            log.info('Exp %i : %r', jj, bba_input[jj])
            # log.info('union_focal element %r',  focal_set_union(bba_input) )
        if focal_set_union(bba_input):
            output = conj_combination(bba_input, log, intervall, pos_inex_union, r)
            combination=True
        else:
            app, ii_index = focal_set_get_union(bba_input)
            # print ii_index
            if len(ii_index) >= 2:
                # uso la ConJ rule if two or more have  same common intervall
                log.info('Subset of expert combined with Conj Rule %r', ii_index)
                bba_input = [bba_input[i] for i in ii_index]
                output = conj_combination(bba_input, log, intervall, pos_inex_union, r)
                combination=True
                # else:
                #   output = disj_combination (bba_input, log, intervall, out_map_set,pos_inex_union)

        log.info('----- ----- -----')

            #print output
            # " output basic check control
        if combination :
            return output
        else:
            # output not weight
            return -1


# combination of RT for
def mass_assignment_consonat_bf(log,x, model, err, weight_flag,intervall,k,r):
    x = x.values
    if x[~ np.isnan(x)].shape[0] == 1:
        # run normal combination.
    ##    # it does not make sense to combine with just one expert
        return combine_model(x, model, err, weight_flag)
    x_out = [  model[ii].predict(x[ii])[0] for ii in range(len(x)) ]
    err = np.array(err)
    bba_input= []
    log.info( '-----  ----- ----')
    log.info('input values : %r', x)
    log.info('predict values : %r', x_out)
    log.info('error_model : %r ', np.array(err) )
    norm_err =  ( np.array(err)[np.where(~np.isnan(x))]  )  /(np.sum(np.array(err)[np.where(~np.isnan(x))]))
    log.info('norm_error : %r ', np.array(norm_err) )
    log.info( 'radius in min %.4f # interval %r  ', r,intervall.shape)
    pos_inex_union= define_frame(x, model, intervall,7 )
    print pos_inex_union
    log.info( 'frame final %r %r', pos_inex_union,pos_inex_union.shape)
    ## disable fix creation of the frame  A lot of computatin saved
    #out_map,out_map_set = create_dict_frame(pos_inex_union )

    #print out_map

    #//---- end frame computation
    for ii in range(0, len(x)):
        debug__mode= True
        if ~  np.isnan(x[ii]):
            pos= bisect.bisect(intervall[1,:].tolist(),model[ii].predict(x[ii])[0])
            prstd, iv_l, iv_u = wls_prediction_std(model[ii], exog=x[ii],alpha=0.20)
            prstd_2, iv_l_2, iv_u_2 = wls_prediction_std(model[ii], exog=x[ii],alpha=0.05)

            pos_l= bisect.bisect(intervall[1,:].tolist(),iv_l)
            pos_u = bisect.bisect(intervall[1,:].tolist(),iv_u)
            pos_l_2= bisect.bisect(intervall[1,:].tolist(),iv_l_2)
            pos_u_2 = bisect.bisect(intervall[1,:].tolist(),iv_u_2)
            log.info('interval %i  #index highest %i', ii, pos)
            val = model[ii].predict(x[ii])[0]
            pos_index = np.arange(pos_l,pos_u)
            pos_index_2 = np.arange(pos_l_2,pos_u_2)
            dist_min= 0
            if debug__mode:
                all_dist=[]
                for aa in pos_inex_union:
                    cur_val = np.exp( - k *  abs( val - intervall[1,aa]))
                    #cur_val =  abs( val - intervall[1,aa])

                    if cur_val  > dist_min:
                        dist_min = cur_val
                        pos= aa

                    all_dist.append(cur_val )
                log.info('Distance for all')
                all_dist= all_dist/ sum(all_dist)
                log.info( ' %r', all_dist)
            dist_norm = pd.Series( all_dist,index=pos_inex_union  )
            '''
            print dist_norm
            print 'sum all', dist_norm.sum()

            print 'max val', dist_norm[dist_norm == dist_norm.max()]
            print 'interval', dist_norm[pos_index].sum(), dist_norm[pos_index].product() #, dist_norm[pos_index].sum()- dist_norm[dist_norm == dist_norm.max()]
            print 'interval', dist_norm[pos_index_2].sum(),dist_norm[pos_index].product() #, dist_norm[pos_index_2].sum()- dist_norm[dist_norm == dist_norm.max()]
            print dist_norm[pos_index_2].sum()/ pos_index_2.shape[0]
            print pos_index
            print pos_index_2
            print dist_norm[dist_norm == dist_norm.max()].values
            print 1- dist_norm[dist_norm == dist_norm.max()].values, ( 1- dist_norm[dist_norm == dist_norm.max()].values) *(float(pos_index.shape[0]) / float(pos_index_2.shape[0]) )
            print str(dist_norm.argmax())
            '''

            m1 = MassFunction()
            # mod 1
            #second idea  with two  in conf. interval at 80 an 90.

            #m1[ [str(dist_norm.argmax())]  ] = dist_norm[dist_norm == dist_norm.max()].values


            #left_belief= 1- dist_norm[dist_norm == dist_norm.max()].values
            #ratio= (float(pos_index.shape[0]) / float(pos_index_2.shape[0]) )

            #m1[[ str(a)  for a in pos_index.tolist()  ]] = left_belief *ratio
            #m1[[ str(a)  for a in pos_index_2.tolist()  ]] =  left_belief- (left_belief * ratio)


            # mod 2 more simple onlty one interval
            m1[ [str(dist_norm.argmax())]  ] = dist_norm[dist_norm == dist_norm.max()].values
            left_belief= 1- dist_norm[dist_norm == dist_norm.max()].values
            m1[[ str(a)  for a in pos_index.tolist()  ]] = left_belief


            log.info('---    ----')
            bba_input.append(m1)

    #log.info( 'combined_masses %r', bba_input )
    ## print stuff
    for jj in range(len(bba_input)):
        log.info( 'Exp %i : %r',jj, bba_input[jj] )

    #log.info('union_focal element %r',  focal_set_union(bba_input) )
    if focal_set_union(bba_input)  :
        output = conj_combination (bba_input,log, intervall, pos_inex_union ,r)
    else:
        app,ii_index  = focal_set_get_union(bba_input)
        #print ii_index
        if len(ii_index) >= 2:
             # uso la ConJ rule if two or more have  same common intervall
             log.info('Subset of expert combined with Conj Rule %r', ii_index )
             bba_input = [bba_input[i] for i in ii_index]
             output = conj_combination (bba_input,log, intervall, pos_inex_union ,r)
        #else:
         #   output = disj_combination (bba_input, log, intervall, out_map_set,pos_inex_union)




    log.info( '----- ----- -----')

    #print output
    # " output basic check control
    if  output  > 0:
        return output
    else:
        # output not weight
        return -1





def combine_model_GP(x, model, err, weight_flag,log):
    # x = x.values
    #tot_err =  1- ( (np.array(err)[np.where(~np.isnan(x))]) / np.max(np.array(err)[np.where(~np.isnan(x))]))
    tot_err = np.sum(np.array(err)[np.where(~np.isnan(x))])
    #print tot_err
    #print x
    app_sum = 0
    app_sum_2 = 0
    for ii in range(0, len(x)):
        pred , var= model[ii].predict(x[ii].reshape(1,1),include_likelihood=False)
        ress= model[ii].predict_quantiles(x[ii].reshape(1,1))

        #print ' %i Input Rt  %4.4f  Predicted: %4.4f Var %4.4f Interval at 95 %4.4f  <--> %4.4f ' % (ii, x[ii] ,float(pred), float(var), float(pred - (2 * np.sqrt(var))) ,float(pred + (2 * np.sqrt(var))) )
        print ' %i Input Rt  %4.4f  Predicted: %4.4f Var %4.4f  Interval at 95 %4.4f <--> %4.4f  ' % (ii,x[ii],float(pred), float(var), ress[0] ,ress[1] )
        print 'Training Error mean.abs %4.4f' % float(err[ii])
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


def prediction_one_model_local_gp(x, model,cluster,log):
   
    x = x.filter(regex=("rt_*"))
    x.fillna(x.mean(),inplace=True)

    x_point_t= np.reshape(x[0:10],(1,x[0:10].shape[0]))
    
    model_index = cluster.predict(x_point_t)
    
    
    mean, var = model[model_index].predict_y(  x_point_t) 

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
        pred ,var =  model.predict( x,include_likelihood=False)
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
    k = GPflow.kernels.RBF(input_dim=10)
    m = GPflow.gpr.GPR(data_B, data_A, kern=k)
    m.optimize()
    mean, var = m.predict_y(  data_B)

    return m # , ym_train_predicted

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


def	 create_belief_RT_interval (max_rt, min_rt,n_interval):
    # print max_rt, min_rt, float(max_rt-min_rt) / float(20)
	off_set = float(max_rt-min_rt) / float(n_interval)
	#print 'length_interval',off_set
	interval_mat  = np.zeros(shape=(3,n_interval))
	for i in range(0,n_interval):
		#print i , min_rt + (i * off_set )
		interval_mat[0,i] = min_rt + (i * off_set )
		interval_mat[2,i] = (min_rt + (i * off_set )) +  (( float(max_rt-min_rt) / float(n_interval)  ) -0.001)
		interval_mat[1,i] = (interval_mat[0,i] + interval_mat[2,i] ) /2

	print interval_mat[:,0], interval_mat[:,19]
	return interval_mat,off_set


def data_inputing(common):

    fill_value = pd.DataFrame({col: common.ix[:, :].mean(axis=1) for col in common.columns})
    # print  fill_value
    common.fillna(fill_value, inplace=True)


    return common

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
    #print intersect_share
    # most abundant protein :-> 16128008
    # other one : gi|170082857, gi|170083440, gi|16131810
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
        out_df =pd.DataFrame(columns=['prot', 'peptide', 'charge', 'mz', 'rt_0', 'rt_1', 'rt_2',
       'rt_3', 'rt_4', 'rt_5', 'rt_6', 'rt_7', 'rt_8', 'rt_9','time_base',  'rt'])


        print 'MATCHING between RUN for  ', exp_set[fix_rep]
    # workarouand
    #for list_inter in [250]:
        #['AHEILESR_953.493_2','AIDKPFLLPIEDVFSISGR_2116.1568_3','AIDMHISNLR_1168.6022_3','AGNGETILTSELYTSK_1683.8203_2']
        for pep_out  in intersect_share[0:500]:
            print '-- ', pep_out ,'-- '
            ## binning  of the RT space in
            list_inter = 130

            ## fix_rep input della procedura
            exp_t[fix_rep ]= c_data[fix_rep]
            #exp_t= c_data
            exp_t[fix_rep] = exp_t[fix_rep][exp_t[fix_rep].code_unique != pep_out]

            list_name.append('matched')
            for jj in [fix_rep]:
                first = True
                #temp_Data=[]
                temp_data_target=[]
                temp_data_input=[]
                for i in out:
                    if i[0] == fix_rep and i[1] != jj:
                        log_mbr.info('  Matching  %s peptide in   searching in %s ', exp_set[i[0]], exp_set[i[1]])
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

                        ## check the varience for each code unique and gilter only the highest 99
                        ##IMPORTant
                        flag_var_filt = False
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
                        #common = pd.merge(comA, comB, on=['code_unique'], how='inner')
                        temp_data_target.append(comA[['code_unique', 'rt']])
                        temp_data_input.append(comB[['code_unique', 'rt']])

            print len(temp_data_input)
            '''
            # taking the intersect for the trainning set no missing 
            tt = reduce(np.intersect1d, ([x.code_unique.values for x in temp_data_target]))
            df=pd.DataFrame(index= tt)
            for i in range(0,len(temp_data_input)):
                df[str(i)]= temp_data_input[i][ temp_data_input[i]['code_unique'].isin(tt)]['rt'].values
            X = df.ix[:,0:df.shape[0]].values
            Y = temp_data_target[i][ temp_data_target[i]['code_unique'].isin(tt)]['rt'].values
                
            '''
            ## union with missing values
            tt = reduce(np.union1d, ([x.code_unique.values for x in temp_data_target]))
            df=pd.DataFrame(index= tt)
            df2 =pd.DataFrame(index= tt)
            df2['Y'] = np.nan
            for i in range(0,len(temp_data_input)):
                df[str(i)]= np.nan
                df.ix[np.intersect1d(tt,temp_data_input[i]['code_unique']),i ] =temp_data_input[i][ temp_data_input[i]['code_unique'].isin(tt)]['rt'].values
                df2.ix[ np.intersect1d(tt,temp_data_target[i]['code_unique']),0 ] = temp_data_target[i][ temp_data_target[i]['code_unique'].isin(tt)]['rt'].values


            #option for data export
            #df['Y'] = df2['Y']
            #df.to_csv('../RT_data_benchmark/mono_model_test.data', sep='\t', index=False)
            #del df2
            ## filling Na value
            df= data_inputing(df)

            X = df.ix[:,0:(df.shape[0])].values
            Y = df2['Y'].values




            X = np.reshape(X, [X.shape[0], X.shape[1]])
            Y = np.reshape(Y, [Y.shape[0], 1])

            ## train model
            #model ,predicted_train  = train_models_2_ridge(Y,X)
            #selec_model=1
            #print model.intercept_, model.coef_
            #print X.shape, mean_absolute_error(Y,predicted_train )
            #'''
            # original model with GPy
            #model, predicted_train = train_models_3(Y, X)
            #GP flow
            #model,predicted_train = train_models_GP_flow(Y,X)
            # local GP version GPflow
            models, cluster_model = train_local_gp(Y,X,7)
            # not used in local GP
            #selec_model=2
            #print X.shape, mean_absolute_error(Y,predicted_train )
            #'''

            ## from here we have to deal with one one model



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
                        log_mbr.info('Matching peptides found in  %s ', exp_set[i[1]])
                        add_pep_frame = exp_t[i[1]][exp_t[i[1]]['code_unique']== pep_out  ].copy()

                        ## custom case
                        # 'mz'
                        add_pep_frame = add_pep_frame[['peptide',  'charge','mass_th','prot', 'rt']]
                        add_pep_frame = add_pep_frame.groupby(['peptide','charge','mass_th','prot'],as_index=False).mean()


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
            if test.shape[0]>1:
                print 'nooo'

            #test['time_base']= 100
           
            #test['time_base'] = test.ix[:,:].apply( lambda x: prediction_one_model(x, model,log_mbr,selec_model) ,axis=1)
            #  local GP
            test['time_base'] = test.ix[:,:].apply( lambda x: prediction_one_model_local_gp(x, models, cluster_model,log_mbr) ,axis=1)
            #test['rt_pred_width'] = test.ix[:, 4: (4 + (n_replicates - 1))].apply( lambda x: prediction_rt_unc_wind_one_model(x, model, log_mbr), axis=1)




            test['matched'] = 1


            # test= test[['peptide','mass','mz','charge','prot','rt']]
            #print 'original feature ', c_data[fix_rep][c_data[fix_rep].code_unique == pep_out][['prot','rt'] ].shape
            test = test.merge(c_data[fix_rep][c_data[fix_rep].code_unique == pep_out][['prot','rt','code_unique'] ],on=['code_unique','prot'],how='inner')
            out_df= pd.concat([out_df,test], join='outer', axis=0)
    ## print the entire file
    ## the file contains only the shared peptide LOO founded
        print 'final dataframe',out_df.shape
        out_df.to_csv(path_or_buf= output_dir + '/' + str(os.path.split(exp_set[jj])[1].split('.')[0]) + 'localGP_monoModel_match.txt',sep='\t',index=False)
        print 'rimetto i pep tolti per LO0'
        exp_t[fix_rep ]= c_data[fix_rep]
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