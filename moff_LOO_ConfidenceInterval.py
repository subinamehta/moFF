import ConfigParser
import argparse
import ast
import bisect
import copy
import itertools
import logging
import os
import random
import re
from itertools import chain, combinations

import GPy
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import silhouette_score
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
            print 'current',ii,bba_set[ii].core()
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

    log.info( 'All intervall  %r',intervall[:,pos_inex_union])
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
                pred, var = model[ii].predict(x[ii].reshape(1, 1))
                print pred - np.sqrt(var)[0]
                # print ' %i Input Rt  %4.4f  Predicted: %4.4f Var %4.4f Interval at 95 %4.4 %4.4f ' % (ii, x[ii] ,float(pred), float(var), float(pred - (2 * np.sqrt(var))) ,float(pred + (2 * np.sqrt(var))) )
                log.info( ' %i Input Rt  %4.4f  Predicted: %4.4f Var %4.4f  Interval at 95 %4.4f <--> %4.4f  ' % (
                ii, x[ii], float(pred), float(var), float(pred - (2 * np.sqrt(var))), float(pred + (2 * np.sqrt(var)))) )
                iv_l=float(pred - (2 * np.sqrt(var)))
                iv_u=float(pred + (2 * np.sqrt(var)))
                log.info( 'Error mean.abs %4.4f' % float(err[ii]))
                pos = bisect.bisect(intervall[1, :].tolist(), pred)
                pos_l = bisect.bisect(intervall[1, :].tolist(),iv_l )
                pos_u = bisect.bisect(intervall[1, :].tolist(), iv_u)
                log.info('interval %i  #index highest %i', ii, pos)
                val = pred
                pos_index = np.arange(pos_l, pos_u+1)
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
                dist_norm= dist_norm[:]/ dist_norm.sum()
                #print dist_norm
                ## assignment

                print dist_norm[pos_index]
                print dist_norm[pos_index].sum()
                print  dist_norm.max(),dist_norm.argmax()

                ''''''
                # assigment 1
                m1 = MassFunction()

                m1[[str(dist_norm.argmax())]] =   dist_norm.max()
                left_belief = 1 -  dist_norm.max()

                m1[[str(a) for a in pos_index.tolist()]] = left_belief
                print m1
                print ('---    ----')
                ''''''
                bba_input.append(m1)

        print len(bba_input)
        for jj in range(len(bba_input)):
            log.info('Exp %i : %r', jj, bba_input[jj])

            # log.info('union_focal element %r',  focal_set_union(bba_input) )
        if focal_set_union(bba_input):
            output = conj_combination(bba_input, log, intervall, pos_inex_union, r)
        else:
            app, ii_index = focal_set_get_union(bba_input)
            # print ii_index
            if len(ii_index) >= 2:
                # uso la ConJ rule if two or more have  same common intervall
                log.info('Subset of expert combined with Conj Rule %r', ii_index)
                bba_input = [bba_input[i] for i in ii_index]
                output = conj_combination(bba_input, log, intervall, pos_inex_union, r)
                # else:
                #   output = disj_combination (bba_input, log, intervall, out_map_set,pos_inex_union)

        log.info('----- ----- -----')

            #print output
            # " output basic check control
        if output > 0:
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
        pred , var= model[ii].predict(x[ii].reshape(1,1))
        print pred - np.sqrt(var)[0]
        #print ' %i Input Rt  %4.4f  Predicted: %4.4f Var %4.4f Interval at 95 %4.4 %4.4f ' % (ii, x[ii] ,float(pred), float(var), float(pred - (2 * np.sqrt(var))) ,float(pred + (2 * np.sqrt(var))) )
        print ' %i Input Rt  %4.4f  Predicted: %4.4f Var %4.4f  Interval at 95 %4.4f <--> %4.4f  ' % (ii,x[ii],float(pred), float(var), float(pred - (2 * np.sqrt(var))) ,float(pred + (2 * np.sqrt(var))) )
        print 'Error mean.abs %4.4f' % float(err[ii])
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
def combine_model(x, model, err, weight_flag,log):
    # x = x.values
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

    clf = linear_model.RidgeCV(alphas=np.power(2, np.linspace(-30, 30)), scoring='mean_absolute_error', cv =None)
    clf.fit(data_B, data_A)
    #print ( ' alpha of the CV ridge regression model %4.4f',clf.alpha_)
    clf_final = linear_model.Ridge(alpha=  clf.alpha_, solver='svd',random_state=1, fit_intercept=False)
    clf_final.fit(data_B, data_A)



    return clf_final, clf_final.predict(data_B)

def train_models_3(data_A,data_B):
    ''' to get the confidence interval!
    bound_l  =  ym_train_predicted - 1.9600 * np.sqrt(y_var )
    bound_u =  ym_train_predicted + 1.9600 * np.sqrt(y_var)

    bound_l_80  =  ym_train_predicted - 1.2800 * np.sqrt(y_var )
    bound_u_80 =  ym_train_predicted + 1.2800 * np.sqrt(y_var)
    '''
    ## random sampling
    size_train= int (data_A.shape[0] * 0.10)
    print size_train, 'over',  data_A.shape[0]
    rows = random.sample(range(data_A.shape[0]),size_train)
    data_A= data_A[rows,:]
    data_B= data_B[rows,:]

     ## GP by pyGP
    #kern = GPy.kern.Linear(1)
    kern = GPy.kern.RBF(input_dim=1, lengthscale=1.0, variance=1.0)
    m = GPy.models.GPRegression(data_B, data_A,kern)
    #gamma_prior = GPy.priors.Gamma.from_EV(3, 2)
    #m.linear.variances.set_prior(gamma_prior)
    #m.Gaussian_noise.variance.constrain_bounded(0.5, 10)
    m.optimize()
    ym_train_predicted, y_var = m.predict(data_B)


    return m, ym_train_predicted,rows


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
            #" comparison diffrent methods lin Ridge GP
            output_dir ="D:\\bench_mark_dataset\\Com_Methods_weight"
            ## BF consonat  from GP
            output_dir ="D:\\bench_mark_dataset\\BF_consonant_GP"

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
        data_moff['mass'] = data_moff['mass'].map('{:.4f}'.format)
        data_moff['prot']=  data_moff['prot'].apply(lambda x:  x.split('|')[1] )
        data_moff['charge'] = data_moff['charge'].astype(int)
        data_moff['code_unique'] = data_moff['peptide'].astype(str) + '_' + data_moff['mass'].astype(str) + '_' + data_moff['charge'].astype(str)
        data_moff['code_share'] = data_moff['peptide'].astype(str) + '_' + data_moff['mass'].astype(str) + '_' + data_moff['charge'].astype(str) + '_' + data_moff['prot'].astype(str)

        # sort in pandas 1.7.1
        data_moff = data_moff.sort(columns='rt')
        if nn ==0:
            #intersect_share =  data_moff[(data_moff['prot']=='16128008') | ( data_moff['prot']=='170082857' ) ]['code_share'].unique()
            intersect_share =  data_moff[(data_moff['prot']=='16128008')  ]['code_share'].unique()
        else:
            #intersect_share = np.intersect1d( data_moff[(data_moff['prot']=='16128008') | ( data_moff['prot']=='170082857' )]['code_share'],intersect_share)
            intersect_share = np.intersect1d( data_moff[(data_moff['prot']=='16128008')]['code_share'],intersect_share)
        exp_t.append(data_moff)
        exp_out.append(data_moff)
        # Discretization of the RT space: get the max and the min valued
        if data_moff['rt'].min() <= min_RT:
            min_RT = data_moff['rt'].min()
        if data_moff['rt'].max() >= max_RT:
            max_RT = data_moff['rt'].max()
        #print data_moff['rt'].min(), data_moff['rt'].max()


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
    w_mbr = logging.FileHandler(args.loc_in + '/' + args.log_label + '_' + 'mbr_.log', mode='w')
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
    for fix_rep in [9]:
        out_df =pd.DataFrame(columns=['prot', 'peptide', 'charge', 'mz', 'rt_0', 'rt_1', 'rt_2',
       'rt_3', 'rt_4', 'rt_5', 'rt_6', 'rt_7', 'rt_8', 'rt_9',
       'time_base', 'matched', 'rt'])
        print 'MATCHING between RUN for  ', exp_set[fix_rep]
    # workarouand
    #for list_inter in [250]:
        for pep_out in intersect_share :
            ## binning  of the RT space in
            list_inter = 250
            interval,l_int = create_belief_RT_interval( max_RT,min_RT, list_inter )
            print 'radius', (l_int /2), '#interval', list_inter
            ## fix_rep input della procedura
            exp_t[fix_rep ]= c_data[fix_rep]
            #exp_t= c_data
            exp_t[fix_rep] = exp_t[fix_rep][exp_t[fix_rep].code_share != pep_out]
            #print 'after', exp_t[fix_rep].shape
            ##  palce here the inizialization
             # list of the model saved
            model_save = []
            # list of the error in min/or sec
            model_err = []
            # list of the status of the model -1 means model not available for low points in the training set
            model_status = []
            # add matched columns
            list_name.append('matched')
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
                        #common.to_csv('../bench_mark_dataset/export_testSingle_9vsOther_1.txt',sep='\t',index=False)

                        if common.shape[0] <= 50:
                            # print common.shape
                            model_status.append(-1)
                            continue
                        # filtering outlier option
                        else:
                            if int(args.out_flag) == 1:
                                filt_x, filt_y, pos_out = MD_removeOutliers(common['rt_y'].values, common['rt_x'].values,
                                                                            args.w_filt)
                                data_B = filt_x
                                data_A = filt_y
                                data_B = np.reshape(data_B, [filt_x.shape[0], 1])
                                data_A = np.reshape(data_A, [filt_y.shape[0], 1])
                                log_mbr.info('Outlier founded %i  w.r.t %i', pos_out.shape[0], common['rt_y'].shape[0])
                            else:
                                data_B = common['rt_y'].values
                                data_A = common['rt_x'].values
                                #if i[1]==1:
                                #    common.to_csv('../bench_mark_dataset/test_9vs1.txt',sep='\t',index=False)
                                data_B = np.reshape(data_B, [common.shape[0], 1])
                                data_A = np.reshape(data_A, [common.shape[0], 1])
                            log_mbr.info(' Size training shared peptide , %i %i ', data_A.shape[0], data_B.shape[0])


                            # version with confidence interval
                            #model , predicted_train=  train_models_1(data_A,data_B)
                            #model , predicted_train = train_models_1_robust(data_A,data_B)
                            #model , predicted_train = train_models_1_robustLinearModel(data_A,data_B)

                            # version with Ridge regression
                            #model , predicted_train=  train_models_2_ridge(data_A,data_B)
                            #print i[1],data_A.shape, mean_absolute_error(data_A,predicted_train )


                            # version with linear regression
                            #model , predicted_train = train_models_2_linear(data_A,data_B)

                            # GP process test
                            #print 'GP test'

                            model , predicted_train, index_train =  train_models_3(data_A,data_B)
                            #print i[1],data_A.shape, mean_absolute_error(data_A[index_train,:],predicted_train )
                            ##only for model 3
                            model_err.append( mean_absolute_error(data_A[index_train,:],predicted_train ) )
                            log_mbr.info(' Mean absolute error training : %4.4f sec',
                                         mean_absolute_error(data_A[index_train,:],predicted_train ))
                            ## save the model

                            model_save.append(model)

                            ## un comment for the other model
                            #model_err.append(mean_absolute_error(data_A,predicted_train ))
                            #log_mbr.info(' Mean absolute error training : %4.4f sec',            mean_absolute_error(data_A, predicted_train))
                            model_status.append(1)
            if np.where(np.array(model_status) == -1)[0].shape[0] >= (len(aa) / 2):
                log_mbr.warning('MBR aborted :  mbr cannot be run, not enough shared peptide among the replicates ')
                exit('ERROR : mbr cannot be run, not enough shared pepetide among the replicates')
            log_mbr.info('Combination of the  model  --------')
            log_mbr.info('Weighted combination  %s : ', 'Weighted' if int(args.w_comb) == 1 else 'Unweighted')
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
           # print 'input columns',pre_pep_save[0].shape


            if n_replicates == 2:
                test = pre_pep_save[0]
            else:
                #'mz',
                test = reduce(
                    lambda left, right: pd.merge(left, right, on=['code_unique','peptide', 'mass',  'charge', 'prot'], how='inner'),
                    pre_pep_save)

            if test.shape[0]>1:
                print 'nooo'
            #print 'size',test.shape
            #test =test.groupby(['prot','peptide'],as_index=False)['charge','mass','rt_0','rt_1','rt_2','rt_3','rt_4','rt_5','rt_6','rt_7','rt_8','rt_9'].mean()
            #test['time_ev'] = test.ix[:, 5: (5 + (n_replicates - 1))].apply(
                #lambda x: mass_assignment(log_mbr, x, model_save[(jj * (n_replicates - 1)):((jj + 1) * (n_replicates - 1))],
                #                        model_err[(jj * (n_replicates - 1)):((jj + 1) * (n_replicates - 1))], args.w_comb,interval, 1,(l_int /2) ) ,axis=1)

           # test['time_ev']= test.ix[:, 5: (5 + (n_replicates - 1))].apply(lambda x :  mass_assignment(log_mbr, x, model_save,
           #                             model_err, args.w_comb,interval, 1,(l_int /2) ),axis=1)

            #test['time_ev']= test.ix[:, 5: (5 + (n_replicates - 1))].apply(lambda x : mass_assignment_consonat_bf(log_mbr, x, model_save,
            #                         model_err, args.w_comb,interval, 0.5,(l_int /2) ),axis=1)
            #test['time_base'] = test.ix[:, 5: (5 + (n_replicates - 1))].apply(
            #    lambda x: combine_model(x, model_save,model_err, args.w_comb ,log_mbr) ,axis=1)

            # GP computation
            test['time_base'] = test.ix[:, 5: (5 + (n_replicates - 1))].apply(  lambda x: combine_model_GP(x, model_save,model_err, args.w_comb ,log_mbr) ,axis=1)
            test['time_ev'] = test.ix[:, 5: (5 + (n_replicates - 1))].apply(lambda x:combine_model_GP_consonant_bf(x, model_save, model_err, args.w_comb, log_mbr, interval, 0.5,(l_int /2)) ,axis=1)


            test['matched'] = 1
            #test['interval_number']= list_inter
            #test['radious']=(l_int /2)

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
        print 'final dataframe',out_df.shape
        out_df.to_csv(path_or_buf= output_dir + '/' + str(os.path.split(exp_set[jj])[1].split('.')[0]) + '_match.txt',sep='\t',index=False)
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