import numpy as np
import glob as glob
import pandas as pd
import os as os
import sys
import subprocess
import shlex 
import  logging
from StringIO import StringIO

### input###
## - MS2 ID file
## - tol
## - half rt time window in minute
###### output
##  list of intensities..+

file_name=str(sys.argv[1])
tol=str(sys.argv[2])
h_rt_w = float(sys.argv[3])
s_w= float(sys.argv[4])
loc_raw =str(sys.argv[5])
loc_output =str(sys.argv[6])

#print file_name

##output name
## just for debug one file at time
##print os.getcwd()

## ncomment these two lines, when you run from the split
#os.chdir(file_name.split('/')[0] + "/")
#file_name= file_name.split('/')[1]

first_file=False

temp_n =file_name.split('.')[0]
nn =temp_n.split('_')[0] + '_' +  temp_n.split('_')[1] + '_' +  temp_n.split('_')[2] + '_' + temp_n.split('_')[3] + '_' + temp_n.split('_')[4] + '_' + temp_n.split('_')[5] + '_' + temp_n.split('_')[6]  
#nn =  file_name.split('.')[0].split('_')[0]
count = file_name.split('.')[0].split('_')[8]


logging.basicConfig(filename=str(file_name.split('.')[0])+ '__moff.log',filemode='w',level=logging.INFO)






outputname=  loc_output  +  "temp"  + "_" + count  + "_result.txt"
if  int (  count )==0:
        print "First file"
       	first_file=True


#+ "_" + file_name.split('.')[0].split('_')[4]   

logging.basicConfig(filename=str(file_name.split('.')[0])+ '__moff.log',filemode='w',level=logging.INFO)

logging.info('moff Input file %s  XIC_tol %s XIC_win %4.4f moff_rtWin_peak %4.4f ',file_name,tol,h_rt_w,s_w)
logging.info('Output_file in %s', outputname)
logging.info(' name  %s', nn)
logging.info('count %s', count)




##read data from file 
data_ms2 = pd.read_csv(file_name,sep= "\t" ,header=0)
data_ms2['intial_type']='none' 
data_ms2["initial_intensity"]= -1
data_ms2["initial_rt_peak"]=-1
data_ms2["initial_SNR"]=-1
data_ms2["initial_log_L_R"]=-1
data_ms2["initiual_log_int"]=-1
data_ms2["modified_sequence"]= data_ms2["modified_sequence"].astype(str)

#data_ms2["initial_rt_peak"]=data_ms2["initial_rt_peak"].astype('float64')
#data_ms2['initial_intensity'] = data_ms2['initial_intensity'].astype('float64')
#data_ms2["initial_SNR"]= data_ms2['initial_SNR'].astype('float64')
#data_ms2["initial_log_L_R"]= data_ms2['initial_log_L_R'].astype('float64')
#data_ms2["initial_log_int"]= data_ms2['initial_log_int'].astype('float64')


data_ms2['search_type']='none'
data_ms2["search_intensity"]= -1
data_ms2["search_rt_peak"]=-1
data_ms2["search_SNR"]=-1
data_ms2["search_log_L_R"]=-1
data_ms2["search_log_int"]=-1




file_raw =  nn +  ".raw"
logging.info(file_raw)
loc= loc_raw + file_raw

c=0



def compute_shape_anchor ( data_xic,pos_p,val_max):
	log_L_R_rtpoint = [-1,-1]
	c_left=0
	find_5=False
	stop=False
	logging.info("rt L and R points init computation %i", c )
	while c_left < (pos_p-1) and stop != True :
	#print c_left
		if find_5==False and (data_xic.ix[(pos_p-1)-c_left,1].values <= (0.5 * val_max) ) :
			find_5=True
			#print "LWHM",c_left,data_xic.ix[(pos_p-1)-c_left,1]
			log_L_R_rtpoint[0] = data_xic.ix[(pos_p-1)-c_left,0].values*60
			stop=True
		c_left+=1
	find_5=False
	#find_1=False
	stop=False
	r_left=0
	while ((pos_p+1)+r_left  < len(data_xic) )  and stop != True :
		if find_5==False and data_xic.ix[(pos_p+1)+r_left,1].values <= (0.50 *val_max):
			find_5=True
			#print "RWHM",r_left,data_xic.ix[(pos_p+1)+r_left,1]
			log_L_R_rtpoint[1] = data_xic.ix[(pos_p+1)+r_left,0].values*60
			stop=True
		r_left += 1

	return (log_L_R_rtpoint )



def compute_apex( output, time_w ):
	try:
		data_xic = pd.read_csv(StringIO(output.strip()), sep=' ',names =['rt','intensity'] ,header=0 )
		ind_v = data_xic.index
		logging.info ("XIC shape   %i x 2",  data_xic.shape[0] )
		if data_xic[(data_xic['rt']> (time_w - s_w)) & ( data_xic['rt']< (time_w + s_w) )].shape[0] >=1:
			ind_v = data_xic.index
			pp=data_xic[ data_xic["intensity"]== data_xic[(data_xic['rt']> (time_w - s_w)) & ( data_xic['rt']< (time_w + s_w) )]['intensity'].max()].index
			pos_p = ind_v[pp]
			if pos_p.values.shape[0] > 1:
				logging.info("WARNINGS: Rt gap for the time windows searched. Probably the ppm values is too small %i", c )
			## retunr ---
				return ( [-1,-1,-1,-1,-1,-1] )
			val_max = data_xic.ix[pos_p,1].values
		else:
			logging.info("LW_BOUND finestra per il max %4.4f", time_w - s_w )
			logging.info("UP_BOUND finestra per il max %4.4f", time_w + s_w )
			logging.info(data_xic[(data_xic['rt']> (time_w - (+0.60))) & ( data_xic['rt']< (time_w + (s_w+0.60)) )]   )
			logging.info("WARNINGS: moff_rtWin_peak is not enough to detect the max peak line : %i", c )
			logging.info( 'MZ: %4.4f RT: %4.4f Mass: %i',row['mz'] ,row['rt'],index_ms2 )
		## retunr ---
			return ( [-1,-1,-1,-1,-1,-1] )
		pnoise_5 =  np.percentile(data_xic[(data_xic['rt']> (time_w - 1)) & ( data_xic['rt']< (time_w + 1) )]['intensity'],5)
		pnoise_10 = np.percentile(data_xic[(data_xic['rt']> (time_w - 1)) & ( data_xic['rt']< (time_w + 1) )]['intensity'],10)
	except (IndexError,ValueError,TypeError):
		logging.info("WARNINGS:  the time windows of the XIC is not enough to detect the max peak line : %i", c )
         	logging.info( 'MZ: %4.4f RT: %4.4f index: %i',row['mz'] ,row['rt'],index_ms2 )
                return ( -1)
 	except pd.parser.CParserError:
                #print file_name
                #print 'line',c
                logging.info( "WARNINGS: XIC not retrived line: %i",c)
		logging.info( 'MZ: %4.4f RT: %4.4f Mass: %i',row['mz'] ,row['rt'],index_ms2 )
		return ( [-1,-1,-1,-1,-1,-1] )
	logging.info( "Peak computed: %i",c)
	L_R_hm = compute_shape_anchor (data_xic,pos_p,val_max)
	
	return ( [val_max ,data_xic.ix[pos_p,0].values , L_R_hm[0], L_R_hm[1],  pnoise_5, pnoise_10 ] )



for index_ms2, row in data_ms2.iterrows():
	
	logging.info('line: %i',c)
	logging.info('line: %s',row['modified_sequence'])
	

	if not ( (  'K' in  row['modified_sequence'] ) or ( 'R' in row['modified_sequence'] ) ) :
		logging.info('line: %s', 'not labelled peptide')	
		## not labelled 
		mz_opt= "-mz="+str(row['mz'])
                if row['rt']==-1:
                        logging.info('rt not found. Wrong matched peptide in the mbr step line: %i',c)
                        c+=1
                        continue

                ##convert rt to sec to min
                time_w= row['rt']/60
                ## original s_W values is 0.40
                #=0.10 # time of refinement in minutes about 20 sec
                args = shlex.split("./txic " + mz_opt + " -tol="+ tol + " -t " + str(time_w - h_rt_w) + " -t "+ str(time_w +h_rt_w) +" " + loc   )
                p= subprocess.Popen(args,stdout=subprocess.PIPE)
                output, err = p.communicate()

                result= compute_apex( output, time_w )
                # result : aperx, rt_peak, l_rt, r_rt, pnoise5, p_noise10
                data_ms2.ix[index_ms2,12]= 'none'
                data_ms2.ix[index_ms2,13]= result[0] ## int ]
                data_ms2.ix[index_ms2,14]= result[1] * 60
                data_ms2.ix[index_ms2,15]=  20 * np.log10(  result[0]  / result[4] )
                data_ms2.ix[index_ms2,16]=  np.log2(abs(  result[0]   -  result[2] )   /  abs(  result[0]   -  result[3] )  )
                data_ms2.ix[index_ms2,17]=  np.log2(result[0])

	else:
	# check the modification
		if (  ('K<c13>' in  row['modified_sequence'] )  or ( 'R<C13N15>' in  row['modified_sequence'] )):
			## count ocicurence 
			logging.info('line: %s', 'heavy peptide')
			logging.info('init. mz %4.4f  init. charge %i', row['mz'],row['charge'] )
			K_n= len([n for n in xrange(len(row['modified_sequence'])) if row['modified_sequence'].find('K<c13>', n) == n])
			R_n= len([n for n in xrange(len(row['modified_sequence'])) if row['modified_sequence'].find('R<C13N15>', n) == n])
			#print K_n, R_n
			# it s heavy pep
			#exit()
			mz_opt= "-mz="+str(row['mz'])
			if row['rt']==-1:
				logging.info('rt not found. check your input data: %i',c)
				continue
		
		##convert rt to sec to min
			time_w = row['rt']/60
        	## original s_W values is 0.40
		#=0.10 # time of refinement in minutes about 20 sec
			args = shlex.split("./txic " + mz_opt + " -tol="+ tol + " -t " + str(time_w - h_rt_w) + " -t "+ str(time_w +h_rt_w) +" " + loc   )
			p= subprocess.Popen(args,stdout=subprocess.PIPE)
			output, err = p.communicate()
			result= compute_apex( output, time_w )
		# result : aperx, rt_peak, l_rt, r_rt, pnoise5, p_noise10 
			data_ms2.ix[index_ms2,12]= 'heavy'
			if result[0]!=-1:   
				data_ms2.ix[index_ms2,13]= result[0] ## int ]
				data_ms2.ix[index_ms2,14]= result[1] * 60
				data_ms2.ix[index_ms2,15]=  20 * np.log10(  result[0]  / result[5] )
				data_ms2.ix[index_ms2,16]=  np.log2(abs(  result[0]   -  result[2] )   /  abs(  result[0]   -  result[3] )  )
				data_ms2.ix[index_ms2,17]=  np.log2(result[0])

		## find the light petide:
			
			logging.info('line: %s', 'search a light peptide')	
			mass_mod  =  row['exp_mass'] - (K_n * 6 ) - (R_n * 10)
			mz_opt =   "-mz="+ str( (float( mass_mod) + float(row['charge']) )  / float( row['charge']) )
			logging.info(' Light Mod Mass : %f #K_mod %i #R_mod %i' , mass_mod,K_n,R_n)
                        logging.info(' Searched with mz: %4.4f Original charge : %i ' , (float( mass_mod) + float(row['charge']) )  / float( row['charge'])  ,row['charge'])
			#print 'light version : ', mass_light, float( mass_light) / float( row['charge']), 'Original mass', row['exp_mass']
			time_w= row['rt']/60
			args = shlex.split("./txic " + mz_opt + " -tol="+ tol + " -t " + str(time_w - h_rt_w) + " -t "+ str(time_w +h_rt_w) +" " + loc   )
                	p= subprocess.Popen(args,stdout=subprocess.PIPE)
                	output, err = p.communicate()
		
			result= compute_apex( output, time_w )
			# result : aperx, rt_peak, l_rt, r_rt, pnoise5, p_noise10
			data_ms2.ix[index_ms2,18]= 'light'
			if result[0]!=-1:
				data_ms2.ix[index_ms2,19]= result[0] ## int ]
				data_ms2.ix[index_ms2,20]= result[1]* 60
				data_ms2.ix[index_ms2,21]=  20 * np.log10(  result[0]  / result[5] )
				data_ms2.ix[index_ms2,22]=  np.log2(abs(  result[0]   -  result[2] )   /  abs(  result[0]   -  result[3] )  )
				data_ms2.ix[index_ms2,23]=  np.log2(result[0])
		#### peptide _ ligh
		else:
		#if (  ('K' in  row['modified_sequence'] )  or ( 'R' in  row['modified_sequence'] )):
                        ## count ocicurence
                        logging.info('line: %s', 'light peptide')
			logging.info('init. mz %4.4f  init. charge %i', row['mz'],row['charge'] )
                        K_n= len([n for n in xrange(len(row['modified_sequence'])) if row['modified_sequence'].find('K', n) == n])
                        R_n= len([n for n in xrange(len(row['modified_sequence'])) if row['modified_sequence'].find('R', n) == n])
                        #print K_n, R_n
                        # it s heavy pep
                        #exit()
                        mz_opt= "-mz="+str(row['mz'])
                        if row['rt']==-1:
                                logging.info('rt not found. check your input data: %i',c)
                                continue

                ##convert rt to sec to min
                        time_w = row['rt']/60
                ## original s_W values is 0.40
                #=0.10 # time of refinement in minutes about 20 sec
                        args = shlex.split("./txic " + mz_opt + " -tol="+ tol + " -t " + str(time_w - h_rt_w) + " -t "+ str(time_w +h_rt_w) +" " + loc   )
                        p= subprocess.Popen(args,stdout=subprocess.PIPE)
                        output, err = p.communicate()
                        result= compute_apex( output, time_w )
                # result : aperx, rt_peak, l_rt, r_rt, pnoise5, p_noise10
                        data_ms2.ix[index_ms2,12]= 'light'
                        if result[0]!=-1:
                                data_ms2.ix[index_ms2,13]= result[0] ## int ]
                                data_ms2.ix[index_ms2,14]= result[1] *60
                                data_ms2.ix[index_ms2,15]=  20 * np.log10(  result[0]  / result[5] )
                                data_ms2.ix[index_ms2,16]=  np.log2(abs(  result[0]   -  result[2] )   /  abs(  result[0]   -  result[3] )  )
                                data_ms2.ix[index_ms2,17]=  np.log2(result[0])

                ## find the light petide:

                        logging.info('line: %s', 'search a heavy peptide')
                        mass_mod  =  row['exp_mass'] + (K_n * 6 ) + (R_n * 10)
			logging.info(' Heavy Mod Mass : %f #K_mod %i #R_mod %i' , mass_mod,K_n,R_n)
			logging.info(' Searched with mz: %4.4f Original charge : %i ' , (float( mass_mod) + float(row['charge']) )  / float( row['charge']),row['charge'])
                        mz_opt =   "-mz="+ str( (float( mass_mod) + float(row['charge']) )  / float( row['charge']) )
                        #print 'light version : ', mass_light, float( mass_light) / float( row['charge']), 'Original mass', row['exp_mass']
                        time_w= row['rt']/60
                        args = shlex.split("./txic " + mz_opt + " -tol="+ tol + " -t " + str(time_w - h_rt_w) + " -t "+ str(time_w +h_rt_w) +" " + loc   )
                        p= subprocess.Popen(args,stdout=subprocess.PIPE)
                        output, err = p.communicate()

                        result= compute_apex( output, time_w )
                        # result : aperx, rt_peak, l_rt, r_rt, pnoise5, p_noise10
                        data_ms2.ix[index_ms2,18]= 'heavy'
                        if result[0]!=-1:
                                data_ms2.ix[index_ms2,19]= result[0] ## int ]
                                data_ms2.ix[index_ms2,20]= result[1] * 60
                                data_ms2.ix[index_ms2,21]=  20 * np.log10(  result[0]  / result[5] )
                                data_ms2.ix[index_ms2,22]=  np.log2(abs(  result[0]   -  result[2] )   /  abs(  result[0]   -  result[3] )  )
                                data_ms2.ix[index_ms2,23]=  np.log2(result[0])
		

	c+=1
##print result 
if first_file:
	data_ms2.to_csv(path_or_buf =outputname ,sep="\t",header=True,index=False )
else:
	data_ms2.to_csv(path_or_buf =outputname ,sep="\t",header=False,index=False )
