import numpy as np
import glob as glob
import pandas as pd
import os as os
import sys
import subprocess
import shlex 
import  logging
from StringIO import StringIO
import argparse
import pickle 
### input###
## - MS2 ID file
## - tol
## - half rt time window in minute
###### output
##  list of intensities..+



def compute_shape_anchor ( data_xic,pos_p,val_max):
	log_L_R_rtpoint = [-1,-1]
	c_left=0
	find_5=False
	stop=False
	logging.info("	rt L and R points init computation %i" )
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



def compute_apex( output, time_w,c, mz_val,s_w ):
	try:
		data_xic = pd.read_csv(StringIO(output.strip()), sep=' ',names =['rt','intensity'] ,header=0 )
		ind_v = data_xic.index
		logging.info ("	XIC shape   %i x 2",  data_xic.shape[0] )
		if data_xic[(data_xic['rt']> (time_w - s_w)) & ( data_xic['rt']< (time_w + s_w) )].shape[0] >=1:
			ind_v = data_xic.index
			pp=data_xic[ data_xic["intensity"]== data_xic[(data_xic['rt']> (time_w - s_w)) & ( data_xic['rt']< (time_w + s_w) )]['intensity'].max()].index
			pos_p = ind_v[pp]
			if pos_p.values.shape[0] > 1:
				logging.info("	--WARNINGS: Rt gap for the time windows searched. Probably the ppm values is too small %i", c )
			## retunr ---
				return ( [-1,-1,-1,-1,-1,-1] )
			val_max = data_xic.ix[pos_p,1].values
		else:
			logging.info("  --Rt point  %4.4f", time_w )
			logging.info("	--LW_BOUND %4.4f", time_w - s_w )
			logging.info("	--UP_BOUND %4.4f", time_w + s_w )
			logging.info(data_xic[(data_xic['rt']> (time_w - (+1))) & ( data_xic['rt']< (time_w + (+1)) )]   )
			logging.info("	--WARNINGS: moff_rtWin_peak is not enough to detect the max peak line : %i", c )
			logging.info('	--MZ: %4.4f RT: %4.4f  ',mz_val ,time_w )
		## retunr ---
			return ( [-1,-1,-1,-1,-1,-1] )
		pnoise_5 =  np.percentile(data_xic[(data_xic['rt']> (time_w - 1)) & ( data_xic['rt']< (time_w + 1) )]['intensity'],5)
		pnoise_10 = np.percentile(data_xic[(data_xic['rt']> (time_w - 1)) & ( data_xic['rt']< (time_w + 1) )]['intensity'],10)
	except (IndexError,ValueError,TypeError):
		logging.info("	--WARNINGS:  the time windows of the XIC is not enough to detect the max peak line : %i", c )
         	logging.info('	--MZ: %4.4f RT: %4.4f ',mz_val ,time_w )
                return ( -1)
 	except pd.parser.CParserError:
                #print file_name
                #print 'line',c
                logging.info( "	--WARNINGS: XIC not retrived line: %i",c)
		logging.info('	--MZ: %4.4f RT: %4.4f Mass:',mz_val , time_w )
		return ( [-1,-1,-1,-1,-1,-1] )
	logging.info("Peak computed")
	L_R_hm = compute_shape_anchor (data_xic,pos_p,val_max )
	
	return ( [val_max ,data_xic.ix[pos_p,0].values , L_R_hm[0], L_R_hm[1],  pnoise_5, pnoise_10,data_xic ] )


def run_apex( file_name, tol, h_rt_w, s_w, s_w_match, loc_raw, loc_output):
	name = os.path.basename(file_name).split('.')[0]
	logging.basicConfig(filename=  os.path.join(loc_output,name + '__moff.log')  ,filemode='w',level=logging.INFO)


	outputname =   os.path.join(loc_output,name + '__result.txt') 
	logging.info('moff Input file %s  XIC_tol %s XIC_win %4.4f moff_rtWin_peak %4.4f ',file_name,tol,h_rt_w,s_w)
	logging.info('Output_file in %s', outputname)
	logging.info(' name  %s', name)

	moff_path= os.path.dirname( sys.argv[0])

	##read data from file
	data_ms2 = pd.read_csv(file_name,sep= "\t" ,header=0)

	index_offset = data_ms2.columns.shape[0] - 1

	data_ms2['intial_label']='none'
	data_ms2["initial_intensity"]= -1
	data_ms2["initial_rt_peak"]=-1
	data_ms2["initial_SNR"]=-1
	data_ms2["initial_log_L_R"]=-1
	data_ms2["initiual_log_int"]=-1
	#data_2["modified_sequence"]= data_ms2["modified_sequence"].astype(str)

	#data_2["initial_rt_peak"]=data_ms2["initial_rt_peak"].astype('float64')
	#data_2['initial_intensity'] = data_ms2['initial_intensity'].astype('float64')
	#data_2["initial_SNR"]= data_ms2['initial_SNR'].astype('float64')
	#data_2["initial_log_L_R"]= data_ms2['initial_log_L_R'].astype('float64')
	#data_2["initial_log_int"]= data_ms2['initial_log_int'].astype('float64')


	data_ms2['search_label']='none'
	data_ms2["search_intensity"]= -1
	data_ms2["search_rt_peak"]=-1
	data_ms2["search_SNR"]=-1
	data_ms2["search_log_L_R"]=-1
	data_ms2["search_log_int"]=-1


	file_raw =data_ms2['Raw file'].unique()[0] +  ".raw"

	logging.info(file_raw)


	loc= loc_raw + file_raw

	c=0
	raw_xic={}
	for index_ms2, row in data_ms2.iterrows():
		xic_list = [-1,-1]
		logging.info('line: %i',c)
		logging.info('line: %s',row['Sequence'])
		if ( pd.isnull( row['Labeling State'] )  ):
			#couplespoint 
                                ## count ocicurence
			logging.info('line: %s', 'paired peptide using max intensity 1  as Heavy labeled')
                        logging.info('line: %s', 'heavy labeled')
                        logging.info('init. mz %4.4f init. charge %i', row['Max intensity m/z 1'],row['charge'] )
                                #exit()
			mz_opt= "-mz="+str(row['Max intensity m/z 1'])
			if row['rt']==-1:
				logging.info('rt not found. check your input data: %i',c)
				continue

                               ##convert rt to sec to min
			time_w = row['rt'] #/60
			args = shlex.split("./txic " + str(mz_opt) + " -tol="+ str(tol) + " -t " + str(time_w - h_rt_w) + " -t "+ str(time_w +h_rt_w) +" " + loc   )
			p= subprocess.Popen(args,stdout=subprocess.PIPE)
			output, err = p.communicate()
			result= compute_apex( output, time_w ,c,row['Max intensity m/z 1'],s_w)
                               # result : aperx, rt_peak, l_rt, r_rt, pnoise5, p_noise10
			data_ms2.ix[index_ms2,(index_offset+1)] = 'heavy'
			if result[0]!=-1:
				data_ms2.ix[index_ms2, (index_offset+2)]= result[0] ## int ]
				data_ms2.ix[index_ms2, (index_offset+3) ]= result[1] * 60
				data_ms2.ix[index_ms2, (index_offset+4)]=  20 * np.log10(  result[0]  / result[5] )
				data_ms2.ix[index_ms2, (index_offset+5)]=  np.log2(abs(  result[0]   -  result[2] )   /  abs(  result[0]   -  result[3] )  )
				data_ms2.ix[index_ms2, (index_offset+6)]=  np.log2(result[0])
				xic_list[0] = result[6]
                               ## find the light petide:
                        logging.info('line: %s', 'search a light peptide using max intensity mz 0')	
			#mass_mod= row['Max intensity m/z 0']
			mz_opt= "--mz="+ str(row['Max intensity m/z 0'])
				#mass_mod  =  row['exp_mass'] - (K_n * 6 ) - (R_n * 10)
				#mz_opt =   "-mz="+ str( (float( mass_mod) + float(row['charge']) )  / float( row['charge']) )
			#logging.info(' Light oxICAT Mod Mass : %f ' , mass_mod)
			logging.info(' Searched with mz: %4.4f original charge : %i ' , row['Max intensity m/z 0'],row['charge'])
			time_w= row['rt'] #/60
			args = shlex.split("./txic " +  mz_opt  + " -tol="+ str(tol) + " -t " + str(time_w - h_rt_w) + " -t "+ str(time_w +h_rt_w) +" " + loc   )
			p= subprocess.Popen(args,stdout=subprocess.PIPE)
			output, err = p.communicate()

			result= compute_apex( output, time_w,c, row['Max intensity m/z 0'],s_w  )
				# result : aperx, rt_peak, l_rt, r_rt, pnoise5, p_noise10
			data_ms2.ix[index_ms2,(index_offset+7)]= 'light'
			if result[0]!=-1:
				data_ms2.ix[index_ms2,(index_offset+8) ]= result[0] ## int ]
				data_ms2.ix[index_ms2,(index_offset+9)  ]= result[1]* 60
				data_ms2.ix[index_ms2, (index_offset+10) ]=  20 * np.log10(  result[0]  / result[5] )
				data_ms2.ix[index_ms2, (index_offset+11) ]=  np.log2(abs(  result[0]   -  result[2] )   /  abs(  result[0]   -  result[3] )  )
				data_ms2.ix[index_ms2, (index_offset+12)  ]=  np.log2(result[0])
				xic_list[1] = result[6]

		else:
			if (   row['Labeling State']== 1 )  :
				## count ocicurence 
				logging.info('line: %s', 'heavy labeled')
				logging.info('init. mz %4.4f  init. charge %i', row['mz'],row['charge'] )
				#exit()
				mz_opt= "-mz="+str(row['mz'])
				if row['rt']==-1:
					logging.info('rt not found. check your input data: %i',c)
					continue
			       
			       ##convert rt to sec to min
				time_w = row['rt'] #/60
			       ## original s_W values is 0.40
			       #=0.10 # time of refinement in minutes about 20 sec
				args = shlex.split("./txic " + str(mz_opt) + " -tol="+ str(tol) + " -t " + str(time_w - h_rt_w) + " -t "+ str(time_w +h_rt_w) +" " + loc   )
				p= subprocess.Popen(args,stdout=subprocess.PIPE)
				output, err = p.communicate()
				result= compute_apex( output, time_w ,c,row['mz'],s_w)
			       # result : aperx, rt_peak, l_rt, r_rt, pnoise5, p_noise10 
				data_ms2.ix[index_ms2,(index_offset+1)] = 'heavy'
				if result[0]!=-1:   
					data_ms2.ix[index_ms2, (index_offset+2)]= result[0] ## int ]
					data_ms2.ix[index_ms2, (index_offset+3) ]= result[1] * 60
					data_ms2.ix[index_ms2, (index_offset+4)]=  20 * np.log10(  result[0]  / result[5] )
					data_ms2.ix[index_ms2, (index_offset+5)]=  np.log2(abs(  result[0]   -  result[2] )   /  abs(  result[0]   -  result[3] )  )
					data_ms2.ix[index_ms2, (index_offset+6)]=  np.log2(result[0])
					xic_list[0] = result[6]
			       ## find the light petide:
				
				logging.info('line: %s', 'search a light peptide')	
				mass_mod= row['l_mass']
				mz_opt= "--mz="+ str(row['mz_l'])
				#mass_mod  =  row['exp_mass'] - (K_n * 6 ) - (R_n * 10)
				#mz_opt =   "-mz="+ str( (float( mass_mod) + float(row['charge']) )  / float( row['charge']) )
				logging.info(' Light oxICAT Mod Mass : %f ' , mass_mod)
				logging.info(' Searched with mz: %4.4f original charge : %i ' , row['mz_l'],row['charge'])
				#print 'light version : ', mass_light, float( mass_light) / float( row['charge']), 'Original mass', row['exp_mass']
				time_w= row['rt'] #/60
				args = shlex.split("./txic " +  mz_opt  + " -tol="+ str(tol) + " -t " + str(time_w - h_rt_w) + " -t "+ str(time_w +h_rt_w) +" " + loc   )
				p= subprocess.Popen(args,stdout=subprocess.PIPE)
				output, err = p.communicate()
			       
				result= compute_apex( output, time_w,c, row['mz_l'],s_w  )
				# result : aperx, rt_peak, l_rt, r_rt, pnoise5, p_noise10
				data_ms2.ix[index_ms2,(index_offset+7)]= 'light'
				if result[0]!=-1:
					data_ms2.ix[index_ms2,(index_offset+8) ]= result[0] ## int ]
					data_ms2.ix[index_ms2,(index_offset+9)  ]= result[1]* 60
					data_ms2.ix[index_ms2, (index_offset+10) ]=  20 * np.log10(  result[0]  / result[5] )
					data_ms2.ix[index_ms2, (index_offset+11) ]=  np.log2(abs(  result[0]   -  result[2] )   /  abs(  result[0]   -  result[3] )  )
					data_ms2.ix[index_ms2, (index_offset+12)  ]=  np.log2(result[0])
					xic_list[1] = result[6]
			       #### peptide _ ligh
			else:
			       #if (  ('K' in  row['modified_sequence'] )  or ( 'R' in  row['modified_sequence'] )):
				## count ocicurence
				logging.info('line: %s', 'light labeled')
				logging.info('init. mz %4.4f  init. charge %i', row['mz'],row['charge'] )
				mz_opt= "-mz="+str(row['mz'])
				if row['rt']==-1:
					logging.info('rt not found. check your input data: %i',c)
					continue

			       ##convert rt to sec to min
				time_w = row['rt'] #/60
			       ## original s_W values is 0.40
			       #=0.10 # time of refinement in minutes about 20 sec
				args = shlex.split("./txic " + mz_opt + " -tol="+ str(tol) + " -t " + str(time_w - h_rt_w) + " -t "+ str(time_w +h_rt_w) +" " + loc   )
				p= subprocess.Popen(args,stdout=subprocess.PIPE)
				output, err = p.communicate()
				result= compute_apex( output, time_w ,c,row['mz'],s_w)
			       # result : aperx, rt_peak, l_rt, r_rt, pnoise5, p_noise10
				data_ms2.ix[index_ms2, (index_offset+1) ]= 'light'
				if result[0]!=-1:
					data_ms2.ix[index_ms2, (index_offset+2)   ]= result[0] ## int ]
					data_ms2.ix[index_ms2, (index_offset+3)  ]= result[1] *60
					data_ms2.ix[index_ms2, (index_offset+4) ]=  20 * np.log10(  result[0]  / result[5] )
					data_ms2.ix[index_ms2,  (index_offset+5)  ]=  np.log2(abs(  result[0]   -  result[2] )   /  abs(  result[0]   -  result[3] )  )
					data_ms2.ix[index_ms2, (index_offset+6)]=  np.log2(result[0])
					xic_list[0] = result[6]
				
			       ## find the light petide:

				logging.info('line: %s', 'search a heavy peptide')
				mass_mod= str(row['h_mass'])
				mz_opt= "--mz="+ str(row['mz_h'])
				logging.info('  Light oxICAT Mod Mass %s ' , mass_mod)
				logging.info('  Searched with mz: %4.4f Original charge : %i ' ,  row['mz_h'],row['charge'] )
				time_w= row['rt'] #/60
				args = shlex.split("./txic " + mz_opt + " -tol="+ str(tol) + " -t " + str(time_w - h_rt_w) + " -t "+ str(time_w +h_rt_w) +" " + loc   )
				p= subprocess.Popen(args,stdout=subprocess.PIPE)
				output, err = p.communicate()

				result= compute_apex( output, time_w ,c,row['mz_h'],s_w)
				# result : aperx, rt_peak, l_rt, r_rt, pnoise5, p_noise10
				data_ms2.ix[index_ms2,(index_offset+7)]= 'heavy'
				if result[0]!=-1:
					data_ms2.ix[index_ms2,(index_offset+8) ]= result[0] ## int ]
					data_ms2.ix[index_ms2,(index_offset+9)  ]= result[1]* 60
					data_ms2.ix[index_ms2, (index_offset+10) ]=  20 * np.log10(  result[0]  / result[5] )
					data_ms2.ix[index_ms2, (index_offset+11) ]=  np.log2(abs(  result[0]   -  result[2] )   /  abs(  result[0]   -  result[3] )  )
					data_ms2.ix[index_ms2, (index_offset+12)  ]=  np.log2(result[0])	       
					xic_list[1] = result[6]
		raw_xic[c]=xic_list
		c+=1
		##print result 
	xic_data = open( os.path.join(loc_output,name + '__xic.dmp') ,'wb')
	pickle.dump(raw_xic,xic_data)
	xic_data.close()
	print '..............apex terminated'
    	print 'Writing result in %s' % (outputname)
	data_ms2.to_csv(path_or_buf =outputname ,sep="\t",header=True,index=False )
	return 0
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='moFF input parameter')
	parser.add_argument('--input', dest='name', action='store', help='specify the input file with the  of MS2 peptides',
                        required=True)
	parser.add_argument('--tol', dest='toll', action='store', type=float,
                        help='specify the tollerance parameter in ppm', required=True)

    	parser.add_argument('--rt_w', dest='rt_window', action='store', type=float, default=3,
                        help='specify rt window for xic (minute). Default value is 3 min', required=False)

    	parser.add_argument('--rt_p', dest='rt_p_window', action='store', type=float, default=0.1,
                        help='specify the time windows for the peak ( minute). Default value is 0.1 ', required=False)

	parser.add_argument('--rt_p_search', dest='rt_p_window_match', action='store', type=float, default=0.4,
                        help='specify the time windows for the searched  peptide peak ( minute). Default value is 0.4 ',
                        required=False)

    	parser.add_argument('--raw_repo', dest='raw', action='store', help='specify the raw file repository ',
                        required=True)

    	parser.add_argument('--output_folder', dest='loc_out', action='store', default='', help='specify the folder output',
                        required=False)

    	args = parser.parse_args()
    	file_name = args.name
    	tol = args.toll
    	h_rt_w = args.rt_window
    	s_w = args.rt_p_window
    	s_w_match = args.rt_p_window_match
    	loc_raw = args.raw
    	loc_output = args.loc_out

	run_apex(file_name, tol, h_rt_w, s_w, s_w_match, loc_raw, loc_output)

