"""Usage: exp_1.py FILEPATH METHOD [-h] -t=<NTREES>

Options:
  -h --help
  -t --trees=<T>
"""
from docopt import docopt
from six.moves import cPickle as pickle
import numpy as np
from forest_class import forest
from generalTrees_class import flex_binary_trees, master_trees, getDepth
from tree_utilities import *
from make_synth_data import maybe_pickle
import os
import time

DEFAULT_NTREES = 500

def l2_loss(labels, predictions):
    loss = np.linalg.norm(np.array(labels)-np.array(predictions))
    return loss/(len(labels)**(1/2))

if __name__ == '__main__':
	### subsampling sizes
	subsample_sizes = [100, 500, 1000, 2000, 5000, 10000]
	##### load data
	args = docopt(__doc__)
	print('the input arguments received are', args)
	data_dict = maybe_pickle(args['FILEPATH'])
	data_tr = data_dict['x_tr']
	labels_tr = data_dict['y_tr']
	data_tt = data_dict['x_tt']
	labels_tt = data_dict['y_tt']
	##### load base tree 
	alg = args['METHOD']
	if alg == 'median':
		proj_design={'name':'projmat','params':{'name':'breiman','sparsity':1,'target_dim':1}}  
		split_design = {'name':'median'}
		stop_design={'name':'naive'}
		####
		kwargs = {'tree_design':{"tree":'flex','proj_design':proj_design,'split_design':split_design,'stop_design':stop_design}, 
			  'predictor_type':'regress'}
	elif alg == 'rp_spill':
		proj_design={'name':'projmat','params':{'name':'dasgupta','target_dim':1}}  
		split_design = {'name':'median_spill'}
		stop_design={'name':'naive'}
		kwargs = {'tree_design':{'tree':'flex', 'proj_design':proj_design, 'split_design':split_design, 
				   'stop_design':stop_design}, 'predictor_type':'regress'}
	elif alg == 'median_spill':
		proj_design={'name':'projmat','params':{'name':'breiman','sparsity':1,'target_dim':1}}  
		split_design = {'name':'median_spill'}
		stop_design={'name':'naive'}
		kwargs = {'tree_design':{"tree":'flex','proj_design':proj_design,'split_design':split_design,'stop_design':stop_design}, 
			  'predictor_type':'regress'}
	elif alg == 'breiman':
		proj_design={'name':'projmat','params':{'name':'breiman','sparsity':1,'target_dim':1}}  
		split_design = {'name':'cart', 'params':{'regress':True}}
		stop_design={'name':'naive'}
		kwargs = {'tree_design':{"tree":'flex','proj_design':proj_design,'split_design':split_design,'stop_design':stop_design}, 
			  'predictor_type':'regress'}
	else:
		print('Err: unrecognized method')
		exit(1)
	#### Train forest
	if '--trees' in args:
		kwargs['n_trees'] = int(args['--trees'])
	else:
		kwargs['n_trees'] = int(DEFAULT_NTREES)
	fc_estimator = forest(data_tr, labels=labels_tr, **kwargs)
	test_error_list = list()
	for n in subsample_sizes:
		fc_estimator.reset_sample_size(n_samples=n)
		fc_estimator.train()
		test_error_list.append(l2_loss(labels_tt, fc_estimator.predict(data_tt)))
	directory = args['FILEPATH']+'/results/' + alg + '/'
	if not os.path.exists(directory):
		os.makedirs(directory)
	timestamp = time.strftime("%Y%m%d-%H%M%S")
	out_dataname = directory+'t'+str(kwargs['n_trees'])+'_'+timestamp
	maybe_pickle(out_dataname, data=outdataname)
		#directory = args['FILEPATH'] + '/trained_forests/' + alg + '/'
		#if not os.path.exists(directory):
		# 	os.makedirs(directory)
# 		timestamp = time.strftime("%Y%m%d-%H%M%S")
# 		out_dataname = directory+'t'+str(kwargs['n_trees'])+'n'+str(n)+'_'+timestamp
# 		maybe_pickle(out_dataname, data=fc_estimator)
	# predict_on_train = fc_estimator.predict(data_tr)
	# 	print('Training loss: %f' %l2_loss(labels_tr, predict_on_train))
	# 	predict_on_test = fc_estimator.predict(data_tt)
	# 	print('Test loss: %f' %l2_loss(labels_tt, predict_on_test))
	


