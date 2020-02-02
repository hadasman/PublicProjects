from __future__ import division
import os, sys, pdb
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
try:
	import cPickle # Python2
except:
	import _pickle as cPickle # Python3

# ================================================ Define Local Functions =================================================

def CreateCompartment(comp, **kwargs):
	'''
	Create NEURON Section, and set user-specified attributes for it.
	'''

	var = h.Section(name=comp)

	var.insert('pas')

	if 'soma' in comp:
		# Insert passive conductance if Section is soma
		var.insert('kv')
		var.insert('na')

	for attribute in kwargs:
		setattr(var, attribute, kwargs[attribute])

	return var

def CreateSynLocs(start, interval, density):
	'''
	Return uniformly-spaced synapse locations, given a starting position, interval length which included synapses and 
	synapse density (all given in units of micro-meter).
	'''
	
	# Make locations
	locs = sorted(np.arange(start, start + interval, 1/density))
	# Convert to fraction of branch length (as required by the function using locations as input)
	locs = [i / L for i in locs]

	if len([i for i in locs if i>1]) > 0:
		# Sanity check
		raise Exception('Synapse location goes beyond cable!')

	return locs, len(locs)

def PlotGs(exc_syns, inh_syns):

	'''
	Plot channel conductances as a function of time.
	'''
	
	# Record single excitatory synapse conductances from middle of interval
	g_AMPA = h.Vector(); g_AMPA.record(exc_syns[int(len(exc_syns)/2)]._ref_g_AMPA)
	g_NMDA = h.Vector(); g_NMDA.record(exc_syns[int(len(exc_syns)/2)]._ref_g_NMDA)
	
	# Record single inhibitory synapse from middle of interval
	try: # Inhibitory synapse not always present
		g_inh = h.Vector()
		g_inh.record(inh_syns[int(len(inh_syns)/2)]._ref_g)
	except:
		g_inh = []

	t = h.Vector(); t.record(h._ref_t)
	h.finitialize()
	h.run()

	# Plot single synapse Gs
	fig, ax = plt.subplots()
	ax.set_xlabel('T (ms)')
	ax.set_ylabel('G ($\mu$S)')
	ax.plot(t, g_AMPA, label='AMPA')
	ax.plot(t, g_NMDA, label='NMDA')

	try:
		ax.plot(t, g_inh, label='Inh.')
		ax.set_title('Single Synapse Conductances for AMPA, NMDA and Inh. synapses')
	except:
		ax.set_title('Single Synapse Conductances for AMPA & NMDA synapses')

	ax.legend()

def CalcInputR(sec):
	'''
	Calculate input resistance at middle of given Section.
	'''

	im = h.Impedance()
	im.loc(0.5, sec=sec)
	im.compute(0, 1) # 1st argument: impedance frequency to compute; 2nd argument: when ==1, the calculation considers the effect of differential gating states
	Rinput = im.input(0.5, sec=sec)
	return Rinput

def CalcAxialR(comp):
	"""
	Calculates compartment/Section resistance given resistivity, diameter and length of compartment.

	Inputs:
		- comp: compartment object (should include Ra, diam and L properties for volume calculation)
			* Ra = resistivity (specific resistance for unit area) of compartment [ohm*cm] (=> property of cytoplasm)

	Outputs:
		- Ra_T: Total axial resistance [M-ohm] (=> property of cable)
	"""

	L = comp.L * (1e-4) 			# Convert from um to [cm]
	diam = comp.diam * (1e-4)/2   	# Convert from um to [cm]
	A = np.pi * (diam ** 2)			# in units [cm^2]

	Ra_T = (comp.Ra * L) / A 		# units: [ohm-cm] * [cm] / [cm^2] = [ohm]
	Ra_T = Ra_T * (1e-6)			# Convert to M-ohm

	return Ra_T

def UserInput():
	'''
	Deal with input that should be specified by user. Either take from use-given input or ask user.
	'''

	# If not given as input to script- ask user
	if len(sys.argv)<2:
		ok_ = 0
		while not ok_:
			which_params = input('Old or MIT parameters? ')
			put_spines = input('Put Spines? (yes/no) ')
			put_soma = input('Put Soma? (yes/no) ')

			# Follow-up questions, if user selected to put spines
			if put_spines == 'yes':
				where_E_syns = input('E-Synapses on spines or shaft? ')
				where_I_syns = input('I-Synapses on spines or shaft? ')
			else:
				where_E_syns, where_I_syns = 'shaft', 'shaft'

			# Idiot-Proofing
			if (which_params in ['Old','MIT']) \
				and (put_spines in ['yes','no']) \
				and (put_soma in ['yes','no']) \
				and (where_E_syns in ['spines','shaft','spine']) \
				and (where_I_syns in ['spines','shaft','spine']):
					ok_ = 1
			else:
				print('One or more invalid values. Try Again! ')
	# Option for user who knows what's up
	else: 
		which_params = sys.argv[1]
		put_spines = sys.argv[2]
		put_soma = sys.argv[3]
		where_E_syns = sys.argv[4]
		where_I_syns = sys.argv[5]

	return which_params, put_spines, put_soma, where_E_syns, where_I_syns

def ImageSynapses(exc_synapses, inh_synapses):
	'''
	Show Morphological structure, with excitatory and inhibitory synapses.
	'''

	h.define_shape()
	image_syn = h.Shape()
	for i in exc_synapses:
		image_syn.point_mark(i, 2)
	for i in inh_synapses:
		image_syn.point_mark(i, 3)

	return image_syn

# ================================================ Import Special Modules =================================================
# Adjusted to work on local computer. For compatibility with different computer, change <user_path> to local path for the
# folder MIT_spines (containing this script and NEURON compiled .mod files). This folder should be contained in a folder
# that also contains the folder named 'function_scripts', which contains run_functions.py and synapse_functions.py.

home_path = os.path.expanduser('~') 
user_path = 'Documents/GitHubProjects/MIT_spines'
path_ = '%s/%s' %(home_path, user_path)
os.chdir(path_)
from neuron import h, gui

os.chdir('../function_scripts')
from run_functions import RunMain, RunSim
import run_functions

os.chdir('../function_scripts')
from synapse_functions import PutSyns, PutSynsOnSpines, PutSpines
import synapse_functions

# ================================================ Set Simulation Parameters =================================================
h.v_init = -75	# Set membrane resting potential
h.tstop = 250 	# Set the simulation time
num_iters = 100 # Set number of simulation iterations

start_syns = 0  # MY NOTES: 25 # With spines: 74, without spines: 50 (L = 1000) # 70 - 4.5 - (24.13793103448276-1.75) # 50

exc_inh_dt = 29.3 # MY NOTES: When syns on all cable: 29.3 without spines, 25.5/26 with spines (syns on shaft)
exc_tstart = 100
inh_tstart = exc_tstart + exc_inh_dt

for att in ['exc_tstart', 'inh_tstart']:
	setattr(synapse_functions, att, eval(att))

spike_tstart = exc_tstart + 50
recovered_thresh = -40

# ================================================ Set Section Parameters =================================================

which_params, put_spines, put_soma, where_E_syns, where_I_syns = UserInput()

if which_params == 'old':
	L = 1000
	exc_dense = 1.5
	inh_dense = 0.2
	n_exc = 27
elif which_params == 'MIT':
	L = 80 				# From branch data values
	exc_dense = 0.4 	# MT NOTES: 0.415#0.4 # 0.58
	inh_dense = 0.2
	n_exc = 14     		# MY NOTES: int(exc_dense * L)

synapse_filled_length = L
inh_start = start_syns # Same now but might change later, so I keep them separated
exc_start = start_syns # Same now but might change later...

exc_locs, n_exc = CreateSynLocs(exc_start, synapse_filled_length, exc_dense)
inh_locs, n_inh = CreateSynLocs(inh_start, synapse_filled_length, inh_dense)

# ================================================ Create Cable and Put Spines =================================================

dend = CreateCompartment('dend', \
							L = L, \
							diam = 0.4, \
							Ra = 150, \
							e_pas = h.v_init, \
							g_pas = 1.0 / 1500.0, \
							nseg = 500 * 5, \
							cm = 1)

# Put soma if requested
if put_soma == 'yes' or put_soma == 'y':
	soma = CreateCompartment('soma', \
							L = 3, \
							diam = 3, \
							Ra = 150, \
							e_pas = h.v_init, \
							g_pas = 1.0 / 1500.0, \
							nseg = int(L) * 5, \
							cm = 1)
# Put soma if requested
if put_spines == 'yes' or put_spines == 'y':
	spine_heads, spine_necks = PutSpines('cable', dend, exc_locs, \
												neck_diam = 0.0394, \
												neck_len = 1, \
												head_radius = 0.297, \
												Ra = 18.5, \
												cm=dend.cm, \
												e_pas = dend.e_pas, \
												g_pas=4.6716e-5)
	# Adjust spine neck resistance (non-specific)
# Put synapses on Section
if where_E_syns == 'spines' or where_E_syns == 'spine':
	exc_synapses, spine_netstim, spine_netcon = PutSynsOnSpines(spine_heads, 'exc')
	
	# Record voltage at each spine head
	n_spines = len(spine_heads)
	spines_v = [h.Vector()] * n_spines
	[spines_v[i].record(spine_heads[i](1)._ref_v) for i in range(n_spines)]
elif where_E_syns == 'shaft':
	exc_synapses, exc_netstim, exc_netcon = PutSyns(dend, exc_locs, 'exc')
else:
	raise Exception('Didn''t specify synapse location')

record_loc = exc_locs[int(n_exc/2)]
t = h.Vector(); t.record(h._ref_t)  # Record simulation time

run_functions_att = 	['exc_start', \
						'inh_start', \
						'num_iters', \
						'synapse_filled_length', \
						'exc_synapses', \
						'exc_dense', \
						'inh_dense',\
		    			'record_loc', \
		    			't', \
		    			'spike_tstart', \
		    			'recovered_thresh', \
		    			'where_E_syns', \
		    			'where_I_syns']
for att in run_functions_att:
	setattr(run_functions, att, eval(att))

# ================================================ Prepare & Run Simulation (Only E) =================================================
dend_v = RunSim(dend, record_loc)	# Run simulation and return record of voltage

fig, trace_ax = plt.subplots()
trace_ax.set_facecolor('black')
trace_ax.set_xlabel('Time (ms)')
trace_ax.set_ylabel('Voltage (mV)')
trace_ax.set_title('Synapses on %.2f$\mu$m starting from %.1f$\mu$m, Exc. density %.2f, Inh. density %.2f \n \
# 	C$_m$ = %.1f[$\\frac{muF}{cm^2}$], R$_m$ = %s[$\Omega cm^2$]'\
	%(synapse_filled_length, start_syns, exc_dense, inh_dense, int(dend.cm), int(1/dend.g_pas)), size=10)

trace_ax.plot(t, dend_v, color='white', LineStyle='--', label='Only E', zorder=num_iters+2)

# ================================================ Prepare & Run Simulation (Even E & I) =================================================
if where_I_syns == 'shaft':
	inh_synapses, inh_netstim, inh_netcon = PutSyns(dend, inh_locs, 'inh')
elif where_I_syns == 'spine' or where_I_syns=='spines':
	inh_synapses, inh_netstim, inh_netcon = PutSynsOnSpines(spine_heads, 'inh')

run_functions.inh_synapses = inh_synapses
dend_v = RunSim(dend, record_loc)
trace_ax.plot(t, dend_v, LineWidth=3, color='white', label='Uniform E&I', zorder=num_iters+1)

# PlotGs(exc_synapses, inh_synapses)
# ================================================ Prepare & Run Simulation (Random Inhibition) =================================================
recovered_seq, abolished_seq = [], []
recovered_seq, abolished_seq = RunMain(dend, trace_ax, 'Uniform', 'Random', 'red', recovered_seq, abolished_seq)

# ================================================ Prepare & Run Simulation (Random Excitation and Inhibition) =================================================
recovered_seq, abolished_seq = RunMain(dend, trace_ax, 'Random', 'Random', 'blue', recovered_seq, abolished_seq)

#image_syn = ImageSynapses(exc_synapses, inh_synapses)

trace_ax.legend(loc='upper right')
# ================================================ Analyze recovery/abolish sequences =================================================

if recovered_seq and abolished_seq:
	condition = input('Condition name for plots? ')
	which_analyze = input('Which synapses to analyze? (exc / inh)')

	rec_color = 'lightblue'
	abo_color = 'orchid'

	# === Extract Data ===
	# Extract Nearest Neighbor 
	def NN_dist(recovered_seq, abolished_seq, which_analyze):
		'''
		Get nearest neighbor distances. 
		Input:
			- recovered_seq: dictionary of synapse locations in which NMDA-spike recovered (divided to excitatory and inhibitory)
					example: {'exc': [[locs(1)]...[locs(n)]], 'inh': [[locs(1)]...[locs(n)]]}
			- abolished_seq: same dictionary, for locations in which NMDA-spike was abolished
			- which_analyze: string indicating which synapses to check (exc/inh)
		Output:
			returns nearest neighbor distances for all synapse arrangements. Recovered/abolished in different arrays:
			NN_rec, NN_abo
		'''

		def NN_calc(locs):
			NN_dists = []
			locs = [i*dend.L for i in locs]
			
			# Pad locs to consider 1st and last synapse NNs
			locs = [-1000] + locs + [1000]
			for i in range(1, len(locs)-1):
				dist1 = abs(locs[i] - locs[i-1])
				dist2 = abs(locs[i] - locs[i+1])
				NN_dists.append(min(dist1, dist2))
			
			if max(NN_dists)>900: # Sanity check
				pdb.set_trace()

			return NN_dists

		# Get NN distance for relevant synapse (exc/inh) locations
		NN_rec = [NN_calc(i[which_analyze]) for i in recovered_seq] 
		NN_rec = [j for i in NN_rec for j in i]
		NN_abo = [NN_calc(i[which_analyze]) for i in abolished_seq]
		NN_abo = [j for i in NN_abo for j in i]

		return NN_rec, NN_abo
	NN_rec, NN_abo = NN_dist(recovered_seq, abolished_seq, which_analyze)
	
	# Extract Relative Location 
	loc_rec = [i[which_analyze] for i in recovered_seq]
	loc_rec = [m*dend.L for m in [j for i in loc_rec for j in i]] # Pool samples to 1 and convert to micro-m
	loc_abo = [i[which_analyze] for i in abolished_seq]
	loc_abo = [m*dend.L for m in [j for i in loc_abo for j in i]] # Pool samples to 1 and convert to micro-m
	
	# Extract Distance of synapses from middle of cable
	dist_from_mid_rec = [abs(i - (dend.L/2)) for i in loc_rec]
	dist_from_mid_abo = [abs(i - (dend.L/2)) for i in loc_abo]

	# === Analyze & Plot ===
	# Bar plots
	def MeanBars_plot(group1, group2, suptitle_='', xlabel_='', match_sample_size=False):

		fig, h_ax = plt.subplots()

		matched_sample_size = min(len(group1), len(group2))

		if match_sample_size:
			tscore, pval = ttest_ind(group1[:matched_sample_size], group2[:matched_sample_size])
		else:
			tscore, pval = ttest_ind(group1, group2)
		
		h_ax.bar([0, 1], [np.mean(group1), np.mean(group2)], yerr=[np.std(group1)/np.sqrt(len(group1)), np.std(group2)/np.sqrt(len(group2))], color='coral')
		h_ax.set_xticks([0, 1])
		h_ax.set_xticklabels(['Recovered', 'Abolished'])
		fig.suptitle('%s '%suptitle_ +  '(T-test result: p = %.2f, N = %.0f) \n (Error bars are SEM)'%(pval, matched_sample_size))
		h_ax.set_title('%s, '%condition + 'branch length = %.0f$\mu$m, n_exc = %.0f, start synapses at %.1f$\mu$m, '%(dend.L, n_exc, start_syns) + '%s densities'%which_params, size=10)
		h_ax.set_ylabel('%s ($\mu$m)'%xlabel_)

		return h_ax, tscore, pval
	NN_ax, NN_t, NN_p = MeanBars_plot(NN_rec, NN_abo, suptitle_ = 'Mean Nearest Neighbor Distances', xlabel_='Mean NN Distance ($\mu$m)')
	loc_ax, loc_t, loc_p = MeanBars_plot(loc_rec, loc_abo, suptitle_ = 'Mean Location on Dendrite', xlabel_='Mean Location ($\mu$m)')
	mid_ax, mid_t, mid_p = MeanBars_plot(dist_from_mid_rec, dist_from_mid_abo, suptitle_ = 'Mean Distance from Dendrite Center', xlabel_='Mean Distance ($\mu$m)')

	# Plot Normalized Histograms (for absolute numbers on histogram, change normed to False)
	def hist_plot(recovered_group, abolished_group, title_='', normed=True):
		fig, h_ax = plt.subplots()
		h_ax.hist(recovered_group, label='Recovered', color=rec_color, density=normed)
		h_ax.hist(abolished_group, label='Abolished', alpha=0.5, color=abo_color, density=normed);
		fig.suptitle('%s Histogram for %s. Synapses (%s)'%(title_, which_analyze, condition))
		h_ax.set_title('n_exc = %.f, branch length = %.f$\mu$m, synapses start at %.1f$\mu$m'%(n_exc, dend.L, start_syns),size=10)
		h_ax.set_ylabel('# sequences'); h_ax.set_xlabel('%s ($\mu$m)'%title_)
		h_ax.legend()
		
		return h_ax
	NN_hist_ax = hist_plot(NN_rec, NN_abo, title_='NN Distance ($\mu$m)') 
	loc_hist_ax = hist_plot(loc_rec, loc_abo, title_='Location ($\mu$m)')
	mid_hist_ax = hist_plot(dist_from_mid_rec, dist_from_mid_abo, title_='Distance ($\mu$m)')

	# CDF plots
	def CDF_plot(sample1, label1, sample2, label2, title_, h_ax=None, nbins=20):
		
		def GetDistributions(sample1, sample2, nbins):

			# For consistency between CDFs
			lower_limit = min(min(sample1), min(sample2))
			upper_limit = max(max(sample1), max(sample2))
			
			# Create objects (H1, H2) that includes the cumulative frequency and surrounding information
			H1 = stats.cumfreq(sample1, numbins=nbins, defaultreallimits=(lower_limit, upper_limit))
			H2 = stats.cumfreq(sample2, numbins=nbins, defaultreallimits=(lower_limit, upper_limit))
			
			cumdist1 = H1.cumcount 
			cumdist2 = H2.cumcount

			binsize = H1.binsize

			return lower_limit, upper_limit, H1, H2, cumdist1, cumdist2, binsize

		def ErrorMeasure(error_, sample):
			OK_ = 0
			control_std = np.std(sample)

			while not OK_:
				if error_ == 'STD':
					error_measure = control_std
					OK_ = 1
				elif error_ == 'SEM':
					error_measure = [i/np.sqrt(n_iters) for i in control_std]
					OK_ = 1
				else:
					error_ = input('Which error measure? (STD/SEM) ')

			return error_measure

		lower_limit, upper_limit, H1, H2, cumdist1, cumdist2, binsize = GetDistributions(sample1, sample2, nbins)
		B = lower_limit + np.linspace(0, binsize * len(cumdist1), len(cumdist1))

		cumdist1 = [i/max(cumdist1) for i in cumdist1]
		cumdist2 = [i/max(cumdist2) for i in cumdist2]

		if not h_ax:
			fig_, h_ax = plt.subplots()

		h_ax.plot(B, cumdist1, label=label1, color='green')
		h_ax.plot(B, cumdist2, label=label2, color='red')

		h_ax.set_title('%s'%title_, size=13)
		h_ax.set_ylabel('CDF', size=13) 
		h_ax.legend()

		return h_ax	
	
	title_ = 'Nearest Neighbor Distance CDF (normalized by sample size) \n n_exc = %.f, synapses start at %.1f'%(n_exc, start_syns)
	rec_label = 'Recovered (n = %i)'%len(NN_rec)
	abo_label = 'Abolished (n = %i)'%len(NN_abo)
	NN_cdf_ax = CDF_plot(NN_rec, rec_label, NN_abo, abo_label, title_)
	NN_cdf_ax.set_xlabel('NN Distance ($\mu$m)')

	title_ = 'Location CDF (normalized by sample size) \n n_exc = %.f, synapses start at %1.f'%(n_exc, start_syns)
	rec_label = 'Recovered (n = %i)'%len(loc_rec)
	abo_label = 'Abolished (n = %i)'%len(loc_abo)
	loc_cdf_ax = CDF_plot(loc_rec, rec_label, loc_abo, abo_label, title_)
	loc_cdf_ax.set_xlabel('Location ($\mu$m)')









