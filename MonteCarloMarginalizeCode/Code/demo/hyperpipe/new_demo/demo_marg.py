#! /usr/bin/env python
"""
Example marginalization (MARG) script for HyperPipe. 

Compute the marginal likelihood L_k = \prod_k w_k in a tetrahedron, where:
    w_k = p(x)*g_k(x) for given parameters p and data points g_k in the xy-plane
Calls on external prior code for factor examining height on z-axis
"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad
import argparse
import sys

parser = argparse.ArgumentParser()
#HyperPipe API-required arguments:
parser.add_argument("--fname",help="filename of event_*.dat data file [standard ILE output]")
parser.add_argument("--fname-output-samples",default="output-ILE-samples",help="output posterior samples (default output-ILE-samples -> output-ILE)")
parser.add_argument("--fname-output-integral",default="integral_result",help="output filename for integral result. Postfixes appended")
parser.add_argument("--using-eos", type=str, default=None, help="EOS data: either fit parameter list (should physically use lambda1, lambda2 information (but need not)) or filepath to EOS parameter data, starting with 'file:' (e.g., 'file:/home/mydir/myeos.dat').")
parser.add_argument("--using-eos-index", type=int, default=None, help="Index of EOS parameters in file.")
parser.add_argument("--n-events-to-analyze",default=1,type=int,help="Number of EOS realizations/indices to analyze in this instance of the script.")    
#Recommended for HyperPipe runs:
parser.add_argument("--chunk-save",action='store_true',help="Save all output lines to one file instead of 1 file per line.")
parser.add_argument("--save-all-files",action='store_true',help="If present, makes versions of 6 of the 7 files (no .xml) CIP creates (ignored if chunk-save is true)")
#Extra arguments for this script specifically:
parser.add_argument("--std-scale-factor",type=float,default=0.0001,help="Scale factor to adjust error provided by scipy.dblquad (usually very small)")
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
#Supplemental likelihood factors:
parser.add_argument("--supplementary-likelihood-factor-code", default=None,type=str,help="Import a module (in your pythonpath!) containing a supplementary factor for the likelihood.  Used to impose supplementary external priors of arbitrary complexity and external dependence. EXPERTS-ONLY")
parser.add_argument("--supplementary-likelihood-factor-function", default=None,type=str,help="With above option, specifies the specific function used as an external likelihood. EXPERTS ONLY")

opts = parser.parse_args()
    

#function for this specific example; contents of MARG scripts will vary by problem
def compute_product(m_obs,pop_norm):
    partial_sum = 0.0
    partial_var = 0.0
    for i in range(len(m_obs)):
        #distribution around "real" data point:
        g_k = multivariate_normal(mean=m_obs[i,:2], cov=np.diag([m_obs[i,3]**2,m_obs[i,4]**2]))
        
        #integrand is product of gaussians: p(m)*g_k(m)
        if supplemental_ln_likelihood:
            int_rv = lambda y, x: pop_norm.logpdf([x,y])+g_k.logpdf([x,y])+supplemental_ln_likelihood(x,y)
        else:
            int_rv = lambda y, x: pop_norm.logpdf([x,y])+g_k.logpdf([x,y])
        
        #initial integration range (rectangle)
        lxbd = m_obs[i][0] - 0.5 #left x bound
        rxbd = m_obs[i][0] + 0.5 #right x bound
        lybd = m_obs[i][1] - 0.5 #lower y bound
        tybd = m_obs[i][1] + 0.5 #upper y bound
        
        #truncate bounds to be within -0.6 < x0, x1 < 0.6 (square)
        if lxbd < -0.6: 
            lxbd = -0.6
        if rxbd > 0.6:
            rxbd = 0.6
        if lybd < -0.6:
            lybd = 0.6
        if tybd > 0.6:
            tybd = 0.6
        
        #integrate over rectangle:
        w_k, err = dblquad(int_rv, lxbd, rxbd, lybd, tybd)
        partial_sum += w_k #save log_likelihood
        partial_var += (err/w_k)**2 #correct error propagation
    
    if opts.verbose: print(" ",partial_sum, np.sqrt(partial_var))
    return partial_sum, np.sqrt(partial_var)


#Structured to match CIP's behavior; could pipe to save_CIP_output.py (same code)
def save_results(out_grid, header):
    '''
    CIP saves up to 7 files for each EOS line. HyperPipe only uses 1 of those files:
        MARG-$(macroid)-$(macroevent)+annotation.dat
    with format:
        # lnL sigma_lnL x0 x1 x2
    Any MARG script used in HyperPipe MUST, at minimum, create this output file with this format
    '''
    if opts.chunk_save:
        # remove invalid lines
        indx_ok = np.ones(len(out_grid),dtype=bool)
        indx_ok = np.logical_and(indx_ok,  np.logical_not(np.isnan(out_grid[:,0]))) #check nans (shouldn't happen)
        indx_ok = np.logical_and(indx_ok,  np.logical_not(np.isinf(out_grid[:,0]))) #check +/-inf (can happen)
        print('   Ignoring lines with lnL = -inf : {} '.format(len(out_grid)-np.sum(indx_ok)))
        out_grid = out_grid[indx_ok]
        
        var = out_grid[:,1]/out_grid[:,0] #mimics sqrt(line[1]**2)/res behavior for single line
        out_grid[:,1] = var
        
        #File (2/7): MARG-0-0+annotation.dat
        np.savetxt(opts.fname_output_integral+"+annotation.dat",out_grid,header=header[:-1]) #skip newline char in header
        print("Chunk file saved; length =",len(out_grid))
        return

    for i,line in enumerate(out_grid):
        #HyperPipe output filenames not formatted to support multiple lines, by default
        #need to manually update output filenames to be different (hence, prefer chunk_save)
        newout = ""
        try:
            last_digit = int(opts.fname_output_integral[-1]) #will work in hyperpipe
            newout = opts.fname_output_integral[:-1]+str(last_digit+i)
        except:
            if opts.using_eos_index:
                newout = opts.fname_output_integral+"-"+str(opts.using_eos_index+i)
            else: #can't help you, bud
                newout = opts.fname_output_integral+"-"+str(i)
        if opts.fname_output_integral == opts.fname_output_samples: #True for HyperPipe
            opts.fname_output_samples = newout
        opts.fname_output_integral = newout
        
        res = line[0]
        if res == -np.inf:
            print("Note: lnL = -inf detected; skipping filesaves for this line.")
            continue
        var = line[1]**2
        ln_integrand_value = res
        # Save result -- needed for odds ratios, etc.
        #   Warning: integral_result.dat uses *original* prior, before any reweighting
        if opts.save_all_files:
            #File (1/7): MARG-0-0.dat
            np.savetxt(opts.fname_output_integral+".dat", [ln_integrand_value])
        
        params_here = line[2:]
        annotation_header = header # this will/must be lnL sigma_lnL and then parameter names, which we want to preserve
        with open(opts.fname_output_integral+"+annotation.dat", 'w') as file_out:
            file_out.write("# " + annotation_header + "\n")
            file_out.write(" {} {} ".format(ln_integrand_value, np.sqrt(var)/res) + ' '.join(map(str,params_here)))
            #File (2/7): MARG-0-0+annotation.dat
               
        #CIP will save other files at this point; we don't need to do that here        
        print("All files saved for this line.")
        

#----------------------------------------------------------
if (not opts.fname) and opts.using_eos is None: #offline test run defaults
    print("--Warning: Test Mode: using preset files--")
    opts.fname="demo_data.txt" 
    opts.using_eos= "demo_initial_grid.txt"
    opts.using_eos_index = 0
    opts.n_events_to_analyze=100
    opts.verbose = False
    opts.chunk_save = True
    opts.save_all_files = False
    opts.supplementary_likelihood_factor_code = "demo_ext_prior"
    opts.supplementary_likelihood_factor_function = "likelihood_evaluation"

#Access data from event file:
mass_dat = np.genfromtxt(opts.fname,names=True) #will be demo_data.txt (renamed to event-0.net)
param_names = list(mass_dat.dtype.names)
dat_as_array = mass_dat.view((float, len(param_names)))

#Access parameter grid from file:
fname = opts.using_eos.replace('file:', '')
pop_dat = None
try:
    check_dat = np.genfromtxt(fname,names=True)[opts.using_eos_index] #test for index being out of range
    pop_dat = np.genfromtxt(fname,names=True)[opts.using_eos_index:opts.using_eos_index+opts.n_events_to_analyze] #should be 1 line if n_events=1
except Exception as e:
    print(" Fail: EOS index out of range:\n   ",e)
    sys.exit(0)
param_names = list(pop_dat.dtype.names)
pop_as_array = pop_dat.view((float, len(param_names)))
print("dat size: (",len(pop_as_array),len(pop_as_array[0]),")")
npts = len(pop_as_array) # = opts.n_pts_to_analyze (default 1)
print("Num lines to analyze:",npts)


#Set up external prior code, if desired. This setup is only useful for CIP specifically, 
#and is completely unnecessary here. It's included to give an idea of how it works in CIP.
supplemental_ln_likelihood = None
supplemental_init = None
# Supplemental likelihood factor. Must have identical call sequence to 'likelihood_function'. Called with identical raw inputs (including cosines/etc)
if opts.supplementary_likelihood_factor_code and opts.supplementary_likelihood_factor_function:
    print(" EXTERNAL SUPPLEMENTARY LIKELIHOOD FACTOR : {}.{} ".format(opts.supplementary_likelihood_factor_code,opts.supplementary_likelihood_factor_function))
    __import__(opts.supplementary_likelihood_factor_code) #demo_ext_prior.py
    external_likelihood_module = sys.modules[opts.supplementary_likelihood_factor_code] #demo_ext_prior
    supplemental_ln_likelihood = getattr(external_likelihood_module,opts.supplementary_likelihood_factor_function) #look for likelihood_evaluation()
    supplemental_init = getattr(external_likelihood_module, 'initialize_me') #find initialize_me()


#grid to store output for all parameter lines
dat_out = np.zeros((npts,len(pop_as_array[0]))) #effectively forcing a deep copy of pop_dat

#do integral for each parameter line
for n in np.arange(npts):
    line = pop_as_array[n,2:] #ignore lnL, sigma_lnL cols
    if opts.verbose: print("line",n,":",line)
    
    #example of how to handle failure modes:
    if line[2] < 0:
        print(" WARNING: negative weight; line",n+opts.using_eos_index,"is invalid.")
        dat_out[n][0] = -2e6 #return a large negative value to strongly downweight this EOS parameter set; do not set to -inf or exit with success
        dat_out[n][1] = 1. #arbitrary non-zero sigma (MUST be non-zero!)
    else:
        #math for the problem goes here:
        
        #2D Gaussian of parameters
        rv = multivariate_normal(mean=line[:2], cov=0.01*np.diag(np.ones(2)))
        
        #initialize external prior
        if supplemental_ln_likelihood:
            args_init = {'input_line' : line, 'param_names':param_names}  # pass the recordarray broken into parts, for convenience
            supplemental_init(**args_init) #run initialize_me('input_line'=dat_as_array, 'param_names'=param_names)
        
        dat_out[n][0], dat_out[n][1] = compute_product(dat_as_array,rv)
        dat_out[n][1] = dat_out[n][1]/opts.std_scale_factor #rescale to avoid small variances
    dat_out[n][2:] = line #fill in parameters for output grid (necessary)
if opts.verbose: print("Output test:",dat_out[0])

#save results
lineheader = ' '.join(map(str,param_names))+"\n" #to match CIP extracted header
save_results(dat_out,lineheader)

# =============================================================================
# #Scatterplot:
# import matplotlib.pyplot as plt
# #import matplotlib as mpl
# fig1 = plt.figure(figsize=(8,5),dpi=250) 
# ax = fig1.add_subplot(111)
# ax.scatter(dat_out[:,2],dat_out[:,3],c=dat_out[:,0],marker=".")#,cmap=mpl.cm.cool)
# ax.set_xlabel("$\mu_1$", size="11")
# ax.set_ylabel("$\mu_2$", size="11")
# ax.tick_params(axis='both', which='major', labelsize=10) 
# ax.axis('scaled')
# fig1.tight_layout()
# plt.show(block=False)
# 
# =============================================================================

