import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors
import obspy
from obspy.core.utcdatetime import UTCDateTime as UTC
import random
import datetime
import scipy
from scipy.optimize import minimize
from scipy import linalg
from scipy.special import factorial as fac
from geopy.distance import geodesic

import sys
sys.path.append('C:/Users/62813/OFANI-main/OFANI-main/Omori_Project/src/omori')

import src.seis_utils as seis
import src.omori_utils as omori

import emcee
import corner
import os

matplotlib.rcParams.update({'font.family': 'serif', 'font.serif': 'Times New Roman'})
matplotlib.rcParams.update({'font.size': 22})

# Directory to save files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'UPLOAD_OFANI')
save_dir = os.path.join(BASE_DIR, 'static/RESULTS_OFANI')

def calculate(filename):
    # Get the relative path of the filename with respect to UPLOAD_FOLDER
    rel_path = os.path.relpath(filename, UPLOAD_FOLDER)
    
    # Extract the base filename (without extension) from the relative path
    base_filename = os.path.splitext(os.path.basename(rel_path))[0]
    print(base_filename)
    
    # covered time period
    t_start = UTC('20200101')
    t_termi = UTC('20231231')
    cat = seis.read_cat_sep(filename)
    
    # %% Read and present dataset
    evid, obdt, pydt, relt, lat, lon, dep, mag = np.array(cat)
    
    # MS info
    id0 = obdt >= t_start
    ms_id = np.argmax(mag[id0])
    ms = {'id': evid[id0][ms_id], 'obdt': obdt[id0][ms_id], 'pydt': pydt[id0][ms_id], 'tAMS': relt[id0][ms_id],
          'lat': lat[id0][ms_id], 'lon': lon[id0][ms_id], 'dep': dep[id0][ms_id], 'mag': mag[id0][ms_id]}
    
    # save meta parameters
    meta = {'t_start': t_start,
            't_termi': t_termi,
            'Mcut': min(mag),
            'rmax': 10 ** (0.25 * ms['mag'] - .22),  # max radius of influence (Gardner & Knopoff, 1967)
            'nbin': 100,
            'c_bound': [1e-4, 2],
            'K_bound': [2, 1e4],
            'p_bound': [.1, 2],
            'c0': .5,
            'K0': 50,
            'p0': 1.1,
            'ylim': [1e-3, 1e5],
            'xlim': [1e-3, 1e3],
            'syn_c': 0.6,
            'syn_p': 1.3,
            'syn_tStart': 1e-2,
            'syn_tEnd': 1e3,
            'syn_N': 4000, }
    
    # get aftershocks within a radius
    aR = []
    for i in range(len(evid)):
        aR.append(geodesic((ms['lat'], ms['lon']), (lat[i], lon[i])).km)
    aR = np.array(aR)
    rd_id = aR <= meta['rmax']
    
    # selections
    mc_id = mag >= meta['Mcut']
    as_id = obdt > ms['obdt']
    end_id = obdt < t_termi
    rd_id = aR <= meta['rmax']
    select = mc_id * as_id * end_id * rd_id
    evid, obdt, pydt, relt, lat, lon, dep, mag = np.array(cat).T[select].T
    
    print('Mainshock magnitude: %.2f \n' % ms['mag'],
              'Mainshock time: %s \n' % ms['obdt'],
              'Minimum magnitude: %.2f \n' % meta['Mcut'],
              'Maximum radius: %.2f \n' % meta['rmax'],
              'Start time: %s \n' % meta['t_start'],
              'End time: %s \n' % meta['t_termi'],
              '# events selected: %d \n' % select.sum())
    
    relt = relt.astype('float')
    lat = lat.astype('float')
    lon = lon.astype('float')
    dep = dep.astype('float')
    mag = mag.astype('float')
    relt = relt - ms['tAMS']

    def dataatt(lon, lat, dep, mag):
        input_data = [lon, lat, dep, mag]
        
        for i, parameter in enumerate(["Longitude", "Latitude", "Depth", "Magnitude"]):
            data_stats = {
                "Parameter": parameter,
                "Minimum": np.min(input_data[i]),
                "Mean": np.mean(input_data[i]),
                "Median": np.median(input_data[i]),
                "Maximum": np.max(input_data[i])
            }
            
            print(f"Statistics for {parameter}:")
            for stat, value in data_stats.items():
                print(f"{stat}: {value}")
            print()
    
        return data_stats
    
    data1 = dataatt(lon, lat, dep, mag)
    data1

    #Plot 1: mapping of earthquake event
    size = [10, 6]  # [22, 10]
    plt.figure(figsize=size)
    # Plot the earthquakes, using depths as colormap
    sc = plt.scatter(lon, lat, c=dep, cmap="viridis", s=50, ec='k', lw=0.4, alpha=0.8)
    plt.scatter(ms['lon'], ms['lat'], ec='k', fc='red', s=300, marker='*')
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
    norm = mcolors.Normalize(vmin=np.min(dep), vmax=np.max(dep))
    cbar = plt.colorbar(sc, label="Earthquake Depth (m)", norm=norm)
    cbar.set_label(label="Earthquake Depth (m)")
    plt.title("Microseismic Map")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid()
    plt.savefig(f'{save_dir}/{base_filename}_Plot1.png', dpi=100)

    #Plot2
    size = [10, 6]  # [22,10]
    plt.figure(figsize=size)
    sc = plt.scatter(lat, pydt, c=dep, cmap='hot_r',ec='k', lw=0.4, s=100, alpha=0.8)
    plt.scatter(ms['lat'], ms['pydt'], s=600, ec='k', fc='gold', marker='*')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
    norm = mcolors.Normalize(vmin=np.min(dep), vmax=np.max(dep))
    cbar = plt.colorbar(sc, label="Depth (m)", norm=norm)
    cbar.set_label(label="Depth (m)")
    plt.xlabel('Latitude')
    plt.ylabel('Time')
    plt.title("Time vs Event Latitude")
    plt.grid()
    plt.tight_layout()
    # Save the figure
    filepath_to_save = os.path.join(save_dir, f"{base_filename}_Plot2.png")
    plt.savefig(filepath_to_save, dpi=100)

    #Plot3
    size = [10, 6]  # [22,10]
    plt.figure(figsize=size)
    sc = plt.scatter(lon, pydt, c=dep, cmap='hot_r',ec='k', lw=0.4, s=100, alpha=0.8)
    plt.scatter(ms['lon'], ms['pydt'], s=600, ec='k', fc='gold', marker='*')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
    norm = mcolors.Normalize(vmin=np.min(dep), vmax=np.max(dep))
    cbar = plt.colorbar(sc, label="Depth (m)", norm=norm)
    cbar.set_label(label="Depth (m)")
    plt.xlabel('Longitude')
    plt.ylabel('Time')
    plt.title("Time vs Event Longitude")
    plt.grid()
    plt.tight_layout()
    # Save the figure
    filepath_to_save = os.path.join(save_dir, f"{base_filename}_Plot3.png")
    plt.savefig(filepath_to_save, dpi=100)

    #Plot4
    plt.figure(figsize=size)
    sc = plt.scatter(pydt, mag, s=100, fc='lightgrey', ec='k', lw=0.4)
    plt.scatter(ms['pydt'], ms['mag'], s=600, ec='k', fc='gold', marker='*')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.title("Time vs Event Magnitude")
    plt.tight_layout()
    # Save the figure
    plt.savefig(f'{save_dir}/{base_filename}_Plot4.png', dpi=100)

    # %% Run minimizer
    # Choose dataset:
    data = '1'
    if data == '0':
        # Synthetic dataset
        c = meta['syn_c']
        p = meta['syn_p']
        synt = omori_syn(c, p, meta['syn_tStart'], meta['syn_tEnd'], meta['syn_N'])
        otimes = np.array(sorted(synt))*24
        
    elif data == '1':
        # Real dataset
        otimes = np.array(sorted(relt))*24
        
    # Calc likelihood:
    objFunc1 = lambda X: omori.ogata_logL(otimes, X)
    objFunc2 = lambda X: omori.bayes_logL(otimes, X)
        
    disp = 0
    method = 'SLSQP'
        
    # Ogata 1989: MLE
    ogata_fit = scipy.optimize.minimize(objFunc1, np.array([meta['c0'], meta['K0'], meta['p0']]), \
                                                bounds=np.array([meta['c_bound'], meta['K_bound'], meta['p_bound']]), \
                                                tol=1e-4, method=method, options={'disp': disp, 'maxiter': 500})
    print(ogata_fit)
        
    # Holschneider et al., 2012: Bayesian
    bayes_fit = scipy.optimize.minimize(objFunc2, np.array([meta['c0'], meta['p0']]), \
                                                bounds=np.array([meta['c_bound'], meta['p_bound']]), \
                                                tol=1e-4, method=method, options={'disp': disp, 'maxiter': 500})
    finalL, K = omori.bayes_getK(otimes, bayes_fit['x'])
    print('\n', bayes_fit)
    print('       K:', K)
        
    meta['Ogata_fit'] = list(ogata_fit['x'])
    meta['Bayes_fit'] = [bayes_fit['x'][0], K, bayes_fit['x'][1]]
        
    meta
        
    # %% Plot log-likelihood in a 2D parameter space
    Cs = np.linspace(meta['c_bound'][0], meta['c_bound'][1])
    Ps = np.linspace(meta['p_bound'][0], meta['p_bound'][1])
        
    L = np.zeros([len(Cs), len(Ps)])
    for i in range(len(Cs)):
        for j in range(len(Ps)):
            L[i, j] = objFunc2((Cs[i], Ps[j]))
    L = L.T
        
    plt.figure(figsize=[12, 8])
    im = plt.pcolormesh(Cs, Ps, L, cmap='jet_r', shading='auto')
    plt.colorbar(im, label='-log(Likelihood)')
    plt.contour(Cs, Ps, L, levels=100, colors='w')
    plt.scatter(meta['Bayes_fit'][0], meta['Bayes_fit'][2], marker='*', s=600, ec='k', fc='w')
    # plt.xscale('log')
    plt.xlabel('c-value')
    plt.ylabel('p-value')
    plt.title('c vs p Value')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{base_filename}_Plot5.png', dpi=100)

    # %% Plot fitting results and data
    plot_res = 'both'
    
    def dtimes(obdt):
        eqtimestamps = []
    
        # Iterate over indices and values using enumerate
        for i, eqt in enumerate(obdt):
            eqtime = UTC(eqt)
            eqtimestamp = eqtime.timestamp
            eqtimestamps.append(eqtimestamp)
        
        hams = (np.array(eqtimestamps) - eqtimestamps[0]) / 3600.0 #hours after main shock
        bins = np.arange(0, 25, 1)
        event_counts, bin_edges = np.histogram(hams, bins=bins)
        bin_loc = (bin_edges[1:] + bin_edges[:-1]) / 2
        return event_counts, bin_loc
    
    occ_dens, bin_loc = dtimes(obdt)
    #bins = np.logspace(np.log10(otimes[0]), np.log10(otimes[-1]), 24)
    #count, bine = np.histogram(otimes, bins=bins)
    #bin_loc = (bine[1:] + bine[:-1]) / 2
    #occ_dens = count / np.diff(bins)
    
    plt.figure(figsize=[10, 8])
    plt.bar(bin_loc, occ_dens, width=1, ec='k', fc='lightgrey', label='%d events' % len(otimes))
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim(meta['ylim'])
    # plt.xlim(meta['xlim'])
    plt.xlabel('Hours after mainshock')
    plt.ylabel('# events/hour')
    
    lgdstr1 = 'Ogata {c,K,p}={%.2f, %.2f, %.2f}'
    lgdstr2 = 'Holsch {c,K,p}={%.2f, %.2f, %.2f}'
    
    if plot_res == 'o':
        plt.plot(bin_loc, omori.omori(bin_loc, meta['Ogata_fit']), c='r', label=lgdstr1 % tuple(meta['Ogata_fit']))
    elif plot_res == 'h':
        plt.plot(bin_loc, omori.omori(bin_loc, meta['Bayes_fit']), c='b', labe=lgdstr2 % tuple(meta['Bayes_fit']))
    elif plot_res == 'both':
        plt.plot(bin_loc, omori.omori(bin_loc, meta['Ogata_fit']), c='r', label=lgdstr1 % tuple(meta['Ogata_fit']))
        plt.plot(bin_loc, omori.omori(bin_loc, meta['Bayes_fit']), c='b', label=lgdstr2 % tuple(meta['Bayes_fit']))
    
    plt.legend(loc='upper right')
    plt.title('Aftershocks decay')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{base_filename}_Plot6.png', dpi=100)

    # %% Now try to fit with MCMC!
    # define uniform rectangular prior
    def log_prior(theta):
        c, K, p = theta
        c1, c2 = meta['c_bound']
        K1, K2 = meta['K_bound']
        p1, p2 = meta['p_bound']
        if c1 < c < c2 and K1 < K < K2 and p1 < p < p2:
            return 0.0
        return -np.inf
    
    # define post prob = prior * likelihood
    def log_probability(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp - objFunc1(theta)
    
    # %% Run ensemble MCMC samplers
    
    # initialize walkers in gaussian ball around minimized result
    pos0 = meta['Ogata_fit']
    nwalkers = 32
    ndim = len(pos0)
    pos = pos0 + 1e-3 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    sampler.run_mcmc(pos, 5000, progress=True);
    
    # %% Check MCMC results
    
    # Check out the chains
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ['c', 'K', 'p']
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], 'k', alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel('# Step')
    fig.suptitle("MCMC Traces")
    plt.tight_layout()
    fig.savefig(f'{save_dir}/{base_filename}_Plot7.png', dpi=100)

    # check autocorrelation time and corner plot
    tau = sampler.get_autocorr_time()
    print(tau)
    
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    
    matplotlib.rcParams.update({'font.size': 13})
    fig = corner.corner(flat_samples, labels=labels,
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True,
                            title_kwargs={"fontsize": 16})
    matplotlib.rcParams.update({'font.size': 22})
    fig.suptitle("MCMC Corner Plot")
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{base_filename}_Plot8.png', dpi=100)

    # %% Plot data against MCMC ensemble fits
    plt.figure(figsize=[10, 6])
    plt.scatter(bin_loc, occ_dens, s=100, ec='k', fc='lightgrey', label='obs. %d events' % len(otimes))
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim(meta['ylim'])
    # plt.xlim(meta['xlim'])
    plt.xlabel('Hours after mainshock')
    plt.ylabel('# events / hour')
    
    sol_c = np.percentile(flat_samples[:, 0], [16, 50, 84])
    qc = np.diff(sol_c)
    sol_K = np.percentile(flat_samples[:, 1], [16, 50, 84])
    qK = np.diff(sol_K)
    sol_p = np.percentile(flat_samples[:, 2], [16, 50, 84])
    qp = np.diff(sol_p)
    
    inds = np.random.randint(len(flat_samples), size=200)
    for ind in inds:
        sample = flat_samples[ind]
        plt.plot(bin_loc, omori.omori(bin_loc, sample), 'C1', alpha=0.1)
    lgdstr1 = 'Best {c,K,p} =\n{%.2f, %.2f, %.2f}' % (sol_c[1], sol_K[1], sol_p[1])
    plt.plot(bin_loc, omori.omori(bin_loc, (sol_c[1], sol_K[1], sol_p[1])), c='k', lw=2, label=lgdstr1)
    plt.legend(loc='upper right')
    plt.title('MCMC Ensembles Fit')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{base_filename}_Plot9.png', dpi=100)

    #restatement of c, K, p, magnitudo mainshock parameters
    co, Ko, po = meta['Ogata_fit'] #Ogata method
    ch, Kh, ph = meta['Bayes_fit'] #Holsch method
    cmcmc, Kmcmc, pmcmc = sol_c[1], sol_K[1], sol_p[1] #Markov Chain Monte Carlo method
    magms = ms['mag']

    def reentry_prot (c, K, p, magms):
        Tmc = ((K*p*(np.sqrt((2*p+1)/(p+2))))**(1/(1+p))) - c #reccomendation reentry time based on Omori Utsu Law
        Tmcv = 10**(-0.08+0.31*magms)
        Rmin = 10**(1.22+0.25*magms) #Radius of best spherical radius
        Rssm = 10**(1.47+0.31*magms) #Sum of seismic moment radius
        Rseq = 10**(1.46+0.25*magms) #Sequence Radius
        return Tmc, Tmcv, Rmin, Rssm, Rseq
    
    reentry_prot_ogata = reentry_prot(co, Ko, po, magms)
    reentry_prot_holsch = reentry_prot(ch, Kh, ph, magms)
    reentry_prot_mcmc = reentry_prot(cmcmc, Kmcmc, pmcmc, magms)
    
    # Print the results
    print("Ogata Reentry Protocol:")
    print(f"Tmc (Omori Utsu Law): {reentry_prot_ogata[0]} Hours after Blasting")
    print(f"Tmc (Vallejos 2017): {reentry_prot_ogata[1]} Hours after Blasting")
    print(f"Rmin: {reentry_prot_ogata[2]} m")
    print(f"Rssm: {reentry_prot_ogata[3]} m")
    print(f"Rseq: {reentry_prot_ogata[4]} m")
    print("\n")
    
    print("Holsch Reentry Protocol:")
    print(f"Tmc (Omori Utsu Law): {reentry_prot_holsch[0]} Hours after Blasting")
    print(f"Tmc (Vallejos 2017): {reentry_prot_holsch[1]} Hours after Blasting")
    print(f"Rmin: {reentry_prot_holsch[2]}m")
    print(f"Rssm: {reentry_prot_holsch[3]} m")
    print(f"Rseq: {reentry_prot_holsch[4]} m")
    print("\n")
    
    print("MCMC Reentry Protocol:")
    print(f"Tmc (Omori Utsu Law): {reentry_prot_mcmc[0]} Hours after Blasting")
    print(f"Tmc (Vallejos 2017): {reentry_prot_mcmc[1]} Hours after Blasting")
    print(f"Rmin: {reentry_prot_mcmc[2]} m")
    print(f"Rssm: {reentry_prot_mcmc[3]} m")
    print(f"Rseq: {reentry_prot_mcmc[4]} m")

    # Extract the results for each protocol
    ogata_results = np.array(reentry_prot_ogata)
    holsch_results = np.array(reentry_prot_holsch)
    mcmc_results = np.array(reentry_prot_mcmc)
    
    # Extract Tmc and radius values
    Tmc_omori = np.array([ogata_results[0], holsch_results[0], mcmc_results[0]])
    Tmc_vallejos = np.array([ogata_results[1], holsch_results[1], mcmc_results[1]])
    Rmin = np.array([ogata_results[2], holsch_results[2], mcmc_results[2]])
    Rssm = np.array([ogata_results[3], holsch_results[3], mcmc_results[3]])
    Rseq = np.array([ogata_results[4], holsch_results[4], mcmc_results[4]])
    
    # Calculate maximum, minimum, and mean values
    Tmc_omori_max = np.max(Tmc_omori)
    Tmc_omori_min = np.min(Tmc_omori)
    Tmc_omori_mean = np.mean(Tmc_omori)

    plt.figure(figsize=[10, 8])
    plt.bar(bin_loc, occ_dens, width=1, ec='k', fc='lightgrey', label='%d events' % len(otimes))
    plt.axvline(x=Tmc_omori_mean, color='r', linestyle='--', label='Tmc (Omori Utsu Law)')
    plt.axvline(x=Tmc_vallejos[0], color='green', linestyle='--', label='Tmc (Vallejos)')
    plt.xlabel('Hours after mainshock')
    plt.ylabel('# events/hour')
    
    lgdstr1 = 'Ogata {c,K,p}={%.2f, %.2f, %.2f}'
    lgdstr2 = 'Holsch {c,K,p}={%.2f, %.2f, %.2f}'
    
    plt.plot(bin_loc, omori.omori(bin_loc, meta['Ogata_fit']), c='r', label=lgdstr1 % tuple(meta['Ogata_fit']))
    plt.plot(bin_loc, omori.omori(bin_loc, meta['Bayes_fit']), c='b', label=lgdstr2 % tuple(meta['Bayes_fit']))
    
    plt.legend(loc='upper right')
    plt.title('Aftershocks decay with Reentry Protocol')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{base_filename}_Plot10.png', dpi=100)

    # Print the results
    print("Tmc (Omori Utsu Law) - Max, Min, Mean:", Tmc_omori_max, Tmc_omori_min, Tmc_omori_mean, "Hours after Blasting")
    print("Tmc (Vallejos 2017):", Tmc_vallejos[0], "Hours after Blasting")
    print("Rmin :", Rmin[0], "m")
    print("Rssm :", Rssm[0], "m")
    print("Rseq :", Rseq[0], "m")

    return {
            'mainshock_magnitude': ms['mag'],
            'mainshock_time': ms['obdt'],
            'minimum_magnitude': meta['Mcut'],
            'maximum_radius': meta['rmax'],
            'start_time': meta['t_start'],
            'end_time': meta['t_termi'],
            'events_selected': select.sum(),
            'Tmc_mean': Tmc_omori_mean,
            'Tmc_max': Tmc_omori_max,
            'Tmc_min': Tmc_omori_min,
            'Rmin': Rmin[0],
            'Rssm': Rssm[0],
            'Rseq': Rseq[0]
        }