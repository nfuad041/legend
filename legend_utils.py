import json
import pandas as pd
import os
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pygama.math.histogram as pgh



DET_GEOM_DIR = '/global/cfs/cdirs/m2676/sims/prodenv/l200a/v1.0.0/inputs/hardware/detectors/germanium/diodes'

def get_mage_id_from_det_id(det_id, location_config_filename = '/global/cfs/cdirs/m2676/users/nfuad/LEGEND/SIMS/l200-p03-all-config.json', verbose=False):
    
    config_location = json.load(open(location_config_filename))
    mage_ids = []
    relevant_det_ids = []
    for item in config_location.keys():
        if config_location[item]['system']=='geds':
            string = config_location[item]['location']['string']
            position = config_location[item]['location']['position']
            id = '101'+str(string).zfill(2)+str(position).zfill(2)
            id = int(id)
            mage_ids.append(id)
            relevant_det_ids.append(item)
    
    ids_df = pd.DataFrame({'mage_id': mage_ids, 'det_id': relevant_det_ids})
    mage_id = ids_df[ids_df['det_id'] == det_id]['mage_id']
    if len(mage_id) == 0:
        return int(0)
    else:
        return int(mage_id.values[0])


def get_det_id_from_mage_id(mage_id, location_config_filename = '/global/cfs/cdirs/m2676/users/nfuad/LEGEND/SIMS/l200-p03-all-config.json', verbose=False):
        
    config_location = json.load(open(location_config_filename))
    mage_id = int(mage_id)
    for item in config_location.keys():
        if config_location[item]['system']=='geds':
            string = config_location[item]['location']['string']
            position = config_location[item]['location']['position']
            id = '101'+str(string).zfill(2)+str(position).zfill(2)
            id = int(id)
            if id == mage_id:
                return item
    return None


def get_det_geoms(det_geom_dir=DET_GEOM_DIR):
    det_geom_files = os.listdir(det_geom_dir)

    ids = []
    masses_in_g = []
    heights_in_mm = []
    radii_in_mm = []
    fwhms = []

    for file in det_geom_files:
        # if filename starts with 'V' or 'P' or 'B' or 'C'
        if file[0] in ['V', 'P', 'B', 'C']:

            with open(det_geom_dir+'/'+file) as json_file:
                f = json.load(json_file)
                id = f['name']
                mass_in_g = f['production']['mass_in_g']
                height_in_mm = f['geometry']['height_in_mm']
                radius_in_mm = f['geometry']['radius_in_mm']
                fwhm = f['characterization']['l200_site']['fwhm_in_keV']['co60fep']
                ids.append(id)
                masses_in_g.append(mass_in_g)
                heights_in_mm.append(height_in_mm)
                radii_in_mm.append(radius_in_mm)
                fwhms.append(fwhm)

    det_geom_df = pd.DataFrame({'det_id':ids, 'mass_in_g':masses_in_g, 'height_in_mm':heights_in_mm, 'radius_in_mm':radii_in_mm, 'fwhm_co60fep_keV':fwhms})

    mage_ids = [get_mage_id_from_det_id(det_id) for det_id in det_geom_df['det_id']]
    det_geom_df['mage_id'] = mage_ids 

    return det_geom_df


def get_det_type_map(config_filename='/global/cfs/cdirs/m2676/users/nfuad/LEGEND/SIMS/l200-p03-all-config.json'):

    config = json.load(open(config_filename))
    type_df = pd.DataFrame()
    for key in config.keys():    
        if config[key]['system']=='geds':
            string = config[key]['location']['string']
            position = config[key]['location']['position']
            type_df.loc[position-1, 'string '+str(string)] = key[0]
    # count_df = type_df.fillna('.')
    # start row numbers from 1
    type_df.index = np.arange(1, len(type_df)+1)
    return type_df


def attach_det_type_map(map, rounding=None):
    new_map = pd.DataFrame()
    type_map = get_det_type_map()
    for col in map.columns:
        for row in map.index:
            item = map.loc[row, col]
            # print(str(item))
            if ((str(item) != 'nan') and (str(item) != 'None') and (str(item) != '.')):
                    new_map.loc[row, col] = str(type_map.loc[row, col]) + ':' + str(item)   
            else:
                new_map.loc[row, col] = '.'
    return new_map


def get_mass_map(geom_df=None, config_filename='/global/cfs/cdirs/m2676/users/nfuad/LEGEND/SIMS/l200-p03-all-config.json'):
    if geom_df is None:
        geom_df = get_det_geoms()
    config = json.load(open(config_filename))
    map = pd.DataFrame()
    for key in config.keys():    
        if config[key]['system']=='geds':
            string = config[key]['location']['string']
            position = config[key]['location']['position']

            id = '101'+str(string).zfill(2)+str(position).zfill(2)
            id = int(id)

            mass_in_g = geom_df[geom_df['mage_id']==id]['mass_in_g'].values[0]
            # map.loc[position-1, 'string '+str(string)] = str(key)[0]+':'+str(mass_in_g)
            map.loc[position-1, 'string '+str(string)] = mass_in_g
    map = map.fillna(math.nan)
    # start row numbers from 1
    map.index = np.arange(1, len(map)+1)
    return map


def get_heights_map(geom_df=None, cumulative=True, config_filename='/global/cfs/cdirs/m2676/users/nfuad/LEGEND/SIMS/l200-p03-all-config.json'):
    if geom_df is None:
        geom_df = get_det_geoms()
    config = json.load(open(config_filename))
    map = pd.DataFrame()
    for key in config.keys():    
        if config[key]['system']=='geds':
            string = config[key]['location']['string']
            position = config[key]['location']['position']

            id = '101'+str(string).zfill(2)+str(position).zfill(2)
            id = int(id)

            values = geom_df[geom_df['mage_id']==id]['height_in_mm'].values[0]
            map.loc[position-1, 'string '+str(string)] = values
    map = map.fillna(math.nan)

    if cumulative:
        # add rows to previous rows
        for i in range(1, len(map)):
            for j in range(len(map.columns)):
                if map.iloc[i, j] != '.':
                    map.iloc[i, j] = map.iloc[i-1, j] + map.iloc[i, j]

    # start row numbers from 1
    map.index = np.arange(1, len(map)+1)
    return map

def get_radii_map(geom_df=None, cumulative=True, config_filename='/global/cfs/cdirs/m2676/users/nfuad/LEGEND/SIMS/l200-p03-all-config.json'):
    if geom_df is None:
        geom_df = get_det_geoms()
    config = json.load(open(config_filename))
    map = pd.DataFrame()
    for key in config.keys():    
        if config[key]['system']=='geds':
            string = config[key]['location']['string']
            position = config[key]['location']['position']

            id = '101'+str(string).zfill(2)+str(position).zfill(2)
            id = int(id)

            values = geom_df[geom_df['mage_id']==id]['radius_in_mm'].values[0]
            map.loc[position-1, 'string '+str(string)] = values
    map = map.fillna(math.nan)

    if cumulative:
        # add rows to previous rows
        for i in range(1, len(map)):
            for j in range(len(map.columns)):
                if map.iloc[i, j] != '.':
                    map.iloc[i, j] = map.iloc[i-1, j] + map.iloc[i, j]

    # start row numbers from 1
    map.index = np.arange(1, len(map)+1)
    return map


def convert_height_map_to_mage_ref_frame(height_map=None, pos_0101=357.4):
    if height_map is None:
        height_map = get_height_map(cumulative=True)
    offset = pos_0101 + height_map.iloc[0, 0]
    new_map = pd.DataFrame()
    for col in height_map.columns:
        for row in height_map.index:
            if height_map.loc[row, col] != '.':
                new_map.loc[row, col] = -1*height_map.loc[row, col] + offset
    return new_map


def convert_height_map_to_sis_ref_frame(height_map=None, mage_pos_0101=357.4, sis_pos_0101=8046.4):
    if height_map is None:
        height_map = get_height_map(cumulative=True)
    mage_map = convert_height_map_to_mage_ref_frame(height_map, pos_0101=mage_pos_0101)
    offset = sis_pos_0101 + mage_map.iloc[0, 0]
    new_map = pd.DataFrame()
    for col in mage_map.columns:
        for row in mage_map.index:
            if height_map.loc[row, col] != '.':
                new_map.loc[row, col] = -1*mage_map.loc[row, col] + offset
    return new_map


def get_count_map(df, count_param='mage_id', config_filename='/global/cfs/cdirs/m2676/users/nfuad/LEGEND/SIMS/l200-p03-all-config.json'):
    config = json.load(open(config_filename))
    count_map = pd.DataFrame()
    for key in config.keys():    
        if config[key]['system']=='geds':
            string = config[key]['location']['string']
            position = config[key]['location']['position']

            id = '101'+str(string).zfill(2)+str(position).zfill(2)
            id = int(id)

            count = len(df[df[count_param]==id])
            count_map.loc[position-1, 'string '+str(string)] = count
    count_map = count_map.fillna(math.nan)
    # make percentage each entry
    # count_map = count_map / count_map.sum().sum() * 100
    # count_map = count_map.round(2)
    # count_map = count_map.astype(str) + '%'  # add percentage sign
    # # make 'nan%' into '.'
    
    # count_map = count_map.replace('nan%', '.')
    # start row numbers from 1
    count_map.index = np.arange(1, len(count_map)+1)
    return count_map


def curve_fit_gauss_with_quad_bkg(x, n_bins=50, guess=None, plot=False, verbose=False):
    def gauss(x, A, mu, sigma):
        return A* np.exp(-(x-mu)**2 / (2*sigma**2))
    def bkg(x, p0, p1, p2):
        return p0 + p1*x + p2*x**2
    def gauss_with_quad_bkg(x,  A, mu, sigma, p0, p1, p2):  # bkg = p0 + p1*x + p2*x**2
        return gauss(x, A, mu, sigma) + bkg(x, p0, p1, p2)
    
    lw = 1
 
    hist, bins, vars = pgh.get_hist(x, bins=n_bins)
    if plot:
        pgh.plot_hist(hist, bins, vars, lw=lw, color='k')
    bins_mid = (bins[1:] + bins[:-1])/2
    binsize = (bins[1]-bins[0]) # MeV
    #print('binsize',binsize)

    # find where max of hist happens
    max_idx = np.argmax(hist)
    # print('max_idx',max_idx)
    if guess is None:
        guess = [len(x)/n_bins*10, bins[0]+binsize*max_idx, 0.003, 0, 0, 0]
    # print('guess',guess)

    # popt, pcov = curve_fit(gauss_with_quad_bkg, bins_mid, hist, sigma=np.sqrt(vars), p0=guess)
    popt, pcov = curve_fit(gauss_with_quad_bkg, bins_mid, hist, p0=guess)
    perr = np.sqrt(np.diag(pcov))
    #print(popt)
    #print(perr)
    if verbose:
        print('popt:', popt)
        print('perr:', perr)
    
    # plot gauss+quad_bkg fit
    x = np.linspace(bins[0], bins[-1], 1000)
    total_fit_y = gauss_with_quad_bkg(x, *popt)
    chi_squared = sum((hist - gauss_with_quad_bkg(bins_mid, *popt))**2 / bins_mid/vars)
    chi_squared_per_dof = chi_squared / (len(hist) - len(popt))
    chi_squared_per_dof = round(chi_squared_per_dof, 2)
    if plot:
        # plt.plot(x, total_fit_y, label=r'fit gauss+quad_bkg ($\frac{\chi^2}{dof}=$'+str(chi_squared_per_dof)+')', lw=lw)
        plt.plot(x, total_fit_y, label='fit gauss+quad_bkg', lw=lw)

    # subtract bkg to get signal
    bkg = bkg(x, *popt[3:])
    bkg_subtracted_y = total_fit_y - bkg
    if plot:
        plt.plot(x, bkg_subtracted_y, label='bkg subtracted', lw=lw)
        plt.legend()
        plt.grid(True)
    A = popt[0]
    A_err = perr[0] 

    sigma = popt[2]
    sigma_err = perr[2]
    area = A * sigma * np.sqrt(2*np.pi)/binsize
    area_err =  area * np.sqrt((A_err/A)**2 + (sigma_err/sigma)**2)

    return abs(area), abs(area_err), popt, perr

def get_fwhm_map_from_config(geom_df=None, config_filename='/global/cfs/cdirs/m2676/users/nfuad/LEGEND/SIMS/l200-p03-all-config.json'):
    if geom_df is None:
        geom_df = get_det_geoms()
    config = json.load(open(config_filename))
    map = pd.DataFrame()
    for key in config.keys():    
        if config[key]['system']=='geds':
            string = config[key]['location']['string']
            position = config[key]['location']['position']

            id = '101'+str(string).zfill(2)+str(position).zfill(2)
            id = int(id)

            fwhm_co60fep_keV = geom_df[geom_df['mage_id']==id]['fwhm_co60fep_keV'].values[0]
            # map.loc[position-1, 'string '+str(string)] = str(key)[0]+':'+str(mass_in_g)
            map.loc[position-1, 'string '+str(string)] = fwhm_co60fep_keV
    map = map.fillna(math.nan)
    # start row numbers from 1
    map.index = np.arange(1, len(map)+1)
    return map


