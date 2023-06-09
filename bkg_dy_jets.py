import numpy as np
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import glob
import math
from itertools import repeat
import mplhep as hep
import time
import matplotlib.pyplot as plt
import copy
import random as rand
import pandas as pd
from functools import reduce

from ROOT import TCanvas, gStyle, TH1F,TH2F, TLegend, TFile, RooRealVar, RooCBShape, gROOT, RooFit, RooAddPdf, RooDataHist, RooDataSet
from ROOT import RooArgList

from array import array
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-sl",
    "--slurm",
    dest="slurm_port",
    default=None,
    action="store",
    help="Slurm cluster port (if not specified, will create a local cluster)",
)

parser.add_argument(
    "-r",
    "--region",
    dest="region",
    default=None,
    action="store",
    help="chose region (bb or be)",
)


args = parser.parse_args()


parameters = {
"regions" : args.region
}

print("region is: ", f"{parameters['regions']}") 

node_ip = "128.211.148.61"
client = Client(f"{node_ip}:{args.slurm_port}")

print("connected to cluster")

load_fields = [
        "dilepton_mass",
        "r",
        "dilepton_mass_gen",
        "nbjets",
        "wgt_nominal",
        "bjet1_ll_dR",
    ]


# paths = "/depot/cms/users/kaur214/output/muchannel_2018_allCuts_debug_Zpeak/stage1_output/2018/dyInclusive50/*parquet"

# paths_data = "/depot/cms/users/kaur214/output/muchannel_2018_allCuts_debug_Zpeak/stage1_output/2018/data_*/*parquet"

# paths_ttbar = "/depot/cms/users/kaur214/output/muchannel_2018_allCuts_debug_Zpeak/stage1_output/2018/ttbar_lep_inclusive/*parquet"
# paths_ww = "/depot/cms/users/kaur214/output/muchannel_2018_allCuts_debug_Zpeak/stage1_output/2018/W*/*parquet"
# paths_zz = "/depot/cms/users/kaur214/output/muchannel_2018_allCuts_debug_Zpeak/stage1_output/2018/Z*/*parquet"

paths = "/depot/cms/users/yun79/Zprime-Dilepton/output/test_may_bjet2_incl4/stage1_output_emu/2018/dy*/*parquet"

paths_data = "/depot/cms/users/yun79/Zprime-Dilepton/output/test_may_bjet2_incl4/stage1_output_emu/2018/data_*/*parquet"

paths_ttbar = "/depot/cms/users/yun79/Zprime-Dilepton/output/test_may_bjet2_incl4/stage1_output_emu/2018/ttbar_lep_inclusive/*parquet"
paths_ww = "/depot/cms/users/yun79/Zprime-Dilepton/output/test_may_bjet2_incl4/stage1_output_emu/2018/W*/*parquet"
paths_zz = "/depot/cms/users/yun79/Zprime-Dilepton/output/test_may_bjet2_incl4/stage1_output_emu/2018/Z*/*parquet"

sig_files = glob.glob(paths)
# print(f"sig_files: {sig_files}")
df_temp = dd.read_parquet(sig_files)

data_files = glob.glob(paths_data)
df_data_temp = dd.read_parquet(data_files)

ttbar_files = glob.glob(paths_ttbar)
df_ttbar_temp = dd.read_parquet(ttbar_files)

ww_files = glob.glob(paths_ww)
df_ww_temp = dd.read_parquet(ww_files)

zz_files = glob.glob(paths_zz)
df_zz_temp = dd.read_parquet(zz_files)

df_dy   = df_temp[load_fields]

df_data = df_data_temp[load_fields]
df_ttbar = df_ttbar_temp[load_fields]
df_ww = df_ww_temp[load_fields]
df_zz = df_zz_temp[load_fields]

frames = [df_ttbar, df_ww, df_zz]

df_bkg = dd.concat(frames)

print("computation complete")

df_dy   = df_dy[(df_dy["r"]==f"{parameters['regions']}") & (df_dy["dilepton_mass"] > 60.) & (df_dy["dilepton_mass"] < 500.) & (df_dy["nbjets"] == 0)]

df_data   = df_data[(df_data["r"]==f"{parameters['regions']}") & (df_data["dilepton_mass"] > 60.) & (df_data["dilepton_mass"] < 500.)]

df_bkg   = df_bkg[(df_bkg["r"]==f"{parameters['regions']}") & (df_bkg["dilepton_mass"] > 60.) & (df_bkg["dilepton_mass"] < 500.)]


#df_dy   = df_dy[(df_dy["r"]==f"{parameters['regions']}") & (df_dy["dilepton_mass"] > 200.) & (df_dy["dilepton_mass_gen"] > 200) & (df_dy["nbjets"] == 0) & (df_dy["bjet1_mb1_dR"] == False) & (df_dy["bjet1_mb2_dR"] == False)]

massBinningMuMu = (
    [j for j in range(60, 500, 5)]
    + [500]
)


print("starting .. ")

dy_mass = df_dy["dilepton_mass"].compute().values
data_mass =  df_data["dilepton_mass"].compute().values 
bkg_mass =  df_bkg["dilepton_mass"].compute().values 


wgt_dy = df_dy["wgt_nominal"].compute().values
wgt_data = df_data["wgt_nominal"].compute().values
wgt_bkg = df_bkg["wgt_nominal"].compute().values

print("done complete")

h_dy = TH1F("h_dy", "h_dy", len(massBinningMuMu)-1, array('d', massBinningMuMu))
h_data = TH1F("h_data", "h_data", len(massBinningMuMu)-1, array('d', massBinningMuMu))
h_bkg = TH1F("h_bkg", "h_bkg", len(massBinningMuMu)-1, array('d', massBinningMuMu))

for i in range(len(dy_mass)):
    h_dy.Fill(dy_mass[i], wgt_dy[i])

for i in range(len(data_mass)):
    h_data.Fill(data_mass[i], wgt_data[i])

for i in range(len(bkg_mass)):
    h_bkg.Fill(bkg_mass[i], wgt_bkg[i])

file2 = TFile(f"dy_sys_{parameters['regions']}.root","RECREATE")
file2.cd()
h_dy.Write()
h_data.Write()
h_bkg.Write()
file2.Close()








