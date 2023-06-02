import numpy as np
import dask.dataframe as dd
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

from ROOT import TCanvas, gStyle, TH1F,TH2F, THStack, TLegend, TFile, TPaveStats, RooRealVar, RooCBShape, gROOT, RooFit, RooAddPdf, RooDataHist, RooDataSet
from ROOT import RooArgList

from array import array
import sys
import argparse

parser = argparse.ArgumentParser()


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

node_ip = "128.211.148.60"

print("connected to cluster")

load_fields = [
        "dilepton_mass",
        "r",
        "dilepton_mass_gen",
        "nbjets",
        "wgt_nominal",
        "pt_mu",
        "min_b1l_mass",
        "dataset",
        #"bjet1_mb1_dR",
        #"bjet1_mb2_dR",
    ]

dimuon_load_fields = [
    "dimuon_mass",
    "dimuon_mass_gen",
    "dataset",
    "wgt_nominal",
]

#paths = "/depot/cms/users/kaur214/output/muchannel_2018_allCuts/stage1_output/2018/dyInclusive50/*parquet"

#paths_data = "/depot/cms/users/kaur214/output/muchannel_2018_allCuts/stage1_output/2018/data_*/*parquet"

#paths_ttbar = "/depot/cms/users/kaur214/output/muchannel_2018_allCuts/stage1_output/2018/t*/*parquet"
#paths_ww = "/depot/cms/users/kaur214/output/muchannel_2018_allCuts/stage1_output/2018/W*/*parquet"
#paths_zz = "/depot/cms/users/kaur214/output/muchannel_2018_allCuts/stage1_output/2018/Z*/*parquet"

#paths_ttbar = "/depot/cms/users/kaur214/output/muchannel_2018_allCuts/stage1_output/2018/ttbar_*/*parquet"

dimu_paths_ttbar = "/depot/cms/users/kaur214/output/muchannel_2018_allCuts/stage1_output/2018/t*/*parquet"

# paths_ttbar = "/depot/cms/users/yun79/Zprime-Dilepton/output/test_test/stage1_output_emu/2018/ttbar_lep_inclusive/*parquet"
paths_ttbar = "/depot/cms/users/yun79/Zprime-Dilepton/output/test_test_el_pTcut53/stage1_output_emu/2018/ttbar_lep_inclusive/"
# paths_ttbar = "/depot/cms/users/yun79/Zprime-Dilepton/output/test_test_el_pTcut53_run2/stage1_output_emu/2018/ttbar_lep_inclusive/"


#sig_files = glob.glob(paths)
#df_temp = dd.read_parquet(sig_files)
#
#data_files = glob.glob(paths_data)
#df_data_temp = dd.read_parquet(data_files)

ttbar_files = glob.glob(paths_ttbar)
print(f"ttbar_files: {ttbar_files}")
df_ttbar_temp = dd.read_parquet(ttbar_files)
print("falg")
# df_ttbar_temp = pd.read_parquet(ttbar_files)
# print(f"df_ttbar_temp: \n {df_ttbar_temp.to_string()}")
print(f"df_ttbar_temp: \n {df_ttbar_temp}")
# print(f"df_ttbar_temp.columns: {df_ttbar_temp.columns}")

dimu_ttbar_files = glob.glob(dimu_paths_ttbar)
# print(f"ttbar_files: {dimu_ttbar_files}")
dimu_df_ttbar_temp = dd.read_parquet(dimu_ttbar_files)
# print(f"dimu_df_ttbar_temp.columns: {dimu_df_ttbar_temp.columns.astype(str)}")
print(f"dimu_df_ttbar_temp: \n {dimu_df_ttbar_temp.loc[:1,:].to_string}")


# raise ValueError

#ww_files = glob.glob(paths_ww)
#df_ww_temp = dd.read_parquet(ww_files)
#
#zz_files = glob.glob(paths_zz)
#df_zz_temp = dd.read_parquet(zz_files)
#
#df_dy   = df_temp[load_fields]
#
#df_data = df_data_temp[load_fields]
df_ttbar = df_ttbar_temp[load_fields]
#df_ww = df_ww_temp[load_fields]
#df_zz = df_zz_temp[load_fields]

dimu_df_ttbar = dimu_df_ttbar_temp[dimuon_load_fields]

#frames = [df_ttbar, df_ww, df_zz]

#df_bkg = dd.concat(frames)

print("computation complete")

#df_dy   = df_dy[(df_dy["r"]==f"{parameters['regions']}") & (df_dy["dilepton_mass"] > 60.) & (df_dy["dilepton_mass"] < 120.)]

#df_data   = df_data[(df_data["r"]==f"{parameters['regions']}") & (df_data["dilepton_mass"] > 60.) & (df_data["dilepton_mass"] < 120.)]

# df_ttbar   = df_ttbar[(df_ttbar["r"]==f"{parameters['regions']}") & (~((df_ttbar.dataset == "ttbar_lep_inclusive") & (df_ttbar.dilepton_mass_gen > 500)))]
#df_bkg   = df_bkg[(df_bkg["r"]==f"{parameters['regions']}") & (df_bkg["dilepton_mass"] > 60.) & (df_bkg["dilepton_mass"] < 120.)]

#df_dy   = df_dy[(df_dy["r"]==f"{parameters['regions']}") & (df_dy["dilepton_mass"] > 200.) & (df_dy["dilepton_mass_gen"] > 200) & (df_dy["nbjets"] == 0) & (df_dy["bjet1_mb1_dR"] == False) & (df_dy$

massBinningMuMu = (
    [j for j in range(0, 2000, 2)]
    + [2000]
)


print("starting .. ")
# print(f"df_ttbar: \n {df_ttbar.to_string()}")
#dy_mass = df_dy["dilepton_mass"].compute().values
#data_mass =  df_data["dilepton_mass"].compute().values
#bkg_mass =  df_bkg["dilepton_mass"].compute().values
bkg_mass =  df_ttbar["dilepton_mass"].compute().values
# bkg_mass =  df_ttbar["dilepton_mass"].values
dimu_bkg_mass =  dimu_df_ttbar["dimuon_mass"].compute().values

print(f"bkg_mass: {bkg_mass[:10]}")

#wgt_dy = df_dy["wgt_nominal"].compute().values
#wgt_data = df_data["wgt_nominal"].compute().values
#wgt_bkg = df_bkg["wgt_nominal"].compute().values
wgt_bkg = df_ttbar["wgt_nominal"].compute().values
# wgt_bkg = df_ttbar["wgt_nominal"].values
dimu_wgt_bkg = dimu_df_ttbar["wgt_nominal"].compute().values


print("done complete")
c1 = TCanvas("c1","Example",0,0,500,500)
#h_dy = TH1F("h_dy", "h_dy", len(massBinningMuMu)-1, array('d', massBinningMuMu))
#h_data = TH1F("h_data", "h_data", len(massBinningMuMu)-1, array('d', massBinningMuMu))
emu_bkg = TH1F("emu_bkg", "emu_bkg", len(massBinningMuMu)-1, array('d', massBinningMuMu))
emu_bkg.SetFillColor(416)
emu_bkg.SetLineColor(416)
emu_bkg.SetStats (0)
dimu_bkg = TH1F("dimu_bkg", "dimu_bkg", len(massBinningMuMu)-1, array('d', massBinningMuMu))
dimu_bkg.SetFillColor(632)
dimu_bkg.SetLineColor(632)
dimu_bkg.SetStats (0)
# histograms = THStack("hs","")
# histograms.Add(emu_bkg)
# histograms.Add(dimu_bkg)

#for i in range(len(dy_mass)):
#    h_dy.Fill(dy_mass[i], wgt_dy[i])
#
#for i in range(len(data_mass)):
#    h_data.Fill(data_mass[i], wgt_data[i])

print(f"len(bkg_mass): {len(bkg_mass)}")
for i in range(len(bkg_mass)):
    emu_bkg.Fill( bkg_mass[i], wgt_bkg[i])
for i in range(len(dimu_bkg_mass)):
    dimu_bkg.Fill( dimu_bkg_mass[i], dimu_wgt_bkg[i])

# st = TPaveStats(dimu_bkg.GetListOfFunctions().FindObject("stats"))
# st.SetName("test")
# st.SetY1NDC(12)
emu_bkg.Draw()
print(f"emu_bkg.GetMean(): {emu_bkg.GetMean()}")
print(f"emu_bkg.GetStdDev(): {emu_bkg.GetStdDev()}")
print(f"emu_bkg.GetEntries(): {emu_bkg.GetEntries()}")
dimu_bkg.Draw("same")


legend = TLegend(0.7 ,0.6 ,0.85 ,0.75)
legend.AddEntry( emu_bkg , f"emu processor. Entries: {emu_bkg.GetEntries()}. Mean: {emu_bkg.GetMean()}" )
legend.AddEntry( dimu_bkg ,f"dimuon processor.\n Entries: {dimu_bkg.GetEntries()}.\n Mean: {dimu_bkg.GetMean()}" )
legend.SetLineWidth(2)
legend.SetTextSize(5)
legend.Draw("same")
# dimu_bkg.Draw("sames")
# histograms.Draw("nostack")
# legend = TLegend(0.1,0.7,0.48,0.9)
# legend.AddEntry(emu_bkg, "emu","f")
# legend.AddEntry(dimu_bkg, "dimuon","l")
# legend.Draw()
# # make dimuon hist made by amandeep



c1.Update()
c1.Draw()

c1.Print("hist_test.pdf")

# file2 = TFile("ttbar_sys_BB.root","RECREATE")
# file2.cd()
#h_dy.Write()
#h_data.Write()
# emu_bkg.Write()
# file2.Close()