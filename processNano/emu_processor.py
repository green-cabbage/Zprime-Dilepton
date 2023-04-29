import sys

sys.path.append("copperhead/")

import awkward
import awkward as ak
import numpy as np

import pandas as pd
import coffea.processor as processor
from coffea.lumi_tools import LumiMask
from processNano.timer import Timer
from processNano.weights import Weights

# correction helpers included from copperhead
from copperhead.stage1.corrections.pu_reweight import pu_lookups, pu_evaluator
from copperhead.stage1.corrections.l1prefiring_weights import l1pf_weights

# from copperhead.stage1.corrections.lhe_weights import lhe_weights
# from copperhead.stage1.corrections.pdf_variations import add_pdf_variations

# high mass dilepton specific corrections
from processNano.corrections.kFac import kFac
from processNano.corrections.nnpdfWeight import NNPDFWeight
from copperhead.stage1.corrections.jec import jec_factories, apply_jec
from copperhead.config.jec_parameters import jec_parameters

# from processNano.jets import prepare_jets, fill_jets, fill_bjets, btagSF
from processNano.jets import prepare_jets, btagSF

import copy

from processNano.muons import find_dimuon, fill_muons
from processNano.utils import bbangle, delta_r
from processNano.emus import get_pair_inv_mass, fill_jets, fill_bjets

from config.parameters import parameters, muon_branches, ele_branches, jet_branches
from typing import TypeVar, List

pd_df = TypeVar("pandas_df")

def filter_df_cols(df: pd_df, cols_to_keep: List[str]) -> pd_df:
    """
    Take the given pandas df and keep the columns that contains ANY of the string elements
    in the cols_to_keep list
    """

    bool_filter = ~df.columns.str.contains("") # all false, since we are using or operators
    # print(f"bool_col: {bool_col}")
    for col in cols_to_keep:
        bool_filter = bool_filter | (df.columns.str.contains(col))
    return df.loc[:,bool_filter]

class EmuProcessor(processor.ProcessorABC):
    def __init__(self, **kwargs):
        self.samp_info = kwargs.pop("samp_info", None)
        do_timer = kwargs.pop("do_timer", True)
        self.apply_to_output = kwargs.pop("apply_to_output", None)
        self.pt_variations = kwargs.pop("pt_variations", ["nominal"])
 
        self.year = self.samp_info.year
        self.parameters = {k: v[self.year] for k, v in parameters.items()}

        self.do_btag = True

        if self.samp_info is None:
            print("Samples info missing!")
            return

        self._accumulator = processor.defaultdict_accumulator(int)

        self.applykFac = True
        self.applyNNPDFWeight = True
        self.do_pu = True
        self.auto_pu = False
        self.do_l1pw = True  # L1 prefiring weights
        self.do_jecunc = True
        self.do_jerunc = False

        self.timer = Timer("global") if do_timer else None

        self._columns = self.parameters["proc_columns"]

        self.regions = ["bb", "be"]
        self.channels = ["emu"]

        self.lumi_weights = self.samp_info.lumi_weights

        self.prepare_lookups()

    def process(self, df):
        # Initialize timer
        if self.timer:
            self.timer.update()

        # Dataset name (see definitions in config/datasets.py)
        dataset = df.metadata["dataset"]

        is_mc = True
        if "data" in dataset:
            is_mc = False

        # ------------------------------------------------------------#
        # Apply HLT, lumimask, genweights, PU weights
        # and L1 prefiring weights
        # ------------------------------------------------------------#

        numevents = len(df)

        # All variables that we want to save
        # will be collected into the 'output' dataframe
        output = pd.DataFrame(
            {"run": df.run, "event": df.event, "luminosityBlock": df.luminosityBlock}
        )
        output.index.name = "entry"
        output["npv"] = df.PV.npvs
        output["met"] = df.MET.pt

        # Separate dataframe to keep track on weights
        # and their systematic variations
        weights = Weights(output)
        muon_branches_local = copy.copy(muon_branches)
        ele_branches_local = copy.copy(ele_branches)

        # calculate generated mass from generated particles using the coffea genParticles
        print(f"is_mc: {is_mc}")
        if is_mc:
            genPart = df.GenPart
            print(f"genPart type: {type(genPart)}")
            print(f"genPart: {len(genPart)}")
            genPart = genPart[
                (
                    (abs(genPart.pdgId) == 11) | abs(genPart.pdgId)
                    == 13 | (abs(genPart.pdgId) == 15)
                )
                & genPart.hasFlags(["isHardProcess", "fromHardProcess", "isPrompt"])
            ]

            cut = ak.num(genPart) == 2
            output["emu_mass_gen"] = cut
            output["emu_pt_gen"] = cut
            output["emu_eta_gen"] = cut
            output["emu_phi_gen"] = cut
            genMother = genPart[cut][:, 0] + genPart[cut][:, 1]
            output.loc[
                output["emu_mass_gen"] == True, ["emu_mass_gen"]
            ] = genMother.mass
            output.loc[
                output["emu_pt_gen"] == True, ["emu_pt_gen"]
            ] = genMother.pt
            output.loc[
                output["emu_eta_gen"] == True, ["emu_eta_gen"]
            ] = genMother.eta
            output.loc[
                output["emu_phi_gen"] == True, ["emu_phi_gen"]
            ] = genMother.phi
            output.loc[output["emu_mass_gen"] == False, ["emu_mass_gen"]] = -999.0
            output.loc[output["emu_pt_gen"] == False, ["emu_pt_gen"]] = -999.0
            output.loc[output["emu_eta_gen"] == False, ["emu_eta_gen"]] = -999.0
            output.loc[output["emu_phi_gen"] == False, ["emu_phi_gen"]] = -999.0

        else:
            output["emu_mass_gen"] = -999.0
            output["emu_pt_gen"] = -999.0
            output["emu_eta_gen"] = -999.0
            output["emu_phi_gen"] = -999.0

        output["emu_mass_gen"] = output["emu_mass_gen"].astype(float)
        output["emu_pt_gen"] = output["emu_pt_gen"].astype(float)
        output["emu_eta_gen"] = output["emu_eta_gen"].astype(float)
        output["emu_phi_gen"] = output["emu_phi_gen"].astype(float)

        if is_mc:
            # For MC: Apply gen.weights, pileup weights, lumi weights,
            # L1 prefiring weights
            mask = np.ones(numevents, dtype=bool)
            genweight = df.genWeight
            weights.add_weight("genwgt", genweight)
            print(f"dataset: {dataset}")
            print(f"self.lumi_weights: {self.lumi_weights}")
            print(f"type(weights): {type(weights)}")
            print(f"self.parameters: {self.parameters}")
            weights.add_weight("lumi", self.lumi_weights[dataset])
            if self.do_pu:
                pu_wgts = pu_evaluator(
                    self.pu_lookups,
                    self.parameters,
                    numevents,
                    np.array(df.Pileup.nTrueInt),
                    self.auto_pu,
                )
                weights.add_weight("pu_wgt", pu_wgts, how="all")
            if self.do_l1pw:
                if "L1PreFiringWeight" in df.fields:
                    l1pfw = l1pf_weights(df)
                    weights.add_weight("l1prefiring_wgt", l1pfw, how="all")
                else:
                    weights.add_weight("l1prefiring_wgt", how="dummy_vars")
        

            muon_branches_local += [
                    "genPartFlav",
                    "genPartIdx",
                    "pt_gen",
                    "eta_gen",
                    "phi_gen",
                    "idx",
                ]
            ele_branches_local += ["genPartFlav", "pt_gen", "eta_gen", "phi_gen", "idx"]
        else:
            # For Data: apply Lumi mask
            lumi_info_mu = LumiMask(self.parameters["lumimask_UL_mu"])
            mask_mu = lumi_info_mu(df.run, df.luminosityBlock)
            lumi_info_el = LumiMask(self.parameters["lumimask_UL_el"])
            mask_el = lumi_info_el(df.run, df.luminosityBlock)
            mask = mask_mu and mask_el
        print(f"mask: {mask}")

        # Apply HLT to both Data and MC

        hlt = ak.to_pandas(df.HLT[self.parameters["mu_hlt"]+self.parameters["el_hlt"]])
        # print(f"df.HLT: {df.HLT}")
        # print(f'df.HLT[self.parameters["mu_hlt"]]: {df.HLT[self.parameters["mu_hlt"]]}')
        # print(f"hlt b4: {hlt}")
        # print(f'self.parameters["mu_hlt"]: {self.parameters["mu_hlt"]}')
        # print(f'self.parameters["mu_hlt"]+self.parameters["el_hlt"]: {self.parameters["mu_hlt"]+self.parameters["el_hlt"]}')
        hlt = hlt[self.parameters["mu_hlt"]+self.parameters["el_hlt"]].sum(axis=1)
        # print(f"hlt after: {hlt}")
        # print(f"len(hlt): {len(hlt)}")

        if self.timer:
            self.timer.add_checkpoint("Applied HLT and lumimask")

        # ------------------------------------------------------------#
        # Update muon kinematics with Rochester correction,
        # FSR recovery and GeoFit correction
        # Raw pT and eta are stored to be used in event selection
        # ------------------------------------------------------------#

        # Save raw variables before computing any corrections
        df["Muon", "pt_raw"] = df.Muon.pt
        df["Muon", "eta_raw"] = df.Muon.eta
        df["Muon", "phi_raw"] = df.Muon.phi
        df["Electron", "pt_raw"] = df.Electron.pt
        df["Electron", "eta_raw"] = df.Electron.eta
        df["Electron", "phi_raw"] = df.Electron.phi
        if is_mc:
            df["Muon", "pt_gen"] = df.Muon.matched_gen.pt
            df["Muon", "eta_gen"] = df.Muon.matched_gen.eta
            df["Muon", "phi_gen"] = df.Muon.matched_gen.phi
            df["Muon", "idx"] = df.Muon.genPartIdx
            df["Electron", "pt_gen"] = df.Electron.matched_gen.pt
            df["Electron", "eta_gen"] = df.Electron.matched_gen.eta
            df["Electron", "phi_gen"] = df.Electron.matched_gen.phi
            df["Electron", "idx"] = df.Electron.genPartIdx

            


        # for ...
        if True:  # indent reserved for loop over muon pT variations
            # According to HIG-19-006, these variations have negligible
            # effect on significance, but it's better to have them
            # implemented in the future

            # --- conversion from awkward to pandas --- #
            # muon selection    
            muons = ak.to_pandas(df.Muon[muon_branches_local])
            if self.timer:
                self.timer.add_checkpoint("load muon data")
            muons = muons.dropna()
            muons = muons.loc[:, ~muons.columns.duplicated()]

            # --------------------------------------------------------#
            # Select muons that pass pT, eta, isolation cuts,
            # muon ID and quality flags
            # passing quality cuts and at least one good PV
            # NOTE: do DO NOT count n muons or apply OS muon cut
            # --------------------------------------------------------#
            
            # Apply event quality flag
            output["r"] = None
            output["dataset"] = dataset
            output["year"] = int(self.year)
            if dataset == "dyInclusive50":
                muons = muons[muons.genPartFlav == 15]
            flags = ak.to_pandas(df.Flag)
            flags = flags[self.parameters["event_flags"]].product(axis=1)
            muons["pass_flags"] = True
            if self.parameters["muon_flags"]:
                muons["pass_flags"] = muons[self.parameters["muon_flags"]].product(
                    axis=1
                )
            # Define baseline muon selection (applied to pandas DF!)
            print("flag")
            muons["selection"] = (
                (muons.pt_raw > self.parameters["muon_pt_cut"])
                & (abs(muons.eta_raw) < self.parameters["muon_eta_cut"])
                & (muons.tkRelIso < self.parameters["muon_iso_cut"])
                & (muons[self.parameters["muon_id"]] > 0)
                & (muons.dxy < self.parameters["muon_dxy"])
                & (
                    (muons.ptErr.values / muons.pt.values)
                    < self.parameters["muon_ptErr/pt"]
                )
                & muons.pass_flags
            )
            print(f"muons: {muons.to_string()}")
            # print(f"muons['entry']: {muons['entry']}")
            # print(f"muons['subentry']: {muons['subentry']}")

            # # Find events with at least one good primary vertex
            # good_pv = ak.to_pandas(df.PV).npvsGood > 0
            # # Define baseline event selection

            # output["mu_event_selection"] = (
            #     mask
            #     & (hlt > 0)
            #     & (flags > 0)
            #     & good_pv
            # )
            # n_entries = muons.reset_index().groupby("entry")["subentry"].max()
            # print(f"muons: \n {muons.to_string()}")
            # # amandeep_sort =muons.sort_values(by="pt")
            # # print(50*"-")
            # # print(f"amandeep_sort: \n {amandeep_sort.to_string()}")
            # print(50*"-")
            # test = muons.reset_index().sort_values(by=["entry","pt"])
            # print(f"test : \n {test.to_string()}")
            # # now remove the rows with same entries but with lower pt vals
            # # this is done by droping duplicates of entries column, but 
            # # keeping the last row, which is sorted to have the highest pt
            # print(50*"-")
            # drop_test = test.drop_duplicates(subset=['entry'], keep='last')
            # print(f"drop_test : \n {drop_test.to_string()}")
            # print(50*"-")
            # print(f"n_entries: \n {len(n_entries)}")
            # cols_to_group = muons.reset_index().columns.values.tolist()
            # cols_to_group.remove("pt")
            # print(f"cols_to_group: {cols_to_group}")
            # print(f"muons.reset_index().groupby(cols_to_group): {muons.reset_index().groupby(cols_to_group)}")
            # mupt = muons.reset_index().groupby(["entry"])["pt"].max().reset_index()
            # print(50*"-")
            # print(f"mupt: \n {mupt.to_string()}")
            # mupt = mupt.set_index("entry").sort_index()
            # print(50*"-")
            # print(f"new mupt: \n {mupt.to_string()}")
            # print(f"new mupt length: \n {len(mupt)}")
            
            # nmuons = (
            #     muons[muons.selection]
            #     .reset_index()
            #     .groupby("entry")["subentry"]
            #     .nunique()
            # )
            # print(50*"-")
            # print(f"nmuons: \n {nmuons.to_string()}")

            # pick particle on each entry with the highest pt val
            print(f"muons b4: \n {muons.to_string()}")
            muons = muons.reset_index().sort_values(by=["entry","pt"])
            print(f"muons after sort: \n {muons.to_string()}")
            # now remove the rows with same entries but with lower pt vals
            # this is done by droping duplicates of entries column, but 
            # keeping the last row, which is sorted to have the highest pt
            print(50*"-")
            muons = muons.drop_duplicates(subset=['entry'], keep='last').set_index("entry")
            # muons.drop(columns=["subentry"], inplace=True)
            print(f"muons after drop: \n {muons.to_string()}")

            # --------------------------------------------------------#
            # Electron selection
            # --------------------------------------------------------#
            electrons = ak.to_pandas(df.Electron[ele_branches_local])
            electrons.pt = electrons.pt_raw * (electrons.scEtOverPt + 1.0)
            electrons.eta = electrons.eta_raw + electrons.deltaEtaSC
            electrons = electrons.dropna()
            electrons = electrons.loc[:, ~electrons.columns.duplicated()]
            if is_mc:
                electrons.loc[electrons.idx == -1, "pt_gen"] = -999.0
                electrons.loc[electrons.idx == -1, "eta_gen"] = -999.0
                electrons.loc[electrons.idx == -1, "phi_gen"] = -999.0

            print("flag2")
            # Apply event quality flag
            flags = ak.to_pandas(df.Flag)
            flags = flags[self.parameters["event_flags"]].product(axis=1)

            # Define baseline muon selection (applied to pandas DF!)
            electrons["selection"] = (
                (electrons.pt > self.parameters["electron_pt_cut"])
                & (abs(electrons.eta) < self.parameters["electron_eta_cut"])
                & (electrons[self.parameters["electron_id"]] > 0)
            )
            print("flag3")
            if dataset == "dyInclusive50":
                electrons = electrons[electrons.genPartFlav == 15]
            # # Count electrons
            # nelectrons = (
            #     electrons[electrons.selection]
            #     .reset_index()
            #     .groupby("entry")["subentry"]
            #     .nunique()
            # )
            # if is_mc:
            #     output["el_event_selection"] = mask & (hlt > 0) & (nelectrons >= 2)
            # else:
            #     output["el_event_selection"] = mask & (hlt > 0) & (nelectrons >= 4)
            print("flag4")
            # pick electorn particle on each entry with the highest pt val
            print(f"electrons b4: \n {electrons.to_string()}")
            electrons = electrons.reset_index().sort_values(by=["entry","pt"])
            print(f"electrons after sort: \n {electrons.to_string()}")
            # now remove the rows with same entries but with lower pt vals
            # this is done by droping duplicates of entries column, but 
            # keeping the last row, which is sorted to have the highest pt
            print(50*"-")
            electrons = electrons.drop_duplicates(subset=['entry'], keep='last').set_index("entry")
            # electrons.drop(columns=["subentry"], inplace=True)
            print(f"electrons after drop: \n {electrons.to_string()}")
            
            



            # Now join muons and electrons as one df


            leptons = muons.join(electrons, how="outer", lsuffix='_mu', rsuffix='_el')
            print(f"leptons: {leptons.to_string()}")
            leptons.dropna(inplace=True) # drop na since both an electron and muon has to exist
            print(f"leptons after dropna: {leptons.to_string()}")

            # drop non opposite charge pairs
            leptons["charge cut"] = leptons["charge_mu"]*leptons["charge_el"] < 0 # if opposite charge, the product of two charges should be negative
            print(f'leptons["charge cut"] : \n {leptons["charge cut"].to_string()}')
            leptons = leptons[leptons["charge cut"] ] # drop non opposite charge pairs
            print(f"leptons after non OS charge drop: \n {leptons.to_string()}")
            leptons.drop(columns=["charge cut"], inplace=True) # don't need it anymore
            print(f"leptons after drop charge cut columns: \n {leptons.to_string()}")
            deta, dphi, dr = delta_r(leptons["eta_mu"], leptons["eta_el"], leptons["phi_mu"],leptons["phi_el"])
            print(f"delta_r dr: \n {dr}")
            leptons["dR cut"] = dr > 0.4
            print(f'leptons["dR cut"] : \n {leptons["dR cut"].to_string()}')
            leptons = leptons[leptons["dR cut"] ]
            leptons.drop(columns=["dR cut"], inplace=True) # don't need it anymore
            print(f"leptons after dR cut: \n {leptons.to_string()}")
            pair_inv_mass = get_pair_inv_mass(
                leptons["mass_mu"],
                leptons["mass_el"],
                leptons["pt_mu"],
                leptons["pt_el"],
                leptons["eta_mu"],
                leptons["eta_el"],
                leptons["phi_mu"],
                leptons["phi_el"]
            )
            print(f"pair inv mass: \n {pair_inv_mass}")
            leptons["pair inv mass"] = pair_inv_mass
            # filter out unncessary columns
            print(f"leptons b4 dropping unncessary columns: \n {leptons.to_string()}")
            cols_to_keep = ["mass", "pt", "eta","phi"]
            leptons = filter_df_cols(leptons, cols_to_keep)
            print(f"leptons after dropping unncessary columns: \n {leptons.to_string()}")

            # Selection complete
            if self.timer:
                self.timer.add_checkpoint("Selected events and electrons")
            
            

        # ------------------------------------------------------------#
        # Prepare jets
        # ------------------------------------------------------------#
        prepare_jets(df, is_mc)

        # ------------------------------------------------------------#
        # Apply JEC, get JEC and JER variations
        # ------------------------------------------------------------#
        print("jet selection starting")
        jets = df.Jet
       
        output.columns = pd.MultiIndex.from_product(
            [output.columns, [""]], names=["Variable", "Variation"]
        )

        if self.timer:
            self.timer.add_checkpoint("Jet preparation & event weights")
        print("jet selection starting2")
        print(f"output.columns: {output.columns}")
        # print(f"self.pt_variations: {self.pt_variations}")
        print(f"df: {df}")
        print(f"dataset: {dataset}")
        # print(f"mu2: {mu2}")
        # print(f"mu1: {mu1}")
        print(f"jets: {jets}")
        print(f"jets: {jets[0][0]}")
        print(f"mask: {mask}")
        print(f"weights: {weights}")
        print(f"numevents: {numevents}")
        print(f"jet_branches: {jet_branches}")
        print(f"output.index: {output.index}")
        for v_name in self.pt_variations:
            # output_updated = self.jet_loop(
            #     v_name,
            #     is_mc,
            #     df,
            #     dataset,
            #     mask,
            #     leptons,
            #     # mu1,
            #     # mu2,
            #     jets,
            #     jet_branches,
            #     weights,
            #     numevents,
            #     output.index
            #     # output,
            # )
            # if output_updated is not None:
            #     output = output_updated
            bjets_df, jets_df = self.jet_loop(
                v_name,
                is_mc,
                df,
                dataset,
                mask,
                leptons,
                # mu1,
                # mu2,
                jets,
                jet_branches,
                weights,
                numevents,
                output.index
                # output,
            )

        if self.timer:
            self.timer.add_checkpoint("Computed event weights")
        print(f"leptons b4 adding bjets and jets: \n {leptons.to_string()}")
        leptons["rows to keep"] = True 
        leptons = leptons.join(bjets_df, how="outer", lsuffix='', rsuffix='_bjets') 
        leptons = leptons.join(jets_df, how="outer", lsuffix='', rsuffix='_jets') 
        leptons["rows to keep"].fillna(False, inplace=True)
        print(leptons["rows to keep"])
        leptons = leptons[leptons["rows to keep"]] # only keep rows with lepton values
        # leptons.dropna(inplace=True)
        leptons.fillna(0, inplace=True) # this is to fill NaN values from bjet and jet dfs
        leptons.drop(columns=["rows to keep"], inplace=True) # don't need it anymore
        print(f"leptons after adding bjets and jets: \n {leptons.to_string()}")

        print("jet selection flag 1")
        # ------------------------------------------------------------#
        # Fill outputs
        # ------------------------------------------------------------#

        # output.loc[
        #     ((abs(output.mu1_eta) < 1.2) & (abs(output.mu2_eta) < 1.2)), "r"
        # ] = "bb"
        # output.loc[
        #     ((abs(output.mu1_eta) > 1.2) | (abs(output.mu2_eta) > 1.2)), "r"
        # ] = "be"

        output["year"] = int(self.year)

        for wgt in weights.df.columns:
            if wgt == "pu_wgt_off":
                output["pu_wgt"] = weights.get_weight(wgt)
            if wgt != "nominal":
                output[f"wgt_{wgt}"] = weights.get_weight(wgt)


        if is_mc and "dy" in dataset and self.applykFac:
            mass_bb = output[output["r"] == "bb"].dimuon_mass_gen.to_numpy()
            mass_be = output[output["r"] == "be"].dimuon_mass_gen.to_numpy()
            for key in output.columns:
                if "wgt" not in key[0]:
                    continue
                output.loc[
                    ((abs(output.mu1_eta) < 1.2) & (abs(output.mu2_eta) < 1.2)),
                    key[0],
                ] = (
                    output.loc[
                        ((abs(output.mu1_eta) < 1.2) & (abs(output.mu2_eta) < 1.2)),
                        key[0],
                    ]
                    * kFac(mass_bb, "bb", "mu")
                ).values
                output.loc[
                    ((abs(output.mu1_eta) > 1.2) | (abs(output.mu2_eta) > 1.2)),
                    key[0],
                ] = (
                    output.loc[
                        ((abs(output.mu1_eta) > 1.2) | (abs(output.mu2_eta) > 1.2)),
                        key[0],
                    ]
                    * kFac(mass_be, "be", "mu")
                ).values

        if is_mc and "dy" in dataset and self.applyNNPDFWeight:
            mass_bb = output[output["r"] == "bb"].dimuon_mass_gen.to_numpy()
            mass_be = output[output["r"] == "be"].dimuon_mass_gen.to_numpy()
            leadingPt_bb = output[output["r"] == "bb"].mu1_pt_gen.to_numpy()
            leadingPt_be = output[output["r"] == "be"].mu1_pt_gen.to_numpy()
            for key in output.columns:
                if "wgt" not in key[0]:
                    continue
                output.loc[
                    ((abs(output.mu1_eta) < 1.2) & (abs(output.mu2_eta) < 1.2)),
                    key[0],
                ] = (
                    output.loc[
                        ((abs(output.mu1_eta) < 1.2) & (abs(output.mu2_eta) < 1.2)),
                        key[0],
                    ]
                    * NNPDFWeight(
                        mass_bb, leadingPt_bb, "bb", "mu", float(self.year), DY=True
                    )
                ).values
                output.loc[
                    ((abs(output.mu1_eta) > 1.2) | (abs(output.mu2_eta) > 1.2)),
                    key[0],
                ] = (
                    output.loc[
                        ((abs(output.mu1_eta) > 1.2) | (abs(output.mu2_eta) > 1.2)),
                        key[0],
                    ]
                    * NNPDFWeight(
                        mass_be, leadingPt_be, "be", "mu", float(self.year), DY=True
                    )
                ).values
        if is_mc and "ttbar" in dataset and self.applyNNPDFWeight:
            mass_bb = output[output["r"] == "bb"].dimuon_mass_gen.to_numpy()
            mass_be = output[output["r"] == "be"].dimuon_mass_gen.to_numpy()
            leadingPt_bb = output[output["r"] == "bb"].mu1_pt_gen.to_numpy()
            leadingPt_be = output[output["r"] == "be"].mu1_pt_gen.to_numpy()
            for key in output.columns:
                if "wgt" not in key[0]:
                    continue
                output.loc[
                    ((abs(output.mu1_eta) < 1.2) & (abs(output.mu2_eta) < 1.2)),
                    key[0],
                ] = (
                    output.loc[
                        ((abs(output.mu1_eta) < 1.2) & (abs(output.mu2_eta) < 1.2)),
                        key[0],
                    ]
                    * NNPDFWeight(
                        mass_bb, leadingPt_bb, "bb", "mu", float(self.year), DY=False
                    )
                ).values
                output.loc[
                    ((abs(output.mu1_eta) > 1.2) | (abs(output.mu2_eta) > 1.2)),
                    key[0],
                ] = (
                    output.loc[
                        ((abs(output.mu1_eta) > 1.2) | (abs(output.mu2_eta) > 1.2)),
                        key[0],
                    ]
                    * NNPDFWeight(
                        mass_be, leadingPt_be, "be", "mu", float(self.year), DY=False
                    )
                ).values

        output = output.loc[output.event_selection, :]
        output = output.reindex(sorted(output.columns), axis=1)
        output = output[output.r.isin(self.regions)]
        output.columns = output.columns.droplevel("Variation")

        print("jet selection flag 2")
        if self.timer:
            self.timer.add_checkpoint("Filled outputs")
            self.timer.summary()

        if self.apply_to_output is None:
            return output
        else:
            self.apply_to_output(output)
            return self.accumulator.identity()

    def jet_loop(
        self,
        variation,
        is_mc,
        df,
        dataset,
        mask,
        leptons,
        # mu1,
        # mu2,
        jets,
        jet_branches,
        weights,
        numevents,
        # output,
        indices
    ):
        print("jet loop flag")

        if not is_mc and variation != "nominal":
            return

        # variables = pd.DataFrame(index=output.index)
        variables = pd.DataFrame(index= indices)
        jet_branches_local = copy.copy(jet_branches)
        print("jet loop flag2")
        if is_mc:
            jets["pt_gen"] = jets.matched_gen.pt
            jets["eta_gen"] = jets.matched_gen.eta
            jets["phi_gen"] = jets.matched_gen.phi

            jet_branches_local += [
                "partonFlavour",
                "hadronFlavour",
                "pt_gen",
                "eta_gen",
                "phi_gen",
            ]

        # if variation == "nominal":
        #    if self.do_jec:
        #        jet_branches_local += ["pt_jec", "mass_jec"]
        #    if is_mc and self.do_jerunc:
        #        jet_branches_local += ["pt_orig", "mass_orig"]

        # Find jets that have selected muons within dR<0.4 from them
        # matched_mu_pt = jets.matched_muons.pt
        # matched_mu_iso = jets.matched_muons.pfRelIso04_all
        # matched_mu_id = jets.matched_muons[self.parameters["muon_id"]]
        # matched_mu_pass = (
        #     (matched_mu_pt > self.parameters["muon_pt_cut"])
        #     & (matched_mu_iso < self.parameters["muon_iso_cut"])
        #     & matched_mu_id
        # )
        # clean = ~(
        #     ak.to_pandas(matched_mu_pass)
        #     .astype(float)
        #     .fillna(0.0)
        #     .groupby(level=[0, 1])
        #     .sum()
        #     .astype(bool)
        # )
        print("jet loop flag3")
        if self.timer:
            self.timer.add_checkpoint("Clean jets from matched muons")

        # Select particular JEC variation
        # if "_up" in variation:
        #    unc_name = "JES_" + variation.replace("_up", "")
        #    if unc_name not in jets.fields:
        #        return
        #    jets = jets[unc_name]["up"][jet_branches_local]
        # elif "_down" in variation:
        #    unc_name = "JES_" + variation.replace("_down", "")
        #    if unc_name not in jets.fields:
        #        return
        #    jets = jets[unc_name]["down"][jet_branches_local]
        # else:

        jets = jets[jet_branches_local]

        # --- conversion from awkward to pandas --- #
        jets = ak.to_pandas(jets)
        print(f"jets: \n {jets.to_string()}")

        if jets.index.nlevels == 3:
            # sometimes there are duplicates?
            jets = jets.loc[pd.IndexSlice[:, :, 0], :]
            jets.index = jets.index.droplevel("subsubentry")

        print("jet loop flag4")
        # ------------------------------------------------------------#
        # Apply jetID
        # ------------------------------------------------------------#
        # Sort jets by pT and reset their numbering in an event
        # jets = jets.sort_values(["entry", "pt"], ascending=[True, False])
        jets.index = pd.MultiIndex.from_arrays(
            [jets.index.get_level_values(0), jets.groupby(level=0).cumcount()],
            names=["entry", "subentry"],
        )

        print("jet loop flag5")

        jets = jets.dropna()
        jets = jets.loc[:, ~jets.columns.duplicated()]

        if self.do_btag:
            if is_mc:
                btagSF(jets, self.year, correction="shape", is_UL=True)
                btagSF(jets, self.year, correction="wp", is_UL=True)

                variables["wgt_nominal"] = (
                    jets.loc[jets.pre_selection == 1, "btag_sf_wp"]
                    .groupby("entry")
                    .prod()
                )
                variables["wgt_nominal"] = variables["wgt_nominal"].fillna(1.0)
                variables["wgt_nominal"] = variables[
                    "wgt_nominal"
                ] * weights.get_weight("nominal")
                variables["wgt_btag_up"] = (
                    jets.loc[jets.pre_selection == 1, "btag_sf_wp_up"]
                    .groupby("entry")
                    .prod()
                )
                variables["wgt_btag_up"] = variables["wgt_btag_up"].fillna(1.0)
                variables["wgt_btag_up"] = variables[
                    "wgt_btag_up"
                ] * weights.get_weight("nominal")
                variables["wgt_btag_down"] = (
                    jets.loc[jets.pre_selection == 1, "btag_sf_wp_down"]
                    .groupby("entry")
                    .prod()
                )
                variables["wgt_btag_down"] = variables["wgt_btag_down"].fillna(1.0)
                variables["wgt_btag_down"] = variables[
                    "wgt_btag_down"
                ] * weights.get_weight("nominal")

            else:
                variables["wgt_nominal"] = 1.0
                variables["wgt_btag_up"] = 1.0
                variables["wgt_btag_down"] = 1.0
        else:
            if is_mc:
                variables["wgt_nominal"] = 1.0
                variables["wgt_nominal"] = variables[
                    "wgt_nominal"
                ] * weights.get_weight("nominal")

            else:
                variables["wgt_nominal"] = 1.0

        jets["selection"] = 0
        jets.loc[
            ((jets.pt > 30.0) & (abs(jets.eta) < 2.4) & (jets.jetId >= 2)),
            "selection",
        ] = 1

        print(f"jets:\n  {jets.to_string()}")

        njets = jets.loc[:, "selection"].groupby("entry").sum()
        variables["njets"] = njets

        jets["bselection"] = 0
        jets.loc[
            (
                (jets.pt > 30.0)
                & (abs(jets.eta) < 2.4)
                & (jets.btagDeepFlavB > parameters["UL_btag_medium"][self.year])
                & (jets.jetId >= 2)
            ),
            "bselection",
        ] = 1

        nbjets = jets.loc[:, "bselection"].groupby("entry").sum()
        variables["nbjets"] = nbjets

        bjets = jets.query("bselection==1")
        bjets = bjets.sort_values(["entry", "pt"], ascending=[True, False])
        print(f"bjets: \n {bjets.to_string()}")
        bjet1 = bjets.groupby("entry").nth(0)
        bjet2 = bjets.groupby("entry").nth(1)
        bjets_df = bjet1.join(bjet2, how="outer", lsuffix='_bjet1', rsuffix='_bjet2')
        bJets = [bjet1, bjet2]
        print(f"bjet1: \n {bjet1.to_string()}")
        print(f"bjet2: \n {bjet2.to_string()}")
        print(f"bjets_df: \n {bjets_df.to_string()}")
        mu_col = []
        el_col = []
        for col in leptons.columns:
            if "_mu" in col:
                mu_col.append(col)
            elif "_el" in col:
                el_col.append(col)
        print(f"mu_col: {mu_col}")
        print(f"el_col: {el_col}")
        mu = leptons[mu_col]
        el = leptons[el_col]
        print(f"mu: \n {mu.to_string()}")
        print(f"el: \n {el.to_string()}")
        lepton_l = [mu, el]
        print("jet loop flag6")
        print(f"leptons b4 fill b_jets: \n {leptons.to_string()}")
        fill_bjets(leptons, variables, bJets, lepton_l, is_mc=is_mc)
        print(f"leptons after fill b_jets: \n {leptons.to_string()}")

        jets = jets.sort_values(["entry", "pt"], ascending=[True, False])
        jet1 = jets.groupby("entry").nth(0)
        jet2 = jets.groupby("entry").nth(1)
        jets_df = jet1.join(jet2, how="outer", lsuffix='_jet1', rsuffix='_jet2')
        print(f"jet1: \n {jet1.to_string()}")
        print(f"jet2: \n {jet2.to_string()}")
        print(f"jets_df: \n {jets_df.to_string()}")
        Jets = [jet1, jet2]
        # print("jet loop flag7")
        print(f"leptons b4 fill jets:\n  {leptons.to_string()}")
        fill_jets(leptons, variables, Jets, is_mc=is_mc)
        print(f"leptons after fill jets: \n {leptons.to_string()}")
        if self.timer:
            self.timer.add_checkpoint("Filled jet variables")

        # --------------------------------------------------------------#
        # Fill outputs
        # --------------------------------------------------------------#
        # All variables are affected by jet pT because of jet selections:
        # a jet may or may not be selected depending on pT variation.

        for key, val in variables.items():
            print(f"key: {key}, val: {val}")
            # output.loc[:, key] = val

        del df
        # del muons
        del jets
        del bjets
        # del mu1
        # del mu2
        # return output
        cols_to_keep = ["mass", "pt", "eta","phi"]
        print(f"bjets_df b4: \n {bjets_df.to_string()}")
        bjets_df = filter_df_cols(bjets_df, cols_to_keep)
        print(f"bjets_df after: \n {bjets_df.to_string()}")
        jets_df = filter_df_cols(jets_df, cols_to_keep)
        

        return bjets_df, jets_df

    def prepare_lookups(self):
        # self.jec_factories, self.jec_factories_data = jec_factories(self.year)
        # Muon scale factors
        # self.musf_lookup = musf_lookup(self.parameters)
        # Pile-up reweighting
        self.pu_lookups = pu_lookups(self.parameters)
        # Btag weights
        # self.btag_lookup = BTagScaleFactor(
        #        "data/b-tagging/DeepCSV_102XSF_WP_V1.csv", "medium"
        #    )
        # self.btag_lookup = BTagScaleFactor(
        #    self.parameters["btag_sf_csv"],
        #    BTagScaleFactor.RESHAPE,
        #    "iterativefit,iterativefit,iterativefit",
        # )
        # self.btag_lookup = btagSF("2018", jets.hadronFlavour, jets.eta, jets.pt, jets.btagDeepFlavB)

        # --- Evaluator
        # self.extractor = extractor()
        # PU ID weights
        # puid_filename = self.parameters["puid_sf_file"]
        # self.extractor.add_weight_sets([f"* * {puid_filename}"])

        # self.extractor.finalize()
        # self.evaluator = self.extractor.make_evaluator()

        return

    @property
    def accumulator(self):
        return processor.defaultdict_accumulator(int)

    @property
    def muoncolumns(self):
        return muon_branches

    @property
    def jetcolumns(self):
        return jet_branches

    def postprocess(self, accumulator):
        return accumulator
