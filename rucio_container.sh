echo "--setting up the rucio environment--"
source /cvmfs/cms.cern.ch/cmsset_default.sh
source /cvmfs/cms.cern.ch/rucio/setup-py3.sh

echo "--setting up the voms proxy--"
# voms-proxy-init --voms cms --valid 192:0:0 

echo "==proxy is valid=="

export RUCIO_ACCOUNT=`whoami`

echo "adding rucio container"

# rucio add-container user.hyeonseo:/Analyses/zprimebjets_DYMCInclusive_NanoAOD/USER


# gather data, starting with muon data
data18=( $(dasgoclient --query="dataset = /SingleMuon/Run2018*-UL2018_MiniAODv2_NanoAODv9_GT36-v1/NANOAOD")) 

dy18=( $(dasgoclient --query="dataset = /DYJetsToLL_*J_MLL_*_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18*NanoAOD*v9*/NANOAODSIM"))
dy18+=( $(dasgoclient --query="dataset = /DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM"))
# gather other_mc, starting with ttbar
other_mc18=( $(dasgoclient --query="dataset = /TTToLL_MLL*_TuneCP5_13TeV*/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"))
other_mc18+=( $(dasgoclient --query="dataset = /TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"))
# WW
other_mc18+=( $(dasgoclient --query="dataset = /WWTo2L2Nu_MLL_*_TuneCP5_13TeV*/RunIISummer20UL18NanoAODv9*/NANOAODSIM"))
other_mc18+=( $(dasgoclient --query="dataset = /WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM"))
# rest of the other_mc
other_mc18+=( $(dasgoclient --query="dataset = /ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")) # Wantitop
other_mc18+=( $(dasgoclient --query="dataset = /ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")) # tW
other_mc18+=( $(dasgoclient --query="dataset = /WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")) # WZ1L1Nu2Q
other_mc18+=( $(dasgoclient --query="dataset = /WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")) # WZ2L2Q
other_mc18+=( $(dasgoclient --query="dataset = /WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")) # WZ3LNu
other_mc18+=( $(dasgoclient --query="dataset = /ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")) # ZZ2L2Nu
other_mc18+=( $(dasgoclient --query="dataset = /ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")) # ZZ2L2Q
other_mc18+=( $(dasgoclient --query="dataset = /ZZTo4L_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")) # ZZ4L
# Higgs
higgs_18=( $(dasgoclient --query="dataset = /TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"))
higgs_18+=( $(dasgoclient --query="dataset = /TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"))
higgs_18+=( $(dasgoclient --query="dataset = /ttHJetTobb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"))
higgs_18+=( $(dasgoclient --query="dataset = /ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"))
higgs_18+=( $(dasgoclient --query="dataset = /GluGluHToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"))
higgs_18+=( $(dasgoclient --query="dataset = /VBF_HToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"))
higgs_18+=( $(dasgoclient --query="dataset = /ZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"))
higgs_18+=( $(dasgoclient --query="dataset = /ggZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"))
higgs_18+=( $(dasgoclient --query="dataset = /ttH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM"))
higgs_18+=( $(dasgoclient --query="dataset = /GluGluHToZZTo2L2Q_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"))
higgs_18+=( $(dasgoclient --query="dataset = /ZH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2-minlo-HZJ_JHUGenV7011_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM"))

# Triboson
triboson_18=( $(dasgoclient --query="dataset = /WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"))
triboson_18+=( $(dasgoclient --query="dataset = /WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"))
triboson_18+=( $(dasgoclient --query="dataset = /WZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"))
triboson_18+=( $(dasgoclient --query="dataset = /ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"))






sample18=(${dy18[@]})
sample18+=(${other_mc18[@]})
sample18+=(${data18[@]})
sample18+=(${higgs_18[@]})
sample18+=(${triboson_18[@]})


# gather data, starting with muon data
data17=( $(dasgoclient --query="dataset = /SingleMuon/Run2017B-UL2017_MiniAODv2_NanoAODv9_GT36-v1/NANOAOD")) 
data17+=( $(dasgoclient --query="dataset = /SingleMuon/Run2017C-UL2017_MiniAODv2_NanoAODv9_GT36-v1/NANOAOD")) 
data17+=( $(dasgoclient --query="dataset = /SingleMuon/Run2017D-UL2017_MiniAODv2_NanoAODv9_GT36-v1/NANOAOD")) 
data17+=( $(dasgoclient --query="dataset = /SingleMuon/Run2017E-UL2017_MiniAODv2_NanoAODv9_GT36-v2/NANOAOD")) 
data17+=( $(dasgoclient --query="dataset = /SingleMuon/Run2017F-UL2017_MiniAODv2_NanoAODv9_GT36-v1/NANOAOD")) 
data17+=( $(dasgoclient --query="dataset = /SingleMuon/Run2017G-UL2017_MiniAODv2_NanoAODv9_GT36-v2/NANOAOD")) 
data17+=( $(dasgoclient --query="dataset = /SingleMuon/Run2017H-UL2017_MiniAODv2_NanoAODv9_GT36-v1/NANOAOD")) 
# data17+=( $(dasgoclient --query="dataset = /DoubleEG/Run2017*-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD")) # add electron data

dy17=( $(dasgoclient --query="dataset = /DYJetsToLL_*J_MLL_*_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17*NanoAOD*v9*/NANOAODSIM"))
dy17+=( $(dasgoclient --query="dataset = /DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM"))
# gather other_mc, starting with ttbar
other_mc17=( $(dasgoclient --query="dataset = /TTToLL_MLL*_TuneCP5_13TeV*/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM"))
other_mc17+=( $(dasgoclient --query="dataset = /TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM"))
# WW
other_mc17+=( $(dasgoclient --query="dataset = /WWTo2L2Nu_MLL_*_TuneCP5_13TeV*/RunIISummer20UL17NanoAODv9*/NANOAODSIM"))
other_mc17+=( $(dasgoclient --query="dataset = /WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM"))
# rest of the other_mc
other_mc17+=( $(dasgoclient --query="dataset = /ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")) # Wantitop
other_mc17+=( $(dasgoclient --query="dataset = /ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")) # tW
other_mc17+=( $(dasgoclient --query="dataset = /WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")) # WZ1L1Nu2Q
other_mc17+=( $(dasgoclient --query="dataset = /WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")) # WZ2L2Q
other_mc17+=( $(dasgoclient --query="dataset = /WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")) # WZ3LNu
other_mc17+=( $(dasgoclient --query="dataset = /ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")) # ZZ2L2Nu
other_mc17+=( $(dasgoclient --query="dataset = /ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")) # ZZ2L2Q
other_mc17+=( $(dasgoclient --query="dataset = /ZZTo4L_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")) # ZZ4L
# Higgs
higgs_17=( $(dasgoclient --query="dataset = /TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM"))
higgs_17+=( $(dasgoclient --query="dataset = /TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM"))
higgs_17+=( $(dasgoclient --query="dataset = /ttHJetTobb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM"))
higgs_17+=( $(dasgoclient --query="dataset = /ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM"))
higgs_17+=( $(dasgoclient --query="dataset = /GluGluHToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM"))
higgs_17+=( $(dasgoclient --query="dataset = /VBF_HToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM"))
higgs_17+=( $(dasgoclient --query="dataset = /ZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM"))
higgs_17+=( $(dasgoclient --query="dataset = /ggZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM"))
higgs_17+=( $(dasgoclient --query="dataset = /ttH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM"))
higgs_17+=( $(dasgoclient --query="dataset = /GluGluHToZZTo2L2Q_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM"))
higgs_17+=( $(dasgoclient --query="dataset = /ZH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2-minlo-HZJ_JHUGenV7011_pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM"))

# Triboson
triboson_17=( $(dasgoclient --query="dataset = /WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM"))
triboson_17+=( $(dasgoclient --query="dataset = /WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM"))
triboson_17+=( $(dasgoclient --query="dataset = /WZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM"))
triboson_17+=( $(dasgoclient --query="dataset = /ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM"))



sample17=(${data17[@]})
sample17+=(${dy17[@]})
sample17+=(${other_mc17[@]})
sample17+=(${higgs_17[@]})
sample17+=(${triboson_17[@]})

# for 2016, only higgs and triboson

# Higgs
higgs_16_post=( $(dasgoclient --query="dataset = /TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM"))
higgs_16_post+=( $(dasgoclient --query="dataset = /TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM"))
higgs_16_post+=( $(dasgoclient --query="dataset = /ttHJetTobb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM"))
higgs_16_post+=( $(dasgoclient --query="dataset = /ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM"))
higgs_16_post+=( $(dasgoclient --query="dataset = /GluGluHToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM"))
higgs_16_post+=( $(dasgoclient --query="dataset = /VBF_HToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM"))
higgs_16_post+=( $(dasgoclient --query="dataset = /ZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM"))
higgs_16_post+=( $(dasgoclient --query="dataset = /ggZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM"))
higgs_16_post+=( $(dasgoclient --query="dataset = /ttH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM"))
higgs_16_post+=( $(dasgoclient --query="dataset = /GluGluHToZZTo2L2Q_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM"))
higgs_16_post+=( $(dasgoclient --query="dataset = /ZH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2-minlo-HZJ_JHUGenV7011_pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM"))

# Triboson
triboson_16_post=( $(dasgoclient --query="dataset = /WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM"))
triboson_16_post+=( $(dasgoclient --query="dataset = /WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIMSIM"))
triboson_16_post+=( $(dasgoclient --query="dataset = /WZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM"))
triboson_16_post+=( $(dasgoclient --query="dataset = /ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM"))

sample16_post=(${higgs_16_post[@]})
sample16_post+=(${triboson_16_post[@]})

# Higgs
higgs_16_pre=( $(dasgoclient --query="dataset = /TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM"))
higgs_16_pre+=( $(dasgoclient --query="dataset = /TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM"))
higgs_16_pre+=( $(dasgoclient --query="dataset = /ttHJetTobb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM"))
higgs_16_pre+=( $(dasgoclient --query="dataset = /ttHJetToNonbb_M125_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM"))
higgs_16_pre+=( $(dasgoclient --query="dataset = /GluGluHToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM"))
higgs_16_pre+=( $(dasgoclient --query="dataset = /VBF_HToZZTo4L_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM"))
higgs_16_pre+=( $(dasgoclient --query="dataset = /ZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM"))
higgs_16_pre+=( $(dasgoclient --query="dataset = /ggZH_HToBB_ZToLL_M-125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM"))
higgs_16_pre+=( $(dasgoclient --query="dataset = /ttH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM"))
higgs_16_pre+=( $(dasgoclient --query="dataset = /GluGluHToZZTo2L2Q_M125_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM"))
higgs_16_pre+=( $(dasgoclient --query="dataset = /ZH_HToZZ_4LFilter_M125_TuneCP5_13TeV_powheg2-minlo-HZJ_JHUGenV7011_pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM"))

# Triboson
triboson_16_pre=( $(dasgoclient --query="dataset = /WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM"))
triboson_16_pre+=( $(dasgoclient --query="dataset = /WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM"))
triboson_16_pre+=( $(dasgoclient --query="dataset = /WZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM"))
triboson_16_pre+=( $(dasgoclient --query="dataset = /ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM"))

sample16_pre=(${higgs_16_pre[@]})
sample16_pre+=(${triboson_16_pre[@]})


echo "sample 2018 start"

for i in "total number of 2018 samples: ${sample18[@]}"
do
   echo "$i"
   # rucio attach user.hyeonseo:/Analyses/zprimebjets_DYMCInclusive_NanoAOD/USER cms:$i
  
done

echo "${#sample18[@]}"

echo "sample 2017 start"

for i in "${sample17[@]}"
do
   echo "$i"
   # rucio attach user.hyeonseo:/Analyses/zprimebjets_DYMCInclusive_NanoAOD/USER cms:$i
  
done

echo "total number of 2017 samples: ${#sample17[@]}"

echo "sample 2016 post start"

for i in "${sample16_post[@]}"
do
   echo "$i"
   # rucio attach user.hyeonseo:/Analyses/zprimebjets_DYMCInclusive_NanoAOD/USER cms:$i
  
done

echo "total number of 2016 post samples: ${#sample16_post[@]}"

echo "sample 2016 pre start"

for i in "${sample16_pre[@]}"
do
   echo "$i"
   # rucio attach user.hyeonseo:/Analyses/zprimebjets_DYMCInclusive_NanoAOD/USER cms:$i
  
done

echo "total number of 2016 pre samples: ${#sample16_pre[@]}"