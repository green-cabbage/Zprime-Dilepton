echo "--setting up the rucio environment--"
source /cvmfs/cms.cern.ch/cmsset_default.sh
source /cvmfs/cms.cern.ch/rucio/setup-py3.sh

echo "--setting up the voms proxy--"
# voms-proxy-init --voms cms --valid 192:0:0 

echo "==proxy is valid=="

export RUCIO_ACCOUNT=`whoami`

echo "adding rucio container"

#rucio add-container user.amkaur:/Analyses/zprimebjets_DYMCInclusive_NanoAOD/USER

#
#TT1_dataset=( $(dasgoclient --query="dataset = /TTToLL_MLL_*/RunIISummer20UL16*NanoAOD*v9-106X_mcRun2_asymptotic*/NANOAODSIM"))
#
#TT2_dataset=( $(dasgoclient --query="dataset = /TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16*NanoAOD*v9-106X_mcRun2_asymptotic*/NANOAODSIM"))
#
#elec_dataset+=(${TT1_dataset[@]})
#elec_dataset+=(${TT2_dataset[@]})
#

dy18=( $(dasgoclient --query="dataset = /DYJetsToLL_*J_MLL_*_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18*NanoAOD*v9*/NANOAODSIM"))
dy18+=( $(dasgoclient --query="dataset = /DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM"))
# gather other_mc, starting with ttbar
other_mc18=( $(dasgoclient --query="dataset = /TTToLL_MLL*_TuneCP5_13TeV*/RunIISummer20UL18MiniAODv2*/MINIAODSIM"))
other_mc18+=( $(dasgoclient --query="dataset = /TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM"))
# WW
other_mc18+=( $(dasgoclient --query="dataset = /WWTo2L2Nu_MLL_*_TuneCP5_13TeV*/RunIISummer20UL18NanoAODv9*/NANOAODSIM"))
other_mc18+=( $(dasgoclient --query="dataset = /WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM"))
# rest of the other_mc
other_mc18+=( $(dasgoclient --query="dataset = /ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM")) # Wantitop
other_mc18+=( $(dasgoclient --query="dataset = /ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM")) # tW
other_mc18+=( $(dasgoclient --query="dataset = /WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM")) # WZ1L1Nu2Q
other_mc18+=( $(dasgoclient --query="dataset = /WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM")) # WZ2L2Q
other_mc18+=( $(dasgoclient --query="dataset = /WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM")) # WZ3LNu
other_mc18+=( $(dasgoclient --query="dataset = /ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM")) # ZZ2L2Nu
other_mc18+=( $(dasgoclient --query="dataset = /ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM")) # ZZ2L2Q
other_mc18+=( $(dasgoclient --query="dataset = /ZZTo4L_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM")) # ZZ4L

# gather data
data18=( $(dasgoclient --query="dataset = /SingleMuon/Run2018A-UL2018_MiniAODv2_GT36-v1/MINIAOD"))
data18+=( $(dasgoclient --query="dataset = /SingleMuon/Run2018B-UL2018_MiniAODv2_GT36-v1/MINIAOD"))
data18+=( $(dasgoclient --query="dataset = /SingleMuon/Run2018C-UL2018_MiniAODv2_GT36-v1/MINIAOD"))
data18+=( $(dasgoclient --query="dataset = /SingleMuon/Run2018D-UL2018_MiniAODv2_GT36-v1/MINIAOD"))


sample18=(${dy18[@]})
sample18+=(${other_mc18[@]})
sample18+=(${data18[@]})

dy17=( $(dasgoclient --query="dataset = /DYJetsToLL_*J_MLL_*_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17*NanoAOD*v9*/NANOAODSIM"))
dy17+=( $(dasgoclient --query="dataset = /DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v2/MINIAODSIM"))
# gather other_mc, starting with ttbar
other_mc17=( $(dasgoclient --query="dataset = /TTToLL_MLL*_TuneCP5_13TeV*/RunIISummer20UL17MiniAODv2*/MINIAODSIM"))
other_mc17+=( $(dasgoclient --query="dataset = /TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/MINIAODSIM"))
# WW
other_mc17+=( $(dasgoclient --query="dataset = /WWTo2L2Nu_MLL_*_TuneCP5_13TeV*/RunIISummer20UL17NanoAODv9*/NANOAODSIM"))
other_mc17+=( $(dasgoclient --query="dataset = /WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v2/MINIAODSIM"))
# rest of the other_mc
other_mc17+=( $(dasgoclient --query="dataset = /ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/MINIAODSIM")) # Wantitop
other_mc17+=( $(dasgoclient --query="dataset = /ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/MINIAODSIM")) # tW
other_mc17+=( $(dasgoclient --query="dataset = /WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v2/MINIAODSIM")) # WZ1L1Nu2Q
other_mc17+=( $(dasgoclient --query="dataset = /WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v2/MINIAODSIM")) # WZ2L2Q
other_mc17+=( $(dasgoclient --query="dataset = /WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v2/MINIAODSIM")) # WZ3LNu
other_mc17+=( $(dasgoclient --query="dataset = /ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/MINIAODSIM")) # ZZ2L2Nu
other_mc17+=( $(dasgoclient --query="dataset = /ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v2/MINIAODSIM")) # ZZ2L2Q
other_mc17+=( $(dasgoclient --query="dataset = /ZZTo4L_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v2/MINIAODSIM")) # ZZ4L

# gather data
data17=( $(dasgoclient --query="dataset = /SingleMuon/Run2017B-UL2017_MiniAODv2-v1/MINIAOD"))
data17+=( $(dasgoclient --query="dataset = /SingleMuon/Run2017C-UL2017_MiniAODv2-v1/MINIAOD"))
data17+=( $(dasgoclient --query="dataset = /SingleMuon/Run2017D-UL2017_MiniAODv2-v1/MINIAOD"))
data17+=( $(dasgoclient --query="dataset = /SingleMuon/Run2017E-UL2017_MiniAODv2-v1/MINIAOD"))
data17+=( $(dasgoclient --query="dataset = /SingleMuon/Run2017F-UL2017_MiniAODv2-v1/MINIAOD"))


sample17=(${dy17[@]})
sample17+=(${other_mc17[@]})
sample17+=(${data17[@]})
# dy3=( $(dasgoclient --query="dataset = /DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM"))

# dy4=( $(dasgoclient --query="dataset = /DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM"))


# dy1+=(${dy2[@]})
# dy1+=(${dy3[@]})
# dy1+=(${dy4[@]})

echo "sample 2018"

for i in "${sample18[@]}"
do
   echo "$i"
   #rucio attach user.hyeonseo:/Analyses/zprimebjets_DYMCInclusive_NanoAOD/USER cms:$i
  
done

echo "${#sample18[@]}"

echo "sample 2017"

for i in "${sample17[@]}"
do
   echo "$i"
   #rucio attach user.hyeonseo:/Analyses/zprimebjets_DYMCInclusive_NanoAOD/USER cms:$i
  
done

echo "${#sample17[@]}"