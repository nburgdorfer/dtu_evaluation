% script to calculate the statistics for each scan given this will currently only run if distances have been measured
% for all included scans (UsedSets)

% uncomment if not running this script as part of a separate routine.
% commented out to allow passing 'arg_method' parameter
% clear all
% close all
% format compact

disp(newline + "------------------------------------------")

[dataPath,resultsPath]=getPaths();

MaxDist=20; %outlier thresshold of 20 mm

method_string=arg_method;
light_string='l3'; %'l7'; l3 is the setting with all lights on, l7 is randomly sampled between the 7 settings (index 0-6)
representation_string='Points'; %mvs representation 'Points' or 'Surfaces'

switch representation_string
    case 'Points'
        eval_string='_Eval_IJCV_'; %results naming
        settings_string='';
    case 'Surfaces'
        eval_string='_SurfEval_Trim_IJCV_'; %results naming
        settings_string='_surf_11_trim_8'; %poisson settings for surface input
end

nStat=length(UsedSets);

BaseStat.nStl=zeros(1,nStat);
BaseStat.nData=zeros(1,nStat);
BaseStat.MeanStl=zeros(1,nStat);
BaseStat.MeanData=zeros(1,nStat);
BaseStat.VarStl=zeros(1,nStat);
BaseStat.VarData=zeros(1,nStat);
BaseStat.MedStl=zeros(1,nStat);
BaseStat.MedData=zeros(1,nStat);

nScans = 0;
avg_nStl = 0.0;
avg_nData = 0.0;
avg_MeanStl = 0.0;
avg_MeanData = 0.0;
avg_VarStl = 0.0;
avg_VarData = 0.0;
avg_MedStl = 0.0;
avg_MedData = 0.0;

for cStat=1:length(UsedSets) %Data set number
    
    currentSet=UsedSets(cStat);
    disp(newline + "Evaluating scan " + currentSet + "...")
    
    %input results name
    EvalName=[resultsPath method_string eval_string num2str(currentSet) '.mat'];
    
    load(EvalName)
    
    Dstl=BaseEval.Dstl(BaseEval.StlAbovePlane); %use only points that are above the plane 
    Dstl=Dstl(Dstl<MaxDist); % discard outliers
    
    Ddata=BaseEval.Ddata(BaseEval.DataInMask); %use only points that within mask
    Ddata=Ddata(Ddata<MaxDist); % discard outliers
    
    BaseStat.nStl(cStat)=length(Dstl);
    BaseStat.nData(cStat)=length(Ddata);
    fprintf("Estimated Points %d\n", BaseStat.nData(cStat))
    fprintf("Ground-Truth Points %d\n", BaseStat.nStl(cStat))
    avg_nStl = avg_nStl + BaseStat.nStl(cStat);
    avg_nData = avg_nData + BaseStat.nData(cStat);
    
    BaseStat.MeanStl(cStat)=mean(Dstl);
    BaseStat.MeanData(cStat)=mean(Ddata);
    fprintf("Accuracy Mean %.5f\n", BaseStat.MeanData(cStat))
    fprintf("Completeness Mean %.5f\n", BaseStat.MeanStl(cStat))
    avg_MeanStl = avg_MeanStl + BaseStat.MeanStl(cStat);
    avg_MeanData = avg_MeanData + BaseStat.MeanData(cStat);
    
    BaseStat.VarStl(cStat)=var(Dstl);
    BaseStat.VarData(cStat)=var(Ddata);
    fprintf("Accuracy Variance %.5f\n", BaseStat.VarData(cStat))
    fprintf("Completeness Variance %.5f\n", BaseStat.VarStl(cStat))
    avg_VarStl = avg_VarStl + BaseStat.VarStl(cStat);
    avg_VarData = avg_VarData + BaseStat.VarData(cStat);
    
    BaseStat.MedStl(cStat)=median(Dstl);
    BaseStat.MedData(cStat)=median(Ddata);
    fprintf("Accuracy Median %.5f\n", BaseStat.MedData(cStat))
    fprintf("Completeness Median %.5f\n", BaseStat.MedStl(cStat))
    avg_MedStl = avg_MedStl + BaseStat.MedStl(cStat);
    avg_MedData = avg_MedData + BaseStat.MedData(cStat);
    
    nScans = nScans + 1;
    time=clock;
end
fprintf(newline + "----Averages----\n")
fprintf("Estimated Points %d\n", avg_nData/nScans)
fprintf("Ground-Truth Points %d\n", avg_nStl/nScans)

fprintf("Accuracy Mean %.5f\n", avg_MeanData/nScans)
fprintf("Completeness Mean %.5f\n", avg_MeanStl/nScans)

fprintf("Accuracy Variance %.5f\n", avg_VarData/nScans)
fprintf("Completeness Variance %.5f\n", avg_VarStl/nScans)

fprintf("Accuracy Median %.5f\n", avg_MedData/nScans)
fprintf("Completeness Median %.5f\n", avg_MedStl/nScans)

totalStatName=[resultsPath 'TotalStat_' method_string eval_string '.mat'];
save(totalStatName,'BaseStat','time','MaxDist');


