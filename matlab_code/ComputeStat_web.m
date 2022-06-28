% script to calculate the statistics for each scan given this will currently only run if distances have been measured
% for all included scans (UsedSets)

% uncomment if not running this script as part of a separate routine.
% commented out to allow passing 'arg_method' parameter
% clear all
% close all
% format compact

[dataPath,resultsPath]=getPaths();

MaxDist=20; %outlier thresshold of 20 mm

time=clock;time(4:5), drawnow

method_string=arg_method %'nate'%'tola' %'camp' %'furu';
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

% get sets used in evaluation
% if(strcmp(light_string,'l7'))
%     UsedSets=GetUsedLightSets;
%     eval_string=[eval_string 'l7_'];
% else
%     UsedSets=GetUsedSets;
% end

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
    disp(UsedSets(cStat))
    
    %input results name
    EvalName=[resultsPath method_string eval_string num2str(currentSet) '.mat']
    
    load(EvalName)
    
    Dstl=BaseEval.Dstl(BaseEval.StlAbovePlane); %use only points that are above the plane 
    Dstl=Dstl(Dstl<MaxDist); % discard outliers
    
    Ddata=BaseEval.Ddata(BaseEval.DataInMask); %use only points that within mask
    Ddata=Ddata(Ddata<MaxDist); % discard outliers
    
    BaseStat.nStl(cStat)=length(Dstl);
    BaseStat.nData(cStat)=length(Ddata);
    fprintf("N-Stl %d\n", BaseStat.nStl(cStat))
    fprintf("N-Data %d\n", BaseStat.nData(cStat))
    avg_nStl = avg_nStl + BaseStat.nStl(cStat);
    avg_nData = avg_nData + BaseStat.nData(cStat);
    
    BaseStat.MeanStl(cStat)=mean(Dstl);
    BaseStat.MeanData(cStat)=mean(Ddata);
    fprintf("MeanStl %d\n", BaseStat.MeanStl(cStat))
    fprintf("MeanData %d\n", BaseStat.MeanData(cStat))
    avg_MeanStl = avg_MeanStl + BaseStat.MeanStl(cStat);
    avg_MeanData = avg_MeanData + BaseStat.MeanData(cStat);
    
    BaseStat.VarStl(cStat)=var(Dstl);
    BaseStat.VarData(cStat)=var(Ddata);
    fprintf("VarStl %d\n", BaseStat.VarStl(cStat))
    fprintf("VarData %d\n", BaseStat.VarData(cStat))
    avg_VarStl = avg_VarStl + BaseStat.VarStl(cStat);
    avg_VarData = avg_VarData + BaseStat.VarData(cStat);
    
    BaseStat.MedStl(cStat)=median(Dstl);
    BaseStat.MedData(cStat)=median(Ddata);
    fprintf("MedStl %d\n", BaseStat.MedStl(cStat))
    fprintf("MedData %d\n", BaseStat.MedData(cStat))
    avg_MedStl = avg_MedStl + BaseStat.MedStl(cStat);
    avg_MedData = avg_MedData + BaseStat.MedData(cStat);
    
    nScans = nScans + 1;
    time=clock;[time(4:5) currentSet cStat], drawnow
end
fprintf("----Averages----\n")
fprintf("N-Stl %d\n", avg_nStl/nScans)
fprintf("N-Data %d\n", avg_nData/nScans)

fprintf("MeanStl %d\n", avg_MeanStl/nScans)
fprintf("MeanData %d\n", avg_MeanData/nScans)

fprintf("VarStl %d\n", avg_VarStl/nScans)
fprintf("VarData %d\n", avg_VarData/nScans)

fprintf("MedStl %d\n", avg_MedStl/nScans)
fprintf("MedData %d\n", avg_MedData/nScans)

totalStatName=[resultsPath 'TotalStat_' method_string eval_string '.mat']
save(totalStatName,'BaseStat','time','MaxDist');


