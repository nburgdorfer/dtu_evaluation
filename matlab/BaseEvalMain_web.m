% script to calculate distances have been measured for all included scans (UsedSets)

% uncomment if not running this script as part of a separate routine.
% commented out to allow passing 'arg_method' parameter
% clear all
% close all
% format compact

addpath('MeshSupSamp_web/x64/Release');

[dataPath,resultsPath]=getPaths();

method_string=arg_method;
disp("Running evaluation for " + arg_method + "...")

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

dst=0.2;    %Min dist between points when reducing

for cSet=UsedSets
    t0 = tic();
    
    disp(newline + "Evaluating scan " + cSet + "...")
    
    % data names
    DataInName=[dataPath sprintf('/%s/%s/%s%03d_%s%s.ply',representation_string,lower(method_string),lower(method_string),cSet,light_string,settings_string)];
    StlInName=[dataPath '/Points/stl/stl' sprintf('%03d',cSet) '_total.ply'];
    EvalName=[resultsPath method_string eval_string num2str(cSet) '.mat'];
    
    %check if file is already computed
    if(~exist(EvalName,'file'))        
        disp("Loading estimated point cloud...");
        Mesh = plyread(DataInName);
        Qdata=[Mesh.vertex.x Mesh.vertex.y Mesh.vertex.z]';
        if(strcmp(representation_string,'Surfaces'))
            %upsample triangles
            Tri=cell2mat(Mesh.face.vertex_indices)';
            Qdata=MeshSupSamp(Qdata,Tri,dst);
        end
        
        disp("Loading ground-truth point cloud...");
        StlMesh = plyread(StlInName);  %STL points already reduced 0.2 mm neighbourhood density
        Qstl=[StlMesh.vertex.x StlMesh.vertex.y StlMesh.vertex.z]';
        
        disp("Comparing point clouds...")
        BaseEval=PointCompareMain(cSet,Qdata,Qstl,dst,dataPath);
        
        disp('Saving results...')
        save(EvalName,'BaseEval');
        BaseEval2Obj_web(BaseEval,method_string, resultsPath) 
    end
    
    dt = toc(t0);
    fprintf("Elapsed time: %.3fs\n", dt)
end





