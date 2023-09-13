function [ptsOut,indexSet] = reducePts_haa(pts, dst)

%Reduces a point set, pts, in a stochastic manner, such that the minimum sdistance
% between points is 'dst'. Writen by abd, edited by haa, then by raje

% get number of points
nPoints=size(pts,2);

% create an index set into the points array
indexSet=true(nPoints,1);

% get random permutation of indices 1-nPoints
RandOrd=randperm(nPoints);

% build KD-Tree from points array
NS = KDTreeSearcher(pts');

% search the KNTree for close neighbours in a chunk-wise fashion to save memory if point cloud is really big
Chunks=1:min(4e6,nPoints-1):nPoints;
Chunks(end)=nPoints;

% for each chunk of points
for cChunk=1:(length(Chunks)-1)
    Range=Chunks(cChunk):Chunks(cChunk+1);
    
    % get indices for points in KD-Tree 'NS' that are closer than 'dst' to
    % randomly sampled chunk of points in original point cloud
    idx = rangesearch(NS,pts(:,RandOrd(Range))',dst);
    
    for i = 1:size(idx,1)
        % compute source index 'id'
        id = RandOrd(i-1+Chunks(cChunk));
        
        % if point at 'id' is still included
        if (indexSet(id))
            % "remove" close point 'idx{i}'
            indexSet(idx{i}) = 0;
            
            % re-set 
            indexSet(id) = 1;
        end
    end
end

ptsOut = pts(:,indexSet);
