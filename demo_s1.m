clear;
close all;

mkdir('./data/image');
mkdir('./data/supervoxel');
mkdir('./data/supervoxel/edge');
mkdir('./data/supervoxel/feature');
mkdir('./data/supervoxel/feature/IH');
mkdir('./data/supervoxel/supervoxel_regionprops');

fileID = fopen('indexlist.txt');
image_lst = {};
tline = fgetl(fileID);
while ischar(tline)
    image_lst{end+1} = tline;
    tline = fgetl(fileID);
end

for z=1:length(image_lst)
    disp(['processing ',num2str(z),'/',num2str(length(image_lst)),' data']);
    image_name = image_lst{z};
    load(fullfile('./data',[image_name,'.mat']));
    im_volume = permute(data,[1,2,4,3]);
    save(fullfile('./data/image',[image_name,'.mat']),'im_volume');
    
    input_data_file = fullfile('./data/image',[image_name,'.mat']);
    load(input_data_file);
    im_volume = uint8(squeeze(im_volume));
    numreqiredsupervoxels = 1100;
    compactness = 3;
    [labels, numlabels] = slicsupervoxelmex(im_volume,numreqiredsupervoxels,compactness);
    S = labels+1;
    save(fullfile('./data/supervoxel',[image_name,'.mat']),'S');
    
    load(input_data_file);
    im_volume = squeeze(im_volume);
    supervoxel_num = max(S(:));
    supervoxel_ih = [];
    for j=1:supervoxel_num
        ih = imhist(uint8( im_volume(S==j)), 32);
        ih = ih./sum(ih);
        supervoxel_ih = [supervoxel_ih,ih];
    end
    save(fullfile('./data/supervoxel/feature/IH',[image_name,'.mat']),'supervoxel_ih');
    
    load(fullfile('./data/supervoxel/feature/IH',[image_name,'.mat']));
    load(fullfile('./data/supervoxel',[image_name,'.mat']));
    supervoxel_num = max(S(:));
    volume_size = size(S);
    adjcMerge = zeros(supervoxel_num,supervoxel_num);
    for j=1:volume_size(3)
        adjcMerge = AdjcProcloop(S(:,:,j),adjcMerge);
    end
    adjc=[];
    for k=1:supervoxel_num
        indext=[];
        ind=find(adjcMerge(k,:)==1);
        indext=[indext,ind];
        indext=indext((indext>k));
        indext=unique(indext);
        if(~isempty(indext))
            ed=int32(ones(length(indext),2));
            ed(:,2)=k*ed(:,2);
            ed(:,1)=indext;
            adjc=[adjc;ed];
        end
    end
    adjc = double(adjc);
    fvs = [];
    for u=1:supervoxel_num
        fv = [supervoxel_ih(:,u)];
        fvs = [fvs, fv];
    end
    dst = zeros(1,size(adjc,1));
    for v=1:size(adjc,1)
        dst(v) = intersection_dst(fvs(:,adjc(v,1)), fvs(:,adjc(v,2)));
    end
    weights = dst;
    W=sparse([adjc(:,1);adjc(:,2)],[adjc(:,2);adjc(:,1)], [weights';weights'],double(supervoxel_num),double(supervoxel_num));
    W = full(W);
    save( fullfile('./data/supervoxel/edge',[image_name,'.mat']),'W');
    
    supervoxel_regionprops = regionprops(S, 'all');
    save(fullfile('./data/supervoxel/supervoxel_regionprops',[image_name,'.mat']),'supervoxel_regionprops');
    
end

disp('stage1 completed. please run demo_s2.py');

function adjcMerge = AdjcProcloop(M,T)
% $Description:
%    -compute the adjacent matrix
% $Agruments
% Input;
%    -M: superpixel label matrix
%    -N: superpixel number
% Output:
%    -adjcMerge: adjacent matrix

adjcMerge = T;
[m, n] = size(M);

for i = 1:m-1
    for j = 1:n-1
        if(M(i,j)~=M(i,j+1))
            adjcMerge(M(i,j),M(i,j+1)) = 1;
            adjcMerge(M(i,j+1),M(i,j)) = 1;
        end
        if(M(i,j)~=M(i+1,j))
            adjcMerge(M(i,j),M(i+1,j)) = 1;
            adjcMerge(M(i+1,j),M(i,j)) = 1;
        end
        if(M(i,j)~=M(i+1,j+1))
            adjcMerge(M(i,j),M(i+1,j+1)) = 1;
            adjcMerge(M(i+1,j+1),M(i,j)) = 1;
        end
        if(M(i+1,j)~=M(i,j+1))
            adjcMerge(M(i+1,j),M(i,j+1)) = 1;
            adjcMerge(M(i,j+1),M(i+1,j)) = 1;
        end
    end
end
bd=unique([M(1,:),M(m,:),M(:,1)',M(:,n)']);
for i=1:length(bd)
    for j=i+1:length(bd)
        adjcMerge(bd(i),bd(j))=1;
        adjcMerge(bd(j),bd(i))=1;
    end
end
end

function [d] = intersection_dst(A,B)

A = A(:);
B = B(:);
n = size(A, 1);
d = 0;
for i=1:n
    d = d+min(A(i), B(i));
end
end