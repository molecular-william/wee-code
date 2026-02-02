clc
clear
%% Read files
name='Betaglobin';
filename=strcat(name,'.fasta');
[IDs, U] = fastaread(filename);

%% Read PCH
load('Avgdata.mat')
DD = Avgdata;  %add abs for positive
DD(isnan(Avgdata))=0;
%% mean and standard devision
M=((sum(DD,2))/20)';
S=(std(DD,0,2))';
for i=1:size(DD,1)
 norm_data(i,:) = (DD(i,:) - M(i))/ S(i);
end
%%
Pattern= {'A' 'R' 'N' 'D' 'C' 'Q' 'E' 'G' 'H' 'I' 'L' 'K' 'M' 'F' 'P' 'S' 'T' 'W' 'Y' 'V'};
P = struct();
for k=1:size(norm_data,2)
    p = Pattern{1,k};
    y = norm_data(1:end,k)';
    P.(p) = y;  
end
P.('X') = [0];
%% Block generation1
L = zeros(1,length(U));
for k=1:length(U)
    L(k) = length(U{k});
end
Lmax = max(L);
Lm =50;
Lshift =5;
Le = Lm*fix(Lmax/Lm + 1);
for k=1:length(U)
    U{k}(L(k)+1:Le) = 'X';
end
numBlk = fix(Le/Lm);
%% Block generation2
Y = struct();
for n=1:length(U)
    BB= [];
    for k=-Lshift:Lshift
        str  = U{n}';str(end+1:end+abs(k))='X';
        strs = circshift(str,k)';strs(end-abs(k)+1:end)=[];
        Y(n).Data(k+Lshift+1).Mat = reshape(strs,Lm,[])';        
        for i = 1:size(Y(n).Data(k+Lshift+1).Mat,1)
            for j=1:size(Y(n).Data(k+Lshift+1).Mat,2)
             V = []; 
               strkey = Y(n).Data(k+Lshift+1).Mat(i,j);
               if isfield(P,strkey)
                v(j,:,i) = P.(strkey);
                for m=1:i
                W(m,:)  =sum(v(:,:,m),1);
                end
              Y(n).Count(k+Lshift+1).Data(m,:)= W(m,:);
               end
            end  
        end
    end
end

%% Moment 
load('PatternG.mat');
C = struct();
for n=1:length(U)
    CC= [];
    for k=-Lshift:Lshift
        str  = U{n}';str(end+1:end+abs(k))='X';
        strs = circshift(str,k)';strs(end-abs(k)+1:end)=[];
        C(n).Data(k+Lshift+1).Mat = reshape(strs,Lm,[])';
        
        for c = 1:110
        for i = 1:size(C(n).Data(k+Lshift+1).Mat,1)
            C(n).Count(k+Lshift+1).Data(c,:,i) = moment(C(n).Data(k+Lshift+1).Mat(i,:),PatternG{c});
        end
        end
         
        
for i=1:size(C(n).Count,2)
    
    for j=1:numBlk
     
    C(n).mom(i).Data(:,:,:)=permute(C(n).Count(i).Data,[2,1,3]);
    C(n).non(i).Data(j,:)=reshape(C(n).mom(i).Data(:,:,j),1,440);
   
    end
    
end 

    end
    
 %Normalization 
 
 for i=1:size(C(n).Count,2)
      M=((sum(C(n).non(i).Data,2))/440)';
      S=(std(C(n).non(i).Data,0,2))';
      for j=1:numBlk
      norm_data_mom(j,:) = (C(n).non(i).Data(j,:) - M(j))/ S(j);
      end   
      C(n).son(i).Data = norm_data_mom;
      C(n).son(i).Data(isnan( C(n).son(i).Data))=0;   
 end
end

%% Merge
for n=1:length(U)
    for i=1:size(C(n).Count,2)
F(n).Count(i).Data=[Y(n).Count(i).Data,C(n).son(i).Data];
    end
end
%%  Euclidean distance
[a0,b0]=size(F);
temp0=F(1).Count;
[a1,b1]=size(temp0);
temp1=temp0(1).Data;
[a2,b2]=size(temp1);

out3=zeros(b0);
out2=zeros(a2,1);
out1=zeros(b1);

for u=1:b0
    for v=1:b0
        for p=1:a2
            for i=1:b1
                for j=1:b1
                    G1=F(u).Count(i).Data(p,:);
                    G2=F(v).Count(j).Data(p,:);
                    D= max(abs(bsxfun(@minus,G1,G2)),[],2); 
                    out1(i,j)=D;
                end
            end
            out2(p)=min(min(out1));
        end
        out3(u,v)=sum(out2);
    end
end

out3

%% Mega format
outputfilename=strcat('PCV_',name,'.meg'); %Change Name of File
num_of_seq=length(U);
outfile=fopen(outputfilename, 'w');
fprintf(outfile, '#mega\n');
fprintf(outfile, '!Title: TEST;\n');
fprintf(outfile, '!Format DataType=Distance DataFormat=LowerLeft NTaxa=%d;\n', num_of_seq);
fprintf(outfile, '\n');
for k = 1 : num_of_seq
    fprintf(outfile, '[%d] #%s\n', k,IDs{k});
end
fprintf(outfile, '\n');
for j = 2 : num_of_seq
    fprintf(outfile, '[%d]   ', j);
    for k = 1 : (j-1) 
        fprintf(outfile, ' %8f',  out3(j, k));
    end
    fprintf(outfile, '\n');
end
fprintf(outfile, '\n');
fclose(outfile);
