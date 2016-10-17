clear all;
close all;

%load the hyperspectral data
load('./IndianPines/Indian_pines_corrected.mat');
im=double(indian_pines_corrected);

%load the labels
ip_gt=imread('indian_pines_gt.tif');

%get statistics of the data
stats=Indian_Pines_stats(ip_gt);
num_pix=stats.totalpix;
[h,w,ch]=size(im);

%center the data
mu=mean(double(im(:)));
sig=std(double(im(:)));
%im_cen=(im-mu)./sig;
im_cen=im;

%pad the im_cen
conv_size=5;
pad_no=(conv_size-1)/2;

%padded image
im_X=zeros(h+2*pad_no,w+2*pad_no,ch);
for i=1:ch
    im_i=im_cen(:,:,i);
    im_X(:,:,i)=padarray(im_i,[pad_no,pad_no],'symmetric');
end

ip_gt_pad=padarray(ip_gt,[pad_no,pad_no],'symmetric');

cnt=1;
X=zeros(num_pix,ch,conv_size,conv_size);
labels=zeros(num_pix,1);
verbose=false;
all_labels=true;
for y=1:h
    y
    for x=1:w
        if ip_gt(y,x)~=0
        %if all_labels
            X(cnt,:,:,:)=permute(im_X(y:y+conv_size-1,x:x+conv_size-1,:),...
                [3,1,2]);
            labels(cnt)=ip_gt(y,x);
            
            if verbose
%                 subplot(1,2,1);
                figure(1);
                imagesc(ip_gt_pad);
                rectangle('Position',[x y conv_size conv_size],...
                    'EdgeColor','r','linewidth',2);
                title(sprintf('Label=%d',labels(cnt)));
%                 subplot(1,2,2);
%                 imagesc(ip_gt);
%                 rectangle('Position',[x y conv_size conv_size],...
%                     'EdgeColor','r','linewidth',2);
%                 
                
            end
            drawnow;
            pause(0.01);
            cnt=cnt+1;
        end
        
    end
end

%reduce the number of 0 labels
ind=find(labels==0);
ind=ind(2500:end);
X(ind,:,:,:)=[];
labels(ind)=[];

%permute the labels
num_pix=length(labels);
ind=randperm(num_pix,num_pix);
X=single(X(ind,:,:,:));
labels=uint8(labels(ind));

if all_labels
    save('indian_pines_data_all.mat','X','labels','-v7.3');
else
    save('indian_pines_data.mat','X','labels','-v7.3');
end

[comp,recon]=pcExtract(permute(X,[2,1,3,4]),30);
X_r=permute(comp,[2,1,3,4]);

save('indian_pines_data_pca.mat','X_r','labels','-v7.3');

