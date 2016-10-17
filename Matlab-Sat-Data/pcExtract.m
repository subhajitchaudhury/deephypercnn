function [comp,recon]=pcExtract(Z,r)
    sz=size(Z);
    d=sz(1);
    n=prod(sz(2:end));
    Z_r=reshape(Z,[d,n]);
    
    %principal components 
    [Zpca, T, U, mu] = myPCA(Z_r,r);
    comp=Zpca;
    
    size(comp)
    
    %reconstruction
    recon = U / T * Zpca + repmat(mu,1,n);
    
    err=100*norm(recon(:)-Z(:))/norm(Z(:));
    
    fprintf('\nError is reconstruction is %f percent \n\n',err);
    
    if length(sz)>2
        new_size=[r,sz(2:end)];
        comp=reshape(comp,new_size);
        
        new_size=[d,sz(2:end)];
        recon=reshape(recon,new_size);
    end
    
    