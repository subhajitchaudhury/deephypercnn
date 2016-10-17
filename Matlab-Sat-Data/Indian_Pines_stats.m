function stats=Indian_Pines_stats(indian_pines_gt)
    %load('Indian_pines_gt.mat')
    I=find(indian_pines_gt~=0);
    length(I);
    stats.class=[];
    for i=1:16
        I=find(indian_pines_gt==i);
        fprintf('\nNo of Class %d=%d\n',i,length(I));
        stats.class(i)=length(I);
    end
    stats.totalpix=sum(stats.class(:));
    fprintf('\nTotal no of Classes = %d\n',stats.totalpix);
end