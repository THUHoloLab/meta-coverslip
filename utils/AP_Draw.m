function AP_Draw(x,y,amplitude,phase,title1,title2,xlable,ylable)
    set(gcf,'unit','centimeters','Position',[2,2,18,6])
    f_a=subplot(1,2,1);
    imagesc(x,y,amplitude);
    ylabel(ylable)
    xlabel(xlable)
    ax = gca;
    ax.TickDir = 'in';
    ax.YDir="normal";
    set(gca,'FontName','Arial','FontSize',8,'LineWidth',1);
    colormap(f_a,gray);
    clim([0,1])
    c = colorbar;
    c.Box="on";
    c.FontSize = 8;
    c.LineWidth=1;
    
    title(title1)
    
    f_p=subplot(1,2,2);
    imagesc(x,y,phase);
    ylabel(ylable)
    xlabel(xlable)
    ax = gca;
    ax.TickDir = 'in';
    ax.YDir="normal";
    set(gca,'FontName','Arial','FontSize',8,'LineWidth',1);
    colormap(f_p,addcolorplus(312).*repmat((1-linspace(-1,1,64).^4)',[1,3]));
    colormap(f_p,(addcolorplus(319)));
    colormap(f_p,(addcolorplus(302)));
    %clim([0,pi])
    c = colorbar;
    c.Box="on";
    c.FontSize = 8;
    c.LineWidth=1;

    title(title2)
end