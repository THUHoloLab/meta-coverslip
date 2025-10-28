function Draw_single_profile(x,y,profile,title1,colormapflag,xlable,ylable)

    figure()
    imagesc(x,y,profile);
    xlabel(xlable)
    ylabel(ylable)
    ax = gca;
    ax.TickDir = 'in';
    ax.YDir= 'normal';
    set(gca,'FontName','Arial','FontSize',8,'LineWidth',1);
    %ax.XTick=[];ax.YTick=[];
    c = colorbar;
    c.Box="on";
    c.FontSize = 8;
    c.LineWidth=1;
    switch colormapflag
        case 'phase'
            colormap(addcolorplus(312).*repmat((1-linspace(-1,1,64).^4)',[1,3]));
 %           c.Limits=[-pi pi];
        case 'amplitude'
            colormap(flip(addcolorplus(272),1));
        case 'real'
            lim=max(max(max(max(abs(profile)))));
            colormap(addcolorplus(299));
%             c.Limits=[-lim lim];
    end
    title(title1)
end
