def adjustDM(DM1, dDMo, DM2, dDM_i, COLindex, xlim=(-2,2), ylim=(-0.9,0.9), 
             quad=False, indx=None, xlabel='', ylabel='', bias1=0, bias2=0, single=False, dx=0.2):
    
    DMo  = DM1 + bias1
    DM_i = DM2 + bias2
    
    fig = py.figure(figsize=(12, 4), dpi=100)    
    fig.subplots_adjust(wspace=0, top=0.92, bottom=0.12, left=0.06, right=0.98)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
    ax1 = plt.subplot(gs[0])
    if not single:
        ax2 = plt.subplot(gs[1])
    from matplotlib.ticker import MultipleLocator
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    if not single:
        ax2.yaxis.set_major_locator(MultipleLocator(0.5))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
    _, y_ax =  set_axes(ax1, xlim, ylim, fontsize=14)
    y_ax.yaxis.set_major_locator(MultipleLocator(0.5))
    y_ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    if not single:
        _, y_ax =  set_axes(ax2, xlim, ylim, fontsize=14)
        y_ax.yaxis.set_major_locator(MultipleLocator(0.5))
        y_ax.yaxis.set_minor_locator(MultipleLocator(0.1))    
    dDM =  DMo - DM_i 
    dDM_e = np.sqrt(dDMo**2+dDM_i**2)
    
    v = np.linspace(xlim[0]+0.1, xlim[1]-0.1,200)
        
    if indx is None:
        indx = np.arange(len(dDM))
    
    ax1.plot(COLindex[indx], dDM[indx], '.', alpha=0.05, color='k')    
    
    if not quad:
        fit, cov = curve_fit(lineF, COLindex[indx], dDM[indx], sigma=dDM_e[indx])
        correction = (fit[0]*COLindex+fit[1])
        if not single:
            ax1.plot(v,fit[0]*v+fit[1], 'r--')
    else:
        fit, cov = np.polyfit(COLindex[indx], dDM[indx], 2, cov=True, w=1./dDM_e[indx])
        correction = (fit[0]*COLindex**2+fit[1]*COLindex+fit[2])
        if not single:
            ax1.plot(v,fit[0]*v**2+fit[1]*v+fit[2], 'r--')
    
    print "Fit params:"
    for i in range(len(fit)):
        print '%.3f'%fit[i]+'\pm'+'%.3f'%np.sqrt(cov[i][i])
    
    dDM_mod = dDM - correction
    if not single:
        ax2.plot(COLindex[indx], dDM_mod[indx], '.', alpha=0.05)
    

    ##############################
    X = COLindex[indx]
    Y = dDM_mod[indx]
    Y0 = dDM[indx]
    Ye = dDM_e[indx]
    for i in np.arange(xlim[0]+0.1, xlim[1]-0.1,dx):
        xp = []
        yp = []
        yp0 = []
        for ii in range(len(X)):
            xi = X[ii]
            if xi>=i and xi<i+dx:
                xp.append(xi)
                yp.append(Y[ii])
                yp0.append(Y0[ii])
        if len(xp)>0:
            if not single:
                ax2.errorbar(np.median(xp), np.median(yp), yerr=np.std(yp), xerr=np.std(xp), fmt='o', 
                        color='r', ms=6)     
            ax1.errorbar(np.median(xp), np.median(yp0), yerr=np.std(yp0), xerr=np.std(xp), fmt='o', 
                        color='r', ms=6)   
    ###############################    
    ax1.plot([-10,10], [0,0], 'k:')
    if not single:
        ax2.plot([-10,10], [0,0], 'k:')
    ###############################
    c = np.polyfit(X,Y, 2, w=1./Ye)
    ###############################
    
    if not single:
        Ylm = ax2.get_ylim() ; Xlm = ax2.get_xlim()
        x0 = 0.9*Xlm[0]+0.1*Xlm[1]
        y0 = 0.9*Ylm[0]+0.10*Ylm[1]
        RMS = np.std(Y)
        ax2.text(x0,y0, r"$RMS$" +": %.2f [mag]" % RMS, fontsize=16, color='k')    
        x0 = 0.13*Xlm[0]+0.87*Xlm[1]
        ax2.errorbar(x0, [-0.5], yerr=np.median(dDM_e),xerr=0.05*1.414, fmt='.', color='green', capsize=3)
    
    ax1.set_xlabel(xlabel, fontsize=18) 
    ax1.set_ylabel(r'$'+ylabel+'- DM_{i}$', fontsize=18) 
    if not single:
        ax2.set_xlabel(xlabel, fontsize=18) 
        ax2.set_ylabel(r'$C('+ylabel+') - DM_{i}$', fontsize=18) 
    

    
    plt.subplots_adjust(hspace=.0, wspace=0.25)
    
    if single:
        ax2=None
    
    return [ax1,ax2], DMo-correction, fit, cov
    
###############################    ###############################

