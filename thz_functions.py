"""
Contains functions that are define in Dr. Thomas Chiou's THzProc code. These functions were put into
this file so that they can be called by another program.

Function Descriptions:
  Remap -
      The data that is output by the TeraView Software does not have the data points aligned
      correctly. Remap takes the waveform that is output by the TeraView software and realigns the
      data points to produce a cleaner image.
"""

import numpy as np
import sys


def ReMap(waveform,X,Y,Xmax,Xmin,Ymin,Xres,Yres,Xstep,Ystep,ScanType,axis1,wavlen,wavtlen,XCorTol,
          XDiffTol,YDiffTol,XChkTol,tiny,TrendOff,nlim):

# given the non-uniform X (or S, theata) cooirdinates, remap to a uniform rectangular grid
# 1st method: remap within each line by interpolation
# I/O: transformed from 1-D array "waveform" on non-uniform grid at 1-D X and Y coord. arrays for each element (scan position)
#      of "waveform" into 2-D "WaveformNew" on uniform grid at much shorter 1-D Xnew and Ynew arrays only designated to common
#      coord.
# handle both bi- and uni-directional scans

# CAN'T RELY ON TVL'S Xstep !!!!! A major bug found! Tvl reported correct Xstep (most time?), but in some larger scans,
# the actual data points on a X line came out short by as much as 5% (although the scan did reach the desired physical scan
# position). Has to do time-consuming check on every data point (the old way takes a shortcut to check only around the estimated
# ends)!  DEC 2012

# the angular coord. of rotational scan was found incorrect also.  The actual pyhysic position seemed to be right (had tested) and
# the number of scan points are also fine, but the angular coords. stored in data file are off significantly. Before Teraview fixes
# it, the temp.remedy is to map the max. and min. of inccorect coords. to the desired range and scale all coords. accordingly.
# FEB 2013

# now use true Xres and Yres.  Fix the 'middle points' at the ends. It seems the Y coord. and X spacing in the central 'stable' portion
# of a line are accurate to 0.01mm on average.  X coords. are generally off due to backlash.  These observations are based on the 6 PW
# TBC tablet data 126 X 26 @0.2mm, 30ps, 4096pt, 1avg.
# 26AUG2013

# fix flyback and rotational scan and np.max -> np.amax
# 22SEP2013

# add baseline trend removal using waveforms around edges
# 15NOV2013

# add more baseline removal option
# 01JAN2016

# last update: 01JAN2016

  # Xres and Yres are true values set by ScanAcquire
  XChkLen=int(Xstep/4) ; tol=10 ; #print Xstep
  XCor=XCorTol*Xres ; YDiff=Yres*YDiffTol ; XDiff=Xres*XDiffTol ; pos=[] ; Ynew=[] #Ynew=np.zeros(Ystep,dtype='<f')
  # ibeg=(1 if np.abs(Y[0]-Y[1])>YDiff else 0) # see if the 1st scan position is an outlier <--- this is not enough; replaced by below

  datlen=len(waveform)
# for rotational scan, just inspect Y coord. of initial points if X[0] is off too much; throw them away if they are way out of line
  if axis1=='Turntable' and np.abs(X[0]-Xmin)>5.*XDiff:
    for i in range(datlen):
      if np.abs(Y[i]-Ymin)<YDiff:
        ibeg=i
        print 'rotational scan starting point:',i
        break
  else:
# for linear scan, first inspect both coords. of initial points; throw them away if they are way out of line   15DEC2012 add
# (e.g. staring 2 Ys of "R-R 30492E blade 1 mil purge scan 2 HD (50mm 40ps 2048pt 1avg).tvl" were way off)
    for i in range(datlen):
      if np.abs(Y[i]-Ymin)<YDiff and np.abs(X[i]-Xmin)<XDiff:
        ibeg=i
        break

# next find coordinate info and make a slight corection to backlash for bi-directional scan
  i=1 ; j=ibeg ; nline=0 ; Ynow=Ymin ; Ynext=Ymin+Yres
  #print ibeg,Xstep,XChkLen,j+Xstep-XChkLen,j+Xstep+tol
  #print 'datalen', datlen
  while (j<datlen):
    for k in range(j+Xstep-XChkLen,j+Xstep+tol): # jump to check last XChkLen pts.
      if np.abs(Y[k]-Ynow)>YDiff:
        # print i,k,Y[k], Ynow,YDiff
        nline=k-ibeg
        j=k-1
        break
                          #print 'nline=',nline
    if nline<nlim*Xstep: print 'line ',i,'has ',nline,' data point; too short!' # give a warning for short line which is likely problem in the data
    # compute the avg Y of the lcentral portion of the line; will use it for the new "map"
    tmp=np.sum(Y[ibeg+XChkLen:j-XChkLen])/float(j-2*XChkLen-ibeg) ; Ynew.append(tmp)
    if ScanType=='2D Image Scan with encoder': # make the correction, even the pulse-on position is not too bad given the backlash
      X[ibeg:j]=X[ibeg:j]+(XCor if np.mod(i,2)==0 else -XCor)
    pos.append([nline,X[ibeg],Y[ibeg],ibeg,X[j],Y[j],j,tmp,Ynow,Ynow-tmp])
    for k in range(j+1,j+tol): # find the starting point of a new line
      if np.abs(Y[k]-Ynext)<YDiff:  # skip as many "middle points" as needed
        ibeg=k
        j=ibeg
        nline=0
        Ynow=Ynext
        i=i+1
        Ynext=Ymin+float(i)*Yres
        break
    if i==Ystep:
      nline=datlen-j+1
      tmp=np.sum(Y[ibeg+XChkLen:datlen-XChkLen])/float(datlen-2*XChkLen-ibeg) ; Ynew.append(tmp)
      if ScanType=='2D Image Scan with encoder': # make the correction, even the pulse-on position is not too bad given the backlash
        X[ibeg:]=X[ibeg:]+(XCor if np.mod(i,2)==0 else -XCor)
      pos.append([nline,X[ibeg],Y[ibeg],ibeg,X[-1],Y[-1],datlen,tmp,Ynow,Ynow-tmp])
      break

  Ynew=np.array(Ynew,dtype='<f')

                    #for k in range(len(pos)):
                    #  print k,pos[k]

  if i !=Ystep:
    print '!!! Ystep is wrong! should be',i,'!!!'  # this is fatal error: the steps in Y found should agree with Ystep 99.99%
    Ystep=i
  Xstep=np.sum([pos[i][0] for i in range(Ystep)])/Ystep
  print 'avg Xstep=',Xstep, ' as new Xstep' # this is the new Xstep to be used from this point on

  print 'max. Y coord. deviation=',np.amax([pos[i][-1] for i in range(Ystep)]),'mm'

# then determine the X bound for a smaller, trimmed grid

  if ScanType=='2D Image Scan with encoder': #bi-directional
                 #if np.sign(pos[0][1])!=np.sign(pos[1][1]) or \
                 #  (np.sign(pos[0][1])==np.sign(pos[1][1]) and np.abs(pos[0][1]-pos[1][1])>Xres): # test bi-directional or unidirectional
                         #print [pos[i][1] for i in range(0,Ystep,2)]+[pos[i][4] for i in range(1,Ystep,2)]
                         #print [pos[i][4] for i in range(0,Ystep,2)]+[pos[i][1] for i in range(1,Ystep,2)]
    Xbeg=np.amax([pos[i][1] for i in range(0,Ystep,2)]+[pos[i][4] for i in range(1,Ystep,2)])
    Xend=np.min([pos[i][4] for i in range(0,Ystep,2)]+[pos[i][1] for i in range(1,Ystep,2)])
  elif ScanType=='Flyback 2D scan': # unidirectional
    Xbeg=np.amax([pos[i][1] for i in range(Ystep)])
    Xend=np.min([pos[i][4] for i in range(Ystep)]) # was Xbeg=np.max([pos[:][1]), etc. Got Wrong number!
  else:
    print "unknown scan type!"
    sys.exit(1)

# correct angular coords. of rotational scan: map the current wrong coords. closer to the (Xmin,Xmax) size
# enlarge the size a bit (by adding tiny at each end) to allow later interpolation possible for all new positions
  scale=(Xmax-Xmin+2.*tiny)/(Xend-Xbeg)
  if axis1=='Turntable' and (scale>1.1 or scale<0.9):
    X[:]=(X[:]-Xbeg)*scale+Xmin
    print 'new Xbeg=Xmin, Xend=Xmax'
    Xnew=np.linspace(Xmin,Xmax,Xstep)
  else:
    print 'new Xbeg,Xend=',Xbeg+tiny,Xend-tiny,'mm which is',(Xend-Xbeg-2.*tiny),'mm long comparing with desired ',Xmax-Xmin,'mm'
    Xnew=np.linspace(Xbeg+tiny,Xend-tiny,Xstep) # add tiny to make sure all new allocated points are inside bounds

  WaveformNew=np.zeros((Ystep,Xstep,wavlen),dtype='<f')
  CscanAmpNew=np.zeros((Ystep,Xstep),dtype='<f')

# remap each line by interpolation -----------------------------------------------------------------------------------------------------------------
# WARNING for rotational scan! Xs in pos list are now incorrect (still the values beofore corrections)
  if ScanType=='2D Image Scan with encoder':
    for i in range(0,Ystep,2):
      b=pos[i][3] ; e=pos[i][6]
      for j in range(Xstep):
                            #print 'forward b,e',b,e
        for k in range(b,e):
          if Xnew[j]>=X[k] and Xnew[j]<X[k+1]:
                           #if i==78: print 'i,j,k,Xnew[j],X[k],X[k+1]',i,j,k,Xnew[j],X[k],X[k+1]
            w1=np.abs(Xnew[j]-X[k]) ; w2=np.abs(X[k+1]-Xnew[j]) ; w=w1+w2
            WaveformNew[i,j,:]=(w2/w)*waveform[k][:]+(w1/w)*waveform[k+1][:]
            b=k
            break
    for i in range(1,Ystep,2): # reverse direction
      b=pos[i][3] ; e=pos[i][6]
      for j in range(Xstep-1,-1,-1):
                          #print ' reverse b,e',b,e
        for k in range(b,e):
                          #if i==79 and j==0: print 'b,e,i,j,k,Xnew[j],X[k],X[k+1]',b,e,i,j,k,Xnew[j],X[k],X[k+1]
          if Xnew[j]<X[k] and Xnew[j]>=X[k+1]:
                          #if i==79: print 'i,j,k,Xnew[j],X[k],X[k+1]',i,j,k,Xnew[j],X[k],X[k+1]
            w1=np.abs(X[k]-Xnew[j]) ; w2=np.abs(Xnew[j]-X[k+1]) ; w=w1+w2
            WaveformNew[i,j,:]=(w2/w)*waveform[k][:]+(w1/w)*waveform[k+1][:]
            b=k
            break

  elif ScanType=='Flyback 2D scan':
    for i in range(Ystep):
      b=pos[i][3] ; e=pos[i][6]
      for j in range(Xstep):
        for k in range(b,e):
          if Xnew[j]>=X[k] and Xnew[j]<X[k+1]:
            w1=Xnew[j]-X[k] ; w2=X[k+1]-Xnew[j] ; w=w1+w2
            WaveformNew[i,j,:]=(w2/w)*waveform[k][:]+(w1/w)*waveform[k+1][:]
            b=k
            break
  else:
    print "unknown scan type!"
    sys.exit(1)

# remove baseline trend in apertured near-field -------------------------------------------------------------------------------------------------

  # average several "blank" waveforms (i.e. not on the sample) at beginning and end of row as reference to be substracted from the sample waveforms
  # assume there are at least some (1mm or more preferred) "blank" at beginning and end of row and X spacing is sufficiently smaller (0.05mm preferred)
  if TrendOff==5:
    refwav=np.zeros(wavlen,dtype='<f')
    if Xres<0.11:
      print '!!! there are at least 1mm blank at beginning and end of row, or baseline removal may not work properly !!!'
    if Xres>0.11: #mm
      for i in range(Ystep):
        refwav[:]=(WaveformNew[i,1,:]+WaveformNew[i,-1,:])/2. # take 2nd points, as first may deviate from the correct locations
        for j in range(Xstep):
          WaveformNew[i,j,:]=WaveformNew[i,j,:]-refwav[:]
    elif Xres<0.11 and Xres>0.051:
      for i in range(Ystep):
        refwav[:]=(WaveformNew[i,3,:]+WaveformNew[i,-3,:]+WaveformNew[i,7,:]+WaveformNew[i,-7,:])/4.
        for j in range(Xstep):
          WaveformNew[i,j,:]=WaveformNew[i,j,:]-refwav[:]
    elif Xres<0.051:
      for i in range(Ystep):
        refwav[:]=(WaveformNew[i,5,:]+WaveformNew[i,-5,:]+WaveformNew[i,10,:]+WaveformNew[i,-10,:]+WaveformNew[i,15,:]+WaveformNew[i,-15,:])/6.
        for j in range(Xstep):
          WaveformNew[i,j,:]=WaveformNew[i,j,:]-refwav[:]

  # use the avg (option 5) waveform of each row as reference; aligned each waveform in that row with the rescaled (option 2) reference
  # probably the best option of 2nd gen
  elif TrendOff==-1:
    refwav=np.zeros(wavlen,dtype='<f')
    early=np.zeros(100,dtype=int) ; late=np.zeros(100,dtype=int) ; total=float(Xstep*Ystep)
    if Xres<0.11:
        print '!!! there are at least 1mm blank at beginning and end of row, or baseline removal may not work properly !!!'
    for i in range(Ystep):
      if Xres>0.11: #mm
        refwav[:]=(WaveformNew[i,1,:]+WaveformNew[i,-1,:])/2. # take 2nd points, as first may deviate from the correct locations
      elif Xres<0.11 and Xres>0.051:
        refwav[:]=(WaveformNew[i,3,:]+WaveformNew[i,-3,:]+WaveformNew[i,7,:]+WaveformNew[i,-7,:])/4.
      elif Xres<0.051:
        refwav[:]=(WaveformNew[i,5,:]+WaveformNew[i,-5,:]+WaveformNew[i,10,:]+WaveformNew[i,-10,:]+WaveformNew[i,15,:]+WaveformNew[i,-15,:])/6.
      H=np.argmax(refwav[0:wavlen/3]) ; L=np.argmin(refwav[0:wavlen/3]) ; refctr=(H+L)/2 ; Vpprefctr=refwav[H]-refwav[L]
      for j in range(Xstep):
        H=np.argmax(WaveformNew[i,j,0:wavlen/3]) ; L=np.argmin(WaveformNew[i,j,0:wavlen/3]) ; ctr=(H+L)/2
        Vppctr=WaveformNew[i,j,H]-WaveformNew[i,j,L] ; fac=Vppctr/Vpprefctr
        if refctr<ctr:
          shift=ctr-refctr
          late[shift]+=1
          WaveformNew[i,j,shift:]=WaveformNew[i,j,shift:]-fac*refwav[0:-shift] ; WaveformNew[i,j,0:shift]=WaveformNew[i,j,shift]
        else:
          shift=refctr-ctr
          early[shift]+=1
          WaveformNew[i,j,0:wavlen-shift]=WaveformNew[i,j,0:wavlen-shift]-fac*refwav[shift:]
          WaveformNew[i,j,wavlen-shift:]=WaveformNew[i,j,wavlen-shift-1]

  # use one waveform of each row as reference; aligned each waveform in that row with the reference at mid pt between max. and min. peaks in the first 1/3 of waveform
  # this is the best option of 1st gen
  elif TrendOff==1:
    refwav=np.zeros(wavlen,dtype='<f')
    early=np.zeros(100,dtype=int) ; late=np.zeros(100,dtype=int) ; total=float(Xstep*Ystep)
    for i in range(Ystep):
      refwav[:]=WaveformNew[i,1,:]
      H=np.argmax(refwav[0:wavlen/3]) ; L=np.argmin(refwav[0:wavlen/3]) ; refctr=(H+L)/2
      for j in range(Xstep):
        H=np.argmax(WaveformNew[i,j,0:wavlen/3]) ; L=np.argmin(WaveformNew[i,j,0:wavlen/3]) ; ctr=(H+L)/2
        if refctr<ctr:
          shift=ctr-refctr
          late[shift]+=1
          WaveformNew[i,j,shift:]=WaveformNew[i,j,shift:]-refwav[0:-shift] ; WaveformNew[i,j,0:shift]=WaveformNew[i,j,shift]
        else:
          shift=refctr-ctr
          early[shift]+=1
          WaveformNew[i,j,0:wavlen-shift]=WaveformNew[i,j,0:wavlen-shift]-refwav[shift:]
          WaveformNew[i,j,wavlen-shift:]=WaveformNew[i,j,wavlen-shift-1]


  elif TrendOff==2: # also rescale refwav to have same Vpp as each waveform
    refwav=np.zeros(wavlen,dtype='<f')
    early=np.zeros(100,dtype=int) ; late=np.zeros(100,dtype=int) ; total=float(Xstep*Ystep)
    for i in range(Ystep):
      refwav[:]=WaveformNew[i,1,:]
      H=np.argmax(refwav[0:wavlen/3]) ; L=np.argmin(refwav[0:wavlen/3])
      refctr=(H+L)/2 ; Vpprefctr=refwav[H]-refwav[L]
      for j in range(Xstep):
        H=np.argmax(WaveformNew[i,j,0:wavlen/3]) ; L=np.argmin(WaveformNew[i,j,0:wavlen/3])
        ctr=(H+L)/2 ; Vppctr=WaveformNew[i,j,H]-WaveformNew[i,j,L]
        fac=Vppctr/Vpprefctr # rescale refwav to have same Vpp as each waveform (assuming there is a system amp variation during scan)
        if refctr<ctr:
          shift=ctr-refctr
          late[shift]+=1
          WaveformNew[i,j,shift:]=WaveformNew[i,j,shift:]-fac*refwav[0:-shift] ; WaveformNew[i,j,0:shift]=WaveformNew[i,j,shift]
        else:
          shift=refctr-ctr
          early[shift]+=1
          WaveformNew[i,j,0:wavlen-shift]=WaveformNew[i,j,0:wavlen-shift]-fac*refwav[shift:]
          WaveformNew[i,j,wavlen-shift:]=WaveformNew[i,j,wavlen-shift-1]

  elif TrendOff==3 or TrendOff==4:

    time,tmp,n,delt,Tdat=open_asn_dat2(basedir,refname)
    if abs(Tdat-wavtlen)>0.2:
      print 'reference waveform time length not agree with data!'
      sys.exit(1)
    refwav=np.zeros(wavlen,dtype='<f')
    if n==wavlen:
      refwav[:]=tmp[:]
    elif n==2*wavlen:
      refwav[:]=tmp[::2]
    else:
      print 'invalid length of reference waveform for baseline removal!'
      sys.exit(1)
    early=np.zeros(50,dtype=int) ; late=np.zeros(50,dtype=int) ; total=float(Xstep*Ystep)
    H=np.argmax(refwav[0:wavlen/3]) ; L=np.argmin(refwav[0:wavlen/3])
    refctr=(H+L)/2 ; Vpprefctr=refwav[H]-refwav[L]

    if TrendOff==3:
      for i in range(Ystep):
        for j in range(Xstep):
          H=np.argmax(WaveformNew[i,j,0:wavlen/3]) ; L=np.argmin(WaveformNew[i,j,0:wavlen/3]) ; ctr=(H+L)/2
          if refctr<ctr:
            shift=ctr-refctr
            late[shift]+=1
            WaveformNew[i,j,shift:]=WaveformNew[i,j,shift:]-refwav[0:-shift] ; WaveformNew[i,j,0:shift]=WaveformNew[i,j,shift]
          else:
            shift=refctr-ctr
            early[shift]+=1
            WaveformNew[i,j,0:wavlen-shift]=WaveformNew[i,j,0:wavlen-shift]-refwav[shift:]
            WaveformNew[i,j,wavlen-shift:]=WaveformNew[i,j,wavlen-shift-1]

    elif TrendOff==4:
      for i in range(Ystep):
        for j in range(Xstep):
          H=np.argmax(WaveformNew[i,j,0:wavlen/3]) ; L=np.argmin(WaveformNew[i,j,0:wavlen/3])
          ctr=(H+L)/2 ; Vppctr=WaveformNew[i,j,H]-WaveformNew[i,j,L]
          fac=Vppctr/Vpprefctr # rescale refwav to have same Vpp as each waveform
          if refctr<ctr:
            shift=ctr-refctr
            late[shift]+=1
            WaveformNew[i,j,shift:]=WaveformNew[i,j,shift:]-fac*refwav[0:-shift] ; WaveformNew[i,j,0:shift]=WaveformNew[i,j,shift]
          else:
            shift=refctr-ctr
            early[shift]+=1
            WaveformNew[i,j,0:wavlen-shift]=WaveformNew[i,j,0:wavlen-shift]-fac*refwav[shift:]
            WaveformNew[i,j,wavlen-shift:]=WaveformNew[i,j,wavlen-shift-1]

  if TrendOff!=0 and TrendOff!=5:
    print
    print 'stat of baseline removal'
    print 'shift   late     %     no/early    %'
    for i in range(50):
      if late[i]!=0 or early[i]!=0:
        print '%2d %10d %6.3f %10d %6.3f' %(i, late[i],100.*(late[i]/total),early[i],100.*(early[i]/total))

# Xstep,Ystep may be updated on return. CscanAmpNew is created but contains just zeros
  return WaveformNew,CscanAmpNew,Xnew,Ynew,pos,Xstep,Ystep


def AmpCor300(TLL,TLR,ALL,ALR,TRL,TRR,ARL,ARR,p,wavlen,delt,Xstep,Ystep,WaveformNew):

# fit A=a*T**b on both ends to remove excessive amplitude amplification for 300 ps waveforms
# last update: 08JUN2015

  nLL=int(TLL/delt) ; nLR=int(TLR/delt)
  Tl=np.linspace(nLL*delt,nLR*delt,nLR-nLL+1)
  KL=np.power(ALL/ALR,1./p)
  betaL=TLR+(TLR-TLL)/(KL-1.)
  alphaL=ALL/(betaL-TLL)**p
  Al=alphaL*(betaL-Tl)**p

  nRL=int(TRL/delt) ; nRR=int(TRR/delt) ; print 'nLL,nRR',nLL,nRR
  Tr=np.linspace(nRL*delt,nRR*delt,nRR-nRL+1)
  KR=np.power(ARL/ARR,1./p)
  betaR=TRR-(TRL-TRR)/(KR-1.)
  alphaR=ARL/(betaR-TRL)**p
  Ar=alphaR*(betaR-Tr)**p

  #plt.figure()
  #plt.plot(WaveformNew[10,20,:])

  if nRR==wavlen:
    nRR-=1  # prevent index out of range
  for i in range(Ystep):
    for j in range(Xstep):
       WaveformNew[i,j,nLL:nLR+1]=WaveformNew[i,j,nLL:nLR+1]/Al[0:]
       WaveformNew[i,j,nRL:nRR+1]=WaveformNew[i,j,nRL:nRR+1]/Ar[0:]


def FindPeaks(waveform,Xstep,Ystep,wavlen,nHalfPulse,fthres,BinRange,PulseLen,FollowGateOn):

# given list BinRange, find peak(s) and gate(s) in wavform in terms of bin locations to PeakBin[i,npeak,Ystep,Xstep],
# i=0: pos. peak, =1 neg., =2 half way, 3=left gate, 4=right gate

# BinRange[0][0-1] are for the leading peak, ususally FSE, BinRange[1][0-1] is for 1st layer, etc.
# assume max. and min. peaks vary within 2*nHalfPulse width and individual pulses are bound within their corresponding BinRange
# future improvement: use of peakdet

# last update: 29DEC2015


  npeak=len(BinRange)
  PeakBin=np.zeros((5,npeak,Ystep,Xstep),dtype=np.int16)

  # if FollowGateOn>0, BinRange[0][0-1] are used as fixed gate following pos. peak of leading pulse (usually FSE).
  # all trailing gates are related to this pos. peak

  if FollowGateOn>0:

    if npeak<2: # must have at least 2 peaks (including the lead peak to be followed)
      print "incorrect BinRange setting!"
      sys.exit(1)
    for i in range(Ystep):
      for j in range(Xstep):
        PeakBin[0,0,i,j]=np.argmax(waveform[i,j,BinRange[0][0]:BinRange[0][1]+1])+BinRange[0][0] # important! need to add BinRange[0][0] to argmax result
        # check more carefully to see if FSE is smaller (due to defocusing, etc) than the largest peak in the gate
        if PeakBin[0,0,i,j]-nHalfPulse-BinRange[0][0] > 0: # in case PeakBin[0,0,i,j] too early
          itmp=(np.where(waveform[i,j,BinRange[0][0]:PeakBin[0,0,i,j]-nHalfPulse]>fthres*waveform[i,j,PeakBin[0,0,i,j]]))
          if len(itmp[0])!=0: # see if any peak above the threshold before maxfloc[i,j]
            itmp2=itmp[0][0]+BinRange[0][0] # np.where returns tuple in itmp
            itmp2=np.argmax(waveform[i,j,itmp2:itmp2+nHalfPulse])+itmp2
            PeakBin[0,0,i,j]=itmp2
        # find the neg. peak within pulse width=2*nHalfPulse
        L=PeakBin[0,0,i,j]-nHalfPulse ; L=(L if L>0 else 0)
        R=PeakBin[0,0,i,j]+nHalfPulse ; R=(R if R<=wavlen else wavlen)
        PeakBin[1,0,i,j]=np.argmin(waveform[i,j,L:R])+L
        PeakBin[2,0,i,j]=(PeakBin[0,0,i,j]+PeakBin[1,0,i,j])/2
        PeakBin[3,0,i,j]=BinRange[0][0] ; PeakBin[4,0,i,j]=BinRange[0][1]
        # all trailing gates are then shifted wrt the pos. leading peak, PeakBin[0,0,i,j]
        for k in range(1,npeak):
          L=PeakBin[0,0,i,j]+BinRange[k][0] ; R=PeakBin[0,0,i,j]+BinRange[k][1]+1
          if L<0:
            L=0
          elif L>wavlen:
            print "incorrect left gate setting in ",k,"th gate!"
            sys.exit(1)
          if R<L:  #corrected 15SEP2013
            print "incorrect right gate setting in ",k,"th gate!"
            sys.exit(1)
          elif R>wavlen:
            R=wavlen
          PeakBin[3,k,i,j]=L ; PeakBin[4,k,i,j]=R
          PeakBin[0,k,i,j]=np.argmax(waveform[i,j,L:R])+L
          if PulseLen<0.: # 29DEC2015: enforce searcg of neg. peak within original gate
            L2=L ; R2=R
          else:
            LL=int(PulseLen*nHalfPulse)
            if LL<1:  LL=1 # 29DEC2015: to prevent zero-length gate
            L2=PeakBin[0,k,i,j]-LL ; L2=(L2 if L2>L else L)
            R2=PeakBin[0,k,i,j]+LL ; R2=(R2 if R2<=wavlen else wavlen)
          PeakBin[1,k,i,j]=np.argmin(waveform[i,j,L2:R2])+L2
          PeakBin[2,k,i,j]=(PeakBin[0,k,i,j]+PeakBin[1,k,i,j])/2

  else: # =0: the whole waveform (the BinRange is given only [0,wavlen] in main)

    for i in range(Ystep):
      for j in range(Xstep):
        PeakBin[0,0,i,j]=np.argmax(waveform[i,j,BinRange[0][0]:BinRange[0][1]+1])+BinRange[0][0]
        L2=PeakBin[0,0,i,j]-nHalfPulse ; L2=(L2 if L2>BinRange[0][0] else BinRange[0][0])
        R2=PeakBin[0,0,i,j]+nHalfPulse ; R2=(R2 if R2<=BinRange[0][1] else BinRange[0][1])
        PeakBin[1,0,i,j]=np.argmin(waveform[i,j,L2:R2])+L2
        PeakBin[2,0,i,j]=(PeakBin[0,0,i,j]+PeakBin[1,0,i,j])/2
        PeakBin[3,0,i,j]=BinRange[0][0] ; PeakBin[4,0,i,j]=BinRange[0][1]

  return PeakBin