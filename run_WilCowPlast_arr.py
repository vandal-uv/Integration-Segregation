# -*- coding: utf-8 -*-.
"""
The Huber_braun neuronal model function.

@author: porio
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from anarpy.models import netWilsonCowanPlastic as WC
from anarpy.utils.FCDutil import fcd
# from utils import Networks
import os, sys

rank=int(os.environ['SLURM_ARRAY_TASK_ID'])
threads=int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1

# filename = 'powerLaw18/01-powerLaw-18-C0'
# filename = 'sw-18/06-smallworld-18-p0.15'
# net = np.loadtxt(f"../newNets/{filename}.txt")

if len(sys.argv)>1:
    seed=int(sys.argv[1])
else:
    seed=None

if len(sys.argv)>2:
    netFolder=sys.argv[2] + '/'
else:
    netFolder="sw-18/"

if not os.path.isdir(netFolder) and rank==0:
    os.mkdir(netFolder)

nets=os.listdir('../newNets/'+netFolder)

WC.tTrans=50  #transient removal with accelerated plasticity
WC.tstop=102   # actual simulation

WC.G=0.2 #Connectivity strength
WC.D=0.002        #noise factor
WC.rhoE=0.125   #target value for mean excitatory activation
WC.dt=0.002   # Sampling interval

# P=.4
Gvals=np.logspace(-2,0,11)
Gvals=np.insert(Gvals,0,0)

combis=[(a,b) for a in Gvals for b in nets]

#%%

for j,(G,net) in enumerate(combis):    
    filename = netFolder + net.replace('.txt',f'-s{seed}-D002')
    i=np.where(Gvals==G)[0][0]
    
    if j%threads == rank and not os.path.isfile(f'{filename}/{i:02d}-FCD.png'):
        
        WC.CM = np.loadtxt(f"../newNets/{netFolder}{net}")
        nnodes=len(WC.CM)
        WC.N=nnodes
        
        np.random.seed(seed)
        WC.P=np.random.uniform(0.3,0.5,nnodes)

        if not os.path.isdir(filename) and G==0:
            tree=filename.split('/')
            curr_path='.'
            for branch in tree:
                curr_path = curr_path + '/' + branch
                if not os.path.isdir(curr_path):
                    os.mkdir(curr_path)
        
        WC.G = G
        Vtrace,time=WC.Sim(verbose=True)    
        
        # These lines can be adapted to write the parameters to a text file
        # print(str(WC.ParamsNode()).replace(", '","\n '"))
        # print(str(WC.ParamsSim()).replace(", '","\n '"))
        # print(str(WC.ParamsNet()).replace(", '","\n '"))
               
        E_t=Vtrace[:,0,:]
        
        #%%
        # spec=np.abs(np.fft.fft(E_t-np.mean(E_t,0),axis=0))
        # freqs=np.fft.fftfreq(len(E_t),WC.dt)
        freqs,spec=signal.welch(E_t,fs=1/WC.dt, nperseg=1000, nfft=4000,noverlap=100,axis=0)
        
        b,a=signal.bessel(4,[5*2*WC.dt, 15*2*WC.dt],btype='bandpass')
        E_filt=signal.filtfilt(b,a,E_t,axis=0)
        
        analytic=signal.hilbert(E_filt,axis=0)
        
        remove_time = 1  # seconds at beggining and end
        remove_samples = int(remove_time/WC.dt)
        
        Trun = WC.tstop - 2*remove_time
        
        
        envelope=np.abs(analytic[remove_samples:-remove_samples,:])
        phase = np.angle(analytic[remove_samples:-remove_samples,:])
        
        FC=np.corrcoef(envelope,rowvar=False)
        
        FCphase=fcd.phaseFC(phase)
        #%%
                
        phasesynch=np.abs(np.mean(np.exp(1j*phase),1))
        MPsync=np.mean(phasesynch)  #Media de la fase en el tiempo
        VarPsync=np.var(phasesynch)  #Varianza de la fase en el tiempo
        
        outfile=f'{filename}/{i:02d}-table.txt'
        with open(outfile,'w') as outf:
            outf.write("# g,Psync,Meta,varFCD1ph,visc,varFCD1env,visc,varFCD2ph,visc,varFCD2000env,visc\n")
            outf.write(f"{G:.3g},{MPsync},{VarPsync}")
        
        
        plt.figure(104,figsize=(10,10))
        plt.clf()
            
        plt.subplot2grid((5,5),(0,0),rowspan=1,colspan=5)
        plt.plot(time[:-remove_samples*2], phasesynch)
        plt.title('mean P sync')
          
        plt.subplot2grid((5,5),(2,4))
        plt.imshow(FC,cmap='seismic',vmax=1,vmin=-1,interpolation='none')
        plt.gca().set_xticks(())
        plt.gca().set_yticks(())
        plt.title('Static FC - envel')

        plt.subplot2grid((5,5),(3,4))
        plt.imshow(FCphase,cmap='jet',vmax=1,vmin=0,interpolation='none')
        plt.gca().set_xticks(())
        plt.gca().set_yticks(())
        plt.title('Static FC - phase')
            
        plt.subplot2grid((5,5),(1,4))
        plt.imshow(WC.CM,cmap='gray_r')
        plt.title('SC')
        
        
        WW =2000
            
        FCD,Pcorr,shift=fcd.extract_FCD(phase.T,maxNwindows=2000,wwidth=WW,olap=0.75,
                                        mode='psync',modeFCD='clarksondist')
        
        varFCD = np.var(FCD[np.triu_indices(len(FCD),k=4)])
        viscosity = np.mean(np.diagonal(FCD,4))
        with open(outfile,'a') as outf:
            outf.write(f",{varFCD},{viscosity}")
            
        plt.subplot2grid((5,5),(1,0),rowspan=2,colspan=2)
        plt.imshow(FCD,vmin=0,extent=(0,Trun,Trun,0),interpolation='none',cmap='jet')
        plt.title(f'FCD phase W{WW}')
        plt.colorbar()
        
        windows=[int(len(Pcorr)*f) for f in (0.18, 0.36, 0.54, 0.72, 0.9)]
        axes2=[plt.subplot2grid((5,5),(3,pos)) for pos in range(5)]
        for axi,ind in zip(axes2,windows):
            corrMat=np.zeros((nnodes,nnodes))
            corrMat[np.tril_indices(nnodes,k=-1)]=Pcorr[ind]
            corrMat+=corrMat.T
            corrMat+=np.eye(nnodes)
                
            axi.imshow(corrMat,vmin=0,vmax=1,interpolation='none',cmap='jet')
                
            axi.set_xticks(())
            axi.set_yticks(())
            
            axi.set_title('t=%.4g'%(ind*Trun/len(Pcorr)))
            axi.grid()
            
        FCDfase=FCD.copy()
        Pcorrfase=Pcorr.copy()
        
        FCD,Pcorr,shift=fcd.extract_FCD(envelope.T,maxNwindows=2000,wwidth=WW,olap=0.75,
                                        mode='corr',modeFCD='euclidean')
        
        varFCD = np.var(FCD[np.triu_indices(len(FCD),k=4)])
        viscosity = np.mean(np.diagonal(FCD,4))
        with open(outfile,'a') as outf:
            outf.write(f",{varFCD},{viscosity}")
            
        plt.subplot2grid((5,5),(1,2),rowspan=2,colspan=2)
        plt.imshow(FCD,vmin=0,extent=(0,Trun,Trun,0),interpolation='none',cmap='jet')
        plt.title(f'FCD envel W{WW}')
        plt.colorbar()
        

        windows=[int(len(Pcorr)*f) for f in (0.18, 0.36, 0.54, 0.72, 0.9)]
        axes2=[plt.subplot2grid((5,5),(4,pos)) for pos in range(5)]
        for axi,ind in zip(axes2,windows):
            corrMat=np.zeros((nnodes,nnodes))
            corrMat[np.tril_indices(nnodes,k=-1)]=Pcorr[ind]
            corrMat+=corrMat.T
            corrMat+=np.eye(nnodes)
                
            axi.imshow(corrMat,vmin=-1,vmax=1,interpolation='none',cmap='seismic')
                
            axi.set_xticks(())
            axi.set_yticks(())
            
            axi.set_title('t=%.4g'%(ind*Trun/len(Pcorr)))
            axi.grid()
        
            
            
        plt.tight_layout()
        plt.savefig(f'{filename}/{i:02d}-FCD.png', dpi=200)
            
        
        envel_file = f'{filename}/{i:02d}-envel2.5k.npy'
        np.save(envel_file, envelope[::20])
        
        rawope = E_t[remove_samples:-remove_samples,:]
        
        raw_file = f'{filename}/{i:02d}-raw5k.npy'
        np.save(raw_file, rawope[::10])
            
            # plt.grid()

        # np.savez(f'{filename}/{i:02d}-FCDdata.npz', FCDenvel = FCD, FCenvel = Pcorr,
        #         FCDphase = FCDfase, FCphase = Pcorrfase)
                
        # with open(outfile,'a') as outf:
        #     outf.write("\n")


#%%

# plt.figure(1,figsize=(10,8))
# plt.clf()
# plt.subplot(321)
# plt.plot(time,Vtrace[:,0,::4])
# plt.ylabel('E')

# plt.subplot(323)
# plt.plot(time,Vtrace[:,2,:])
# plt.xlabel('time')
# plt.ylabel('a_ie')

# plt.subplot(325)
# plt.loglog(freqs[:1200],spec[:1200,::4])
# plt.xlabel('frequency (Hz)')
# plt.ylabel('abs')


# plt.subplot(222)
# plt.imshow(WC.CM,cmap='gray_r')
# plt.title("structural connectivity")
# # plt.yticks((0,5,10,15))

# plt.subplot(224)
# plt.imshow(FC,cmap='BrBG',vmin=-1,vmax=1)
# plt.colorbar()
# plt.title("Envelope correlation (FC)")
# # plt.yticks((0,5,10,15))

# plt.subplots_adjust(hspace=0.3)
