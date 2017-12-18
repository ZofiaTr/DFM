import acor
import numpy as np
import diffusionmap as dm


def postProcess(model, sampler, numberOfIterations, nrSteps, nrRep):

    algo=sampler.algorithmName
    algofile=algo

    compTime=sampler.timeAv.getAverage()*numberOfIterations

    totalNumberofSteps= numberOfIterations*nrSteps*nrRep

    Cwlck=compTime/totalNumberofSteps

    #save long traj
    #np.save(algofile+'_'+model.potentialFlag+'_traj', sampler.sampled_trajectory)

    tau, mean, sigma = acor.acor(sampler.sampled_trajectory[:,0]-np.mean(sampler.sampled_trajectory[:,0], axis=0))

    print('\n'+algo)
    print ('Time per step ' +repr(Cwlck))
    print ('Stat. error in wallclock time ' +repr(sigma/np.sqrt(compTime)))

    # file.write('Computational time is '+repr(compTime)+' s\n')
    #
    # file.write('Kinetic temperature is ' + repr(sampler.kineticTemperature.getAverage())+'\n' )
    # file.write('<X> is ' + repr(sampler.averagePosition.getAverage()) +'\n')
    #
    #
    #
    # file.write ('Autocorrelation time is ' +repr(tau)+'\n')
    # file.write ('Mean ' +repr(mean)+'\n')
    # file.write ('Sigma ' +repr(sigma)+'\n')
    # #statistical precision espilon=sigma^2/number_of_steps - in wallclock time esp=sigma^2/
    # file.write ('Stat. error in wallclock time ' +repr(sigma/np.sqrt(compTime))+'\n')
    #
    # file.close()

    return np.array([tau, mean, sigma, compTime, totalNumberofSteps, (sigma/np.sqrt(compTime))])
    #for i in range(0,6):
    #    results[algoIt, i]=resTmp[i]


def postProcessTemporary(traj, model, sampler, numberOfIterations, nrSteps, nrRep):

    algo=sampler.algorithmName
    algofile=algo

    compTime=sampler.timeAv.getAverage()*numberOfIterations

    totalNumberofSteps= traj.shape[0]

    Cwlck=compTime/totalNumberofSteps

    tau, mean, sigma = acor.acor(traj[:,0]-np.mean(traj[:,0], axis=0))

    print('\n'+algo)
    print ('Time per step ' +repr(Cwlck))
    print ('Stat. error in wallclock time ' +repr(sigma/np.sqrt(compTime)))


    return np.array([tau, mean, sigma, compTime, totalNumberofSteps, (sigma/np.sqrt(compTime))])



def compare_algorithms(results, samplers, idxRef):
    #sigma*comptime for std
    effStd=(results[idxRef,2]**2)*results[idxRef,3]

    strlen=np.zeros( results.shape[0])
    for i in range(0,results.shape[0]):
        strlen[i]=int(len(samplers[i].algorithmName))

    idx=np.argmax(strlen)

    maxl=np.max(strlen)


    print('\n*************************\n')

    print('\n'+ 'algo'+addSpace(int(maxl-4))        +' | tau  | mean    | sigma  | compTime| nr_steps | stat. error | efficiency \n')
    for i in range(0,results.shape[0]):
        strr=samplers[i].algorithmName
        if len(samplers[i].algorithmName)<=maxl:
            strrn=addSpace(int(maxl-len(samplers[i].algorithmName)))
        strr=strr+strrn+' | '
        for j in range(0,6):

            strr=strr+"{0:.4f}".format(round(results[i,j],6))

            strr=strr+' | '
        if (results[i,3]==0):
            stref=repr(0.0)
        else:
            stref="{0:.4f}".format(round(effStd/(results[i,2]**2*results[i,3]),6))
        print(strr+stref+'\n')


    # fileNameAll = potentialFlag+'.txt'
    # fileall = open(fileNameAll,'w')
    #
    # fileall.write('\n*************************\n')
    # fileall.write('\n tau  | mean    | sigma  | compTime| nr_steps | stat. error | efficiency\n')
    # for i in range(0,3):
    #     strr=''
    #     for j in range(0,6):
    #         strr=strr+"{0:.4f}".format(round(results[i,j],6))+' | '
    #     if (results[i,3]==0):
    #         stref=repr(0.0)
    #     else:
    #         stref="{0:.4f}".format(round(effStd/(results[i,2]**2*results[i,3]),6))
    #
    #     fileall.write(strr+stref+'\n')
    # fileall.close()

def addSpace(nr):
    s=''
    for i in range(0,nr):
        s=s+' '
    return s


def unbias(x, eps, sampler, kernel):

    ModNr=1

    if (kernel==None):
        kernelDiff=dm.compute_kernel(x, eps)
    else:
        kernelDiff=kernel

    qImoportanceSampling=kernelDiff.sum(axis=1)

    assert len(x)==len(qImoportanceSampling)

    xIS=np.zeros(x.shape)
    weight=np.zeros(len(x))
    U=np.zeros(len(x))

    count=0

    for xi in x:
        U[count]=sampler.model.energy(xi)
        weight[count]=np.exp(-U[count]/sampler.T)/qImoportanceSampling[count]
        xIS[count,:]=xi*weight[count]
        count=count+1

    Ntilde=np.sum(weight)

    return xIS, Ntilde, weight

def compute_weight_target_distribution(target_distribution, qImoportanceSampling):
    """
    parameters: target_distribution and qImoportanceSampling have length of the trajectory x
    """

    ModNr=1

    weight=np.zeros(len(qImoportanceSampling))

    for count in range(len(qImoportanceSampling)):
        weight[count]=target_distribution[count]/qImoportanceSampling[count]

    Ntilde=np.sum(weight)

    return weight, Ntilde


def computeAverages(x, f, units=1.0, vectorFunction=False):

    if vectorFunction == True:
        return np.mean(f)
    else:
        Vef=np.zeros(len(x))
        ci=0
        for xi in x:

            Vef[ci]=f(xi)/units
            ci=ci+1

        return np.mean(Vef)

def computeUnbiasedAverages(x, f, weight, units=1.0, vectorFunction=False):

    Ntilde=np.mean(weight)

    Vef=np.zeros(len(x))

    assert len(Vef)==len(weight)

    ci=0
    if vectorFunction==False:
        for xi in x:

            Vef[ci]=weight[ci]*f(xi)/units
            ci=ci+1
    else:
        for xi in x:

            Vef[ci]=weight[ci]*f[ci]/units
            ci=ci+1


    return np.mean(Vef)/Ntilde
