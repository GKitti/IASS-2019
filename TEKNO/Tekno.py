import opensees as ops
import numpy as np
import matplotlib.pyplot as plt
import units as un
import Types as tp
import TekGeom as tg
import Imperfection as imp
from plotly.graph_objs import Layout, Figure, Marker
from plotly.graph_objs import Histogram
geompathGK='c:\\Kitti\\Dropbox\\PHD\\_PROCESS\\1_Grid_geometry\\saved geom'

# C R E A T E   G E O M E T R Y   +   O P E N S E E S   M O D E L


def CHSSection(secID, matID, D, t, nc, nr, GJ):
    # create a circular hollow cross-section
    R = D/2
    r = R - t
    ops.section('Fiber', secID, '-GJ', GJ)
    ops.patch('circ', matID, nc, nr, 0., 0., r, R, 0., 360.)
    #  ops.section('Elastic', SecTag, GRS.Es, GRS.secA, GRS.secI, GRS.secI, GRS.Gs, GRS.secIt)


def CreateGeom(GRS, geompath=geompathGK):
    if GRS.GeomType == 1:
        tg.Geometry1(GRS)
    elif GRS.GeomType == 2:
        tg.Geometry2(GRS)
    elif GRS.GeomType == 3:
        tg.Geometry3(GRS) # Kiewitt
    else:
        tg.Geometry4(GRS, geompath=geompathGK)  # Grasshopper relaxed
    tg.SplitBeams(GRS)


def BuildOpsModel(GRS):
    ops.wipe()
    ops.model('BasicBuilder', '-ndm', 3, '-ndf', 6)

    # material
    matID = 66
    matID1 = 67
    sY = 235. * un.MPa
    if GRS.MatNL == False:
        ops.uniaxialMaterial('Elastic', matID, GRS.Es)
    else:
        ops.uniaxialMaterial('Steel4', matID1, sY, GRS.Es, '-kin', 4e-3, 50., 0.05, 0.15, '-ult', 360.* un.MPa, 5., )
        ops.uniaxialMaterial('MinMax', matID, matID1, '-min', -0.20, '-max', 0.20)

        #ops.uniaxialMaterial('Steel4', matID, sY, GRS.Es, '-kin', 1e-2, 50., 0.05, 0.15, '-ult', 360.* un.MPa, 10., )
        #ops.uniaxialMaterial('Steel4', matID, sY, GRS.Es, '-kin', 1e-2, 50., 0.05, 0.15)
        #ops.uniaxialMaterial('ElasticBilin', matID, GRS.Es, 210. * 1e7, sY / GRS.Es)
        #ops.uniaxialMaterial('ElasticBilin', matID, GRS.Es, 210. * 1e6, sY / GRS.Es)
        #ops.uniaxialMaterial('ElasticBilin', matID, GRS.Es, 1., sY / GRS.Es) #Oldalnyomasos igy futott le

    # cross-section
    CHSid = 99
    CHSSection(CHSid, matID, GRS.secD, GRS.secT, 8, 1, GRS.Gs*GRS.secIt)

    # nodes
    for i in range(GRS.nbNsAll):
        ops.node(int(100+i), GRS.nsAll.x[i], GRS.nsAll.y[i], GRS.nsAll.z[i])

    #  ...create zeroLength element nodes...

    # end supports
    if GRS.SupType==0: # all boundary points fixed for deformations
        for i in range(GRS.nbBns):
            ops.fix(100+GRS.bn[i], 1, 1, 1, 0, 0, 0)
    elif GRS.SupType == 1: # oldalnyomasos
        for i in range(len(GRS.bnX)):
            ops.fix(100+GRS.bnX[i], 1, 0, 1, 0, 0, 0)
        for i in range(len(GRS.bnY)):
            ops.fix(100+GRS.bnY[i], 0, 1, 1, 0, 0, 0)
        for i in range(len(GRS.bnC)):
            ops.fix(100+GRS.bnC[i], 1, 1, 1, 0, 0, 0)
    elif GRS.SupType == 2: # oldalnyomasmentes
        for i in range(len(GRS.bnX)):
            ops.fix(100+GRS.bnX[i], 0, 0, 1, 0, 0, 0)
        for i in range(len(GRS.bnY)):
            ops.fix(100+GRS.bnY[i], 0, 0, 1, 0, 0, 0)
        #for i in range(len(GRS.bnC)):
        #    ops.fix(100+GRS.bnC[i], 1, 1, 1, 0, 0, 0)
        ops.fix(100 + GRS.bnC[0], 1, 1, 1, 0, 0, 0)
        ops.fix(100 + GRS.bnC[1], 0, 0, 1, 0, 0, 0)
        ops.fix(100 + GRS.bnC[2], 1, 0, 1, 0, 0, 0)
        ops.fix(100 + GRS.bnC[3], 0, 0, 1, 0, 0, 0)
    elif GRS.SupType == 3: # felmerev #NOT WORKING YET
        pass
    elif GRS.SupType == 4: # sarkok
        for i in range(len(GRS.bnC)):
            ops.fix(100+GRS.bnC[i], 1, 1, 1, 0, 0, 0)
    elif GRS.SupType == 5:  # dome oldalnyomasos
        for i in range(GRS.nbBns):
            if i==0: ops.fix(100+GRS.bn[i], 1, 1, 1, 0, 0, 0)
            elif i==10: ops.fix(100+GRS.bn[i], 0, 1, 1, 0, 0, 0)
            else: ops.fix(100+GRS.bn[i], 0, 0, 1, 0, 0, 0)
    elif GRS.SupType == 6:  # dome #NOT WORKING YET
        pass

    # transformations
    TRtag = 55
    if GRS.GeomNL == 1:
        TRType = 'Corotational'
        ops.geomTransf('Corotational', TRtag, 0., 0., 1.)
    else:
        TRType = 'Linear'
        ops.geomTransf('Linear', TRtag, 0., 0., 1.)

    # integration points
    gauss = 5
    beamIntTag=44
    ops.beamIntegration('Lobatto', beamIntTag, CHSid, gauss)

    # create elements
    for i in range(GRS.nbElAll):
        sID = GRS.lsAll.sID[i]
        eID = GRS.lsAll.eID[i]
        dx = abs(GRS.nsAll.x[sID] - GRS.nsAll.x[eID])
        dy = abs(GRS.nsAll.y[sID] - GRS.nsAll.y[eID])
        dz = abs(GRS.nsAll.z[sID] - GRS.nsAll.z[eID])
        if dx+dy+dz == 0:
            ops.equalDOF(eID, sID, 1,2,3,4,5,6)  # Grasshopper geomType=4 Zero length elements - should not come here
        else:
            ops.element('forceBeamColumn', int(1000 + i), int(100+sID), int(100+eID), TRtag, beamIntTag)

    #...zeroLength elements for the hinges...


def EqProperties(GRS):

    if GRS.GeomType != 4:
        GRS.LSum = 0
        GRS.lmax=0
        LSigma = 0
        GRS.Ncr = np.zeros(GRS.nbEl)
        for i in range(GRS.nbEl):
            sID = GRS.ls.sID[i]
            eID = GRS.ls.eID[i]
            sPt = tp.GSnodes(GRS.ns.x[sID], GRS.ns.y[sID], GRS.ns.z[sID])
            ePt = tp.GSnodes(GRS.ns.x[eID], GRS.ns.y[eID], GRS.ns.z[eID])
            wL  = ((sPt.x - ePt.x)**2 + (sPt.y - ePt.y)**2 + (sPt.z - ePt.z)**2)**0.5
            GRS.LSum = GRS.LSum + wL
            GRS.Ncr[i]=np.pi**2*GRS.Es*GRS.secI/wL**2
            if GRS.lmax < wL: GRS.lmax = wL
        GRS.Lav = GRS.LSum / GRS.nbEl  # m
        for i in range(GRS.nbEl):
            wL = ((sPt.x - ePt.x) ** 2 + (sPt.y - ePt.y) ** 2 + (sPt.z - ePt.z) ** 2) ** 0.5
            LSigma = LSigma + (GRS.Lav-wL)**2
        LSigma = (LSigma / GRS.nbEl)**0.5  # standard deviation [m]
        GRS.covL = LSigma / GRS.Lav  # coefficient of variation
    GRS.eqB = 3**0.5/4/GRS.Lav*(3*GRS.Es*GRS.secI+GRS.Gs*GRS.secIt)  # kNm
    GRS.eqT = 3*3**0.5/4*GRS.Es*GRS.secA/GRS.Lav  # kN/m
    GRS.teq = 2* ((3*GRS.Es*GRS.secI+GRS.Gs*GRS.secIt)/(GRS.Es*GRS.secA))**0.5  # m
    GRS.Eeq = 2/3**0.5*GRS.secA/GRS.Lav/GRS.teq*GRS.Es  # kN/m2
    rho = GRS.eqB/GRS.eqT/GRS.PlanArea  # GRS.teq**2/GRS.span**2/3
    #1.2566667*(GRS.secI/GRS.secA)/GRS.span**2
    return rho


def calclav(span, lperh, gn, geomtype=1, gr=5):
    tempGRS = tp.GridShell(span, span/lperh, 100, 10, gn, 4, geomtype, gr)
    CreateGeom(tempGRS)
    EqProperties(tempGRS)
    return tempGRS.Lav

def calclav2(span, lperh, gn, geomtype=1, gr=5):
    tempGRS = tp.GridShell(span, span/lperh, 100, 10, gn, 4, geomtype, gr)
    CreateGeom(tempGRS)
    EqProperties(tempGRS)
    return tempGRS.Lav, tempGRS.nbNsAll


# A N A L Y Z E


def Analyze(GRS, Fz, showMaxOnly=0, printOn=0):
    ID = GRS.maxNsID
    ok = True

    lefutott, NDisp, EForce, loadA, reI = RunIterations(GRS, Fz, printOn)

    if (lefutott == 1) and (showMaxOnly):
        ok = False
        loadA = np.zeros(GRS.Steps+1)
        print('load was lower than capacity')
    elif ((lefutott==0) and (reI==0)):
        ok = False
        loadA = np.zeros(GRS.Steps+1)
        print('probably modelling error')

    if (lefutott == 0) and (reI<GRS.Steps): # it did not converge
        for i in range(reI, GRS.Steps+1):
            NDisp[i, :, :] = NDisp[reI-1, :, :]
            EForce[i, :] = EForce[reI-1, :]
            loadA[i] = loadA[reI - 1]

    ID = np.argmax(NDisp[:, :, 2], 1)  # node number with maximum z displacement in every load step
    ID = ID[-1]  # node number with maximum z displacement in the last load step
    if ID == 0: ID = GRS.maxNsID

    return NDisp, EForce, ID, loadA, ok


def RunIterations(GRS, Fz, printOn):
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    loadA = np.linspace(0, -Fz, GRS.Steps + 1)  # kN
    for i in range(GRS.nbnBns):
        if GRS.LoadType == 0 \
                or (GRS.GeomType in {0, 1, 2} and GRS.LoadType == 1 and GRS.nsAll.x[GRS.nbn[i]] <= 0) \
                or (GRS.GeomType in {0, 1, 2} and GRS.LoadType == 2 and GRS.nsAll.y[GRS.nbn[i]] <= 0) \
                or (GRS.GeomType in {4} and GRS.LoadType == 1 and GRS.nsAll.x[GRS.nbn[i]] <= GRS.span / 2) \
                or (GRS.GeomType in {4} and GRS.LoadType == 2 and GRS.nsAll.y[GRS.nbn[i]] <= GRS.span / 2):
            ops.load(int(100 + GRS.nbn[i]), 0., 0., Fz * un.kN, 0., 0., 0.)
    GRS.GetTopNode()
    # ops.load(int(100+GRS.maxNsID), 0., 0., Fz * kN, 0., 0., 0.)  # mid-point

    # create SOE
    ops.system('UmfPack')
    # create DOF number
    ops.numberer('RCM')
    # create constraint handler
    ops.constraints('Transformation')
    # create test
    ops.test('EnergyIncr', 1.e-12, 10)
    # create algorithm
    ops.algorithm("Newton")

    NDisp = np.zeros([GRS.Steps + 1, GRS.nbNsAll, 3])
    # EDisp = np.zeros([GRS.Steps + 1, GRS.nbElAll, 6])
    EForce = np.zeros([GRS.Steps + 1, GRS.nbElAll, 12])
    reI=0

    reI = 0
    lefutott = 1
    i=0
    load = 0
    stepSize = 1.0 / GRS.Steps
    ops.integrator("LoadControl", stepSize)
    ops.analysis('Static')
    while ((-stepSize*Fz > GRS.MinStepSize) and (i<GRS.Steps)):
        hiba = ops.analyze(1)
        if hiba == 0:
            load += -stepSize * Fz
            i += 1
            loadA[i] = load
            if i == 1:
                if printOn: print('analysis step 1 completed successfully')
            for j in range(GRS.nbNsAll):
                NDisp[i, j, 0] = - ops.nodeDisp(int(j + 100), 1) / un.mm  # mm displacement
                NDisp[i, j, 1] = - ops.nodeDisp(int(j + 100), 2) / un.mm  # mm displacement
                NDisp[i, j, 2] = - ops.nodeDisp(int(j + 100), 3) / un.mm  # mm displacement
            for j in range(GRS.nbElAll):
                EForce[i, j] = ops.eleResponse(int(j+1000), 'localForce')
                # EDisp[i, j] = ops.eleResponse(int(j + 1000), 'basicDeformation')
        else:
            stepSize = stepSize/2
            if printOn: print('analysis failed to converge in step ', i)
            ops.integrator("LoadControl", stepSize)
            lefutott = 0
            reI = i

    if i == GRS.Steps:
        if reI == 1:
            if printOn: print('analysis failed to converge')

    return lefutott, NDisp, EForce, loadA, reI


def Analyze2(GRS, Fz, showMaxOnly=0, printOn=0):
    ID = GRS.maxNsID
    ok = True
    loadA = np.linspace(0, -Fz, GRS.Steps + 1)  # kN

    lefutott, NDisp, EForce, reI = RunIterations2(GRS, Fz, printOn)

    if GRS.reRunIfNotConv==1 and lefutott==0 and reI>1:
        #  if the previous run did not converge, lets run it one more time with a more suitable load
        #  this can not be done if the convergence problem occured in the first load step,
        #  that indicated a modelling problem (reI=0)
        #  reRunIfNotConv = 0 turns off this extra analysis
        if printOn: print('load was too high, rerun')
        Fz=(reI)/GRS.Steps*Fz  # reI-1 volt korabban
        BuildOpsModel(GRS)
        lefutott, NDisp, EForce, reI = RunIterations2(GRS, Fz, printOn)
        loadA = np.linspace(0, -Fz, GRS.Steps + 1)  # kN
    elif (lefutott == 1) and (showMaxOnly):
        ok = False
        loadA = np.zeros(GRS.Steps)
        print('load was lower than capacity')
    elif (GRS.reRunIfNotConv==0 and lefutott==0 and (reI==0)):
        ok = False
        loadA = np.zeros(GRS.Steps)
        print('probably modelling error')
    if (lefutott == 0) and (reI>GRS.Steps*0.8): #  once again it did not converge in one of the last load steps, but the results can still be used
        for i in range(reI, GRS.Steps+1):
            NDisp[i, :, :] = NDisp[reI-1, :, :]
            EForce[i, :] = EForce[reI-1, :]
            loadA[i] = loadA[reI - 1]

    ID = np.argmax(NDisp[:, :, 2], 1)  # node number with maximum z displacement in every load step
    ID = ID[-1]  # node number with maximum z displacement in the last load step
    if ID == 0: ID = GRS.maxNsID

    return NDisp, EForce, ID, loadA, ok


def RunIterations2(GRS, Fz, printOn):

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for i in range(GRS.nbnBns):
        ops.load(int(100 + GRS.nbn[i]), 0., 0., Fz * un.kN, 0., 0., 0.)
    GRS.GetTopNode()
    # ops.load(int(100+GRS.maxNsID), 0., 0., Fz * kN, 0., 0., 0.)  # mid-point

    # create SOE
    ops.system('UmfPack')
    # create DOF number
    ops.numberer('RCM')
    # create constraint handler
    ops.constraints('Transformation')
    # create integrator
    ops.integrator("LoadControl", 1.0 / GRS.Steps)
    # create algorithm
    ops.algorithm("Newton")
    # create test
    ops.test('EnergyIncr', 1.e-10, 100)

    ops.analysis('Static')

    NDisp = np.zeros([GRS.Steps + 1, GRS.nbNsAll, 3])
    # EDisp = np.zeros([GRS.Steps + 1, GRS.nbElAll, 6])
    EForce = np.zeros([GRS.Steps + 1, GRS.nbElAll, 12])
    reI=0

    reI = 0
    lefutott = 1
    for i in range(1, GRS.Steps + 1):
        hiba = ops.analyze(1)
        if hiba == 0:
            if i == 1:
                if printOn: print('analysis step 1 completed successfully')
            for j in range(GRS.nbNsAll):
                NDisp[i, j, 0] = - ops.nodeDisp(int(j + 100), 1) / un.mm  # mm displacement
                NDisp[i, j, 1] = - ops.nodeDisp(int(j + 100), 2) / un.mm  # mm displacement
                NDisp[i, j, 2] = - ops.nodeDisp(int(j + 100), 3) / un.mm  # mm displacement
            for j in range(GRS.nbElAll):
                EForce[i, j] = ops.eleResponse(int(j+1000), 'localForce')
                # EDisp[i, j] = ops.eleResponse(int(j + 1000), 'basicDeformation')
        else:
            lefutott = 0
            reI = i
            if reI == 1:
                if printOn: print('analysis failed to converge in step ', i)
            break

    return lefutott, NDisp, EForce, reI


def GetResults(disp, force, Ncr=[1], nb=-1):
    lc = -1
    dmax = 0.
    Mmax = 0.
    Nmax = 0.
    Nmin = 0.
    nminid = 0
    nmaxid = 0
    mid = 0
    etaNcrMax = 0
    k=0
    for d in disp[lc]:
        dtemp = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5
        if dtemp > dmax:
            dmax = dtemp
    for M in force[lc]:
        k+=1
        Mtemp = (M[4] ** 2 + M[5]**2) ** 0.5
        if Mtemp > Mmax:
            Mmax = Mtemp
            mid = k
        Mtemp = (M[10] ** 2 + M[11]**2) ** 0.5
        if Mtemp > Mmax:
            Mmax = Mtemp
            mid = k

    k = 0
    el = -1
    for N in force[lc]:
        k += 1
        if k % nb == 1:
            el += 1
        Ntemp = N[0]
        if Ntemp > Nmax:
            Nmax = Ntemp
            nmaxid = k
        if Ntemp < Nmin:
            Nmin = Ntemp
            nminid = k
        if nb >= 0:
            if Ntemp<0:
                etaNcrTemp = - Ntemp / Ncr[el]  # largest compression efficiency
                if etaNcrTemp > etaNcrMax:
                    etaNcrMax = etaNcrTemp

    return dmax, Nmin/1000, Nmax/1000, Mmax/1000, nminid, nmaxid, mid, etaNcrMax  # mm,kN,kN,kNm


def loop_analyse(n, m, LperH, gN, Span, nb, Fz, DStart, DStep, tStart, tStep, MatNL=1, suptype=1, geomtype=1, steps=50, msz=0.1, gR=5):

    nm = n*m
    CapacityF = np.zeros(nm)
    Capacity = np.zeros(nm)
    Mmax = np.zeros(nm)
    Nmin = np.zeros(nm)
    Nmax = np.zeros(nm)
    Dmax = np.zeros(nm)
    rhoA = np.zeros(nm)
    etaNcrMax = np.zeros(nm)
    Height = Span/LperH   # m height at apex

    print('D: ', np.arange(DStart, DStart + n*DStep, DStep))
    print('t: ', np.arange(tStart, tStart + m*tStep, tStep))

    for i in range(n):
        D=DStart+DStep*i
        for j in range(m):
            k=i*m+j
            t=tStart+tStep*j
            GRS = tp.GridShell(Span,Height,D,t,gN,nb,geomtype,gR)
            CreateGeom(GRS)
            GRS.SupType = suptype
            GRS.MatNL = MatNL
            GRS.Steps = steps
            GRS.MinStepSize = msz
            BuildOpsModel(GRS)
            disp, force, ID, loadA, ok = Analyze(GRS, Fz, 1, 0)
            if ok:
                rhoA[k] = EqProperties(GRS)
                CapacityF[k] = loadA[-1]  #kN
                Capacity[k] = loadA[-1]*GRS.nbnBns/GRS.PlanArea  #kN/m2
                Dmax[k], Nmin[k], Nmax[k], Mmax[k],_, _, _, etaNcrMax[k]= GetResults(disp, force, GRS.Ncr, GRS.Nb)
                _ = plt.plot(disp[:,ID,2],loadA, 'g') # dot plot
    plt.xlabel('maximum nodal displacement [mm]')
    plt.ylabel('nodal load [kN]')
    plt.show()
    return GRS, CapacityF, Capacity, rhoA, ID, etaNcrMax


def loop_analyse2(n, m, LperH, gN, Span, nb, Fz, DStart, DStep, tStart, tStep, MatNL=1, suptype=1, geomtype=1, steps=50, msz=0.1, gR=5):
    # no plot, different outputs compared to loop_analyse function

    nm = n*m
    CapacityF = np.zeros(nm)
    Capacity  = np.zeros(nm)
    Mmax      = np.zeros(nm)
    Nmin      = np.zeros(nm)
    Nmax      = np.zeros(nm)
    Dmax      = np.zeros(nm)
    rhoA      = np.zeros(nm)
    loadAA    = np.zeros([nm, steps + 1])
    dispA     = np.zeros([nm, steps + 1])
    etaNcrMax = np.zeros(nm)
    Height = Span/LperH   # m height at apex

    #print('D: ', np.arange(DStart, DStart + n*DStep, DStep))
    #print('t: ', np.arange(tStart, tStart + m*tStep, tStep))

    for i in range(n):
        D=DStart+DStep*i
        for j in range(m):
            k=i*m+j
            t=tStart+tStep*j
            GRS = tp.GridShell(Span,Height,D,t,gN,nb,geomtype,gR)
            CreateGeom(GRS)
            GRS.SupType = suptype
            GRS.MatNL = MatNL
            GRS.Steps = steps
            GRS.MinStepSize = msz
            BuildOpsModel(GRS)
            disp, force, ID, loadA, ok = Analyze(GRS, Fz, 1, 0)
            if ok:
                rhoA[k] = EqProperties(GRS)
                CapacityF[k] = loadA[-1]  #kN
                Capacity[k] = loadA[-1]*GRS.nbnBns/GRS.PlanArea  #kN/m2
                Dmax[k], Nmin[k], Nmax[k], Mmax[k],_, _, _, etaNcrMax[k]= GetResults(disp, force, GRS.Ncr, GRS.Nb)
                dispA[k] = disp[:,ID,2]
                loadAA[k] = loadA
    return GRS, CapacityF, Capacity, rhoA, etaNcrMax, loadAA, dispA


def loop_analyse_imp(n, m, LperH, gN, Span, nb, Fz, DStart, DStep, tStart, tStep, amp=0, MatNL=1, suptype=1, geomtype=1, steps=50, msz=0.1, gR=5, impType=0):
    # no plot, different outputs compared to loop_analyse function

    nm = n*m
    CapacityF = np.zeros(nm)
    Capacity  = np.zeros(nm)
    Mmax      = np.zeros(nm)
    Nmin      = np.zeros(nm)
    Nmax      = np.zeros(nm)
    Dmax      = np.zeros(nm)
    rhoA      = np.zeros(nm)
    loadAA    = np.zeros([nm, steps + 1])
    dispA     = np.zeros([nm, steps + 1])
    etaNcrMax = np.zeros(nm)
    Height = Span/LperH   # m height at apex

    #print('D: ', np.arange(DStart, DStart + n*DStep, DStep))
    #print('t: ', np.arange(tStart, tStart + m*tStep, tStep))

    for i in range(n):
        D=DStart+DStep*i
        for j in range(m):
            k=i*m+j
            t=tStart+tStep*j
            GRS = tp.GridShell(Span,Height,D,t,gN,nb,geomtype,gR)
            CreateGeom(GRS)
            GRS.SupType = suptype
            GRS.MatNL = MatNL
            GRS.Steps = steps
            GRS.MinStepSize = msz
            if amp>0:
                if impType!=1:
                    alfa, GRS = imp.imperfection(GRS, 0, amp, impType)
                    print('{0:.2f}'.format(-alfa / 1000))
                else:
                    GRS = imp.imperfectionDisp(GRS, amp)
            BuildOpsModel(GRS)
            disp, force, ID, loadA, ok = Analyze(GRS, Fz, 1, 0)
            if ok:
                rhoA[k] = EqProperties(GRS)
                CapacityF[k] = loadA[-1]  #kN
                Capacity[k] = loadA[-1]*GRS.nbnBns/GRS.PlanArea  #kN/m2
                Dmax[k], Nmin[k], Nmax[k], Mmax[k],_, _, _, etaNcrMax[k]= GetResults(disp, force, GRS.Ncr, GRS.Nb)
                dispA[k] = disp[:,ID,2]
                loadAA[k] = loadA
    return GRS, CapacityF, Capacity, rhoA, etaNcrMax, loadAA, dispA


# R E S U L T S


def Eval(IO):

    # oldalnyomasos
    nm = IO.n * IO.m
    CapacityF = np.zeros(nm)
    Capacity = np.zeros(nm)
    Mmax = np.zeros(nm)
    Nmin = np.zeros(nm)
    Nmax = np.zeros(nm)
    Dmax = np.zeros(nm)
    rhoA = np.zeros(nm)
    eqBA = np.zeros(nm)
    eqTA = np.zeros(nm)
    EA = np.zeros(nm)
    TA = np.zeros(nm)
    DA = np.zeros(nm)
    tA = np.zeros(nm)
    AA = np.zeros(nm)
    for i in range(IO.n):
        D = IO.DStart + IO.DStep * i
        for j in range(IO.m):
            k = i * IO.m + j
            DA[k] = D
            t = IO.tStart + IO.tStep * j
            tA[k] = t
            GRS = tp.GridShell(IO.Span, IO.Height, D, t, IO.gN, IO.nb)
            CreateGeom(GRS)
            GRS.SupType = 1  # oldalnyomasos
            GRS.MatNL = IO.MatNL
            GRS.GeomNL = IO.GeomNL
            GRS.Steps = IO.Steps
            BuildOpsModel(GRS)
            if IO.LoadType == 'F':
                Fz = IO.qz
            else:
                Fz = - IO.qz * GRS.span ** 2 / GRS.nbnBns  # kN
            disp, force, ID, loadA, _ = Analyze(GRS, Fz, 0 ,IO.printon)
            if IO.LoadType != 'F':
                for l in range(loadA.shape[0]):
                    loadA[l] = loadA[l] / GRS.span ** 2 * GRS.nbnBns
            CapacityF[k] = loadA[-1]  # kN
            Capacity[k] = loadA[-1] * GRS.nbnBns / GRS.span ** 2  # kN/m2
            rhoA[k] = EqProperties(GRS)
            Dmax[k], Nmin[k], Nmax[k], Mmax[k], _, _, _,_ = GetResults(disp, force)
            eqBA[k] = GRS.eqB
            eqTA[k] = GRS.eqT
            EA[k] = GRS.Eeq
            TA[k] = GRS.teq
            AA[k] = GRS.secA
            _ = plt.plot(disp[:, ID, 2], loadA, 'g')  # dot plot

    # oldalnyomasmentes
    CapacityF2 = np.zeros(nm)
    Capacity2 = np.zeros(nm)
    Mmax2 = np.zeros(nm)
    Nmin2 = np.zeros(nm)
    Nmax2 = np.zeros(nm)
    Dmax2 = np.zeros(nm)
    for i in range(IO.n):
        D = IO.DStart + IO.DStep * i
        for j in range(IO.m):
            k = i * IO.m + j
            t = IO.tStart + IO.tStep * j
            GRS = tp.GridShell(IO.Span, IO.Height, D, t, IO.gN, IO.nb)
            CreateGeom(GRS)
            GRS.SupType = 2  # oldalnyomasmentes
            GRS.MatNL = IO.MatNL
            GRS.GeomNL = IO.GeomNL
            GRS.Steps = IO.Steps
            BuildOpsModel(GRS)
            if IO.LoadType == 'F':
                Fz = IO.qz
            else:
                Fz = - IO.qz * GRS.span ** 2 / GRS.nbnBns  # kN
            disp2, force2, ID2, loadA2, _ = Analyze(GRS, Fz, 0, IO.printon)
            if IO.LoadType != 'F':
                for l in range(loadA2.shape[0]):
                    loadA2[l] = loadA2[l] / GRS.span ** 2 * GRS.nbnBns
            CapacityF2[k] = loadA2[-1]  # kN
            Capacity2[k] = loadA2[-1] * GRS.nbnBns / GRS.span ** 2  # kN/m2
            Dmax2[k], Nmin2[k], Nmax2[k], Mmax2[k], _, _, _,_ = GetResults(disp2, force2)
            _ = plt.plot(disp2[:, ID2, 2], loadA2, 'b')  # dot plot

    return Nmin, Nmax, Mmax, rhoA, Nmin2, Nmax2, Mmax2, eqBA, eqTA, EA, TA, AA


def TestResults():

    # compare results with AxisVM model results

    Span = 40  # m span of two-member structure
    Height = 5  # m height at apex
    gN = 16  # number of triangles along each span
    D = 330  # mm CHS cross-section diameter
    t = 16  # mm CHS cross-section thickness
    nb = 4  # number of finite elements along one beam

    GRS = tp.GridShell(Span, Height, D, t, gN, nb)

    ResV = np.zeros((2, 3))
    ResAxis = np.zeros((2, 3))
    ResOps = np.zeros((2, 3))

    # GEOM NL, MAT L
    GRS.MatNL = False
    Fz = -80.  # [kN]

    GRS.SupType = 1  # oldalnyomasos
    BuildModel(GRS)
    disp, force, ID, loadA, _ = Analyze(GRS, Fz)  # load step / node / xyz displacement component [mm]
    ResV[0] = GetResults(disp, force)
    GRS.SupType = 2  # oldalnyomasmentes
    BuildModel(GRS)
    disp2, force2, ID2, loadA2, _ = Analyze(GRS, Fz)  # load step / node / xyz displacement component [mm]
    ResV[1] = GetResults(disp2, force2)
    ResAxis[0] = [12.45, 611.4, 40.98]
    ResAxis[1] = [151.65, 2861.7, 659.7]
    ResOps[0] = [12.52858527, 613.28010789, 40.98835795]
    ResOps[1] = [151.76894853, 2857.8671693, 677.75188794]

    difAxis = (ResV - ResAxis)/ResAxis
    for i in np.ravel(difAxis):
        if i > 0.1:
            return 'Axis results are different'

    difOps = (ResV - ResOps) / ResOps
    for i in np.ravel(difOps):
        if i > 0.01:
            return 'Opensees results are different'

    # GEOM NL, MAT NL
    GRS.MatNL = True
    GRS.SupType = 1  # oldalnyomasos
    BuildModel(GRS)
    disp, force, ID, loadA, _ = Analyze(GRS, Fz, 0, 0)  # load step / node / xyz displacement component [mm]
    ResV[0] = GetResults(disp, force)
    GRS.SupType = 2  # oldalnyomasmentes
    BuildModel(GRS)
    disp2, force2, ID2, loadA2, _ = Analyze(GRS, Fz, 0, 0)  # load step / node / xyz displacement component [mm]
    ResV[1] = GetResults(disp2, force2)

    ResAxis[0] = [12.388, 611.4, 40.97]
    ResAxis[1] = [199.990, 2498, 399.025]
    ResOps[0] = [12.52858527, 613.28010789, 40.98835795]
    ResOps[1] = [206.83377898, 2136.06799021, 368.17068634]

    difAxis = (ResV - ResAxis)/ResAxis
    for i in np.ravel(difAxis):
        if i > 0.1:
            return 'Axis results are different'

    difOps = (ResV - ResOps) / ResOps
    for i in np.ravel(difOps):
        if i > 0.01:
            return 'Opensees results are different'

    return 'ok'


class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'


def calc_error(GRSA,rhoAA,CapacityA,refx,refy,nm=4,runNbfrom=0,runNb=30,Cmin=3, Cmax=20,detailOn=True):
    errMax = 0
    errMin = 0
    errAv = 0
    count = 0
    err = np.zeros(nm)
    errV = []
    if detailOn:
        print('  Nb  -  L   g  LpH  Lav  - L/Lav:            error')
        print(' [-]    [m] [-] [-]  [m]     [-]     sec1  sec2  sec3  sec4')
        print('-----------------------------------------------------------')
    for i in range(runNbfrom,runNbfrom+runNb):
        GRS = GRSA[i]
        if GRS.span / GRS.Lav > 0:
            cc = (GRS.span / GRS.height - 12) ** 2 / 51 + 0.92
            pred = np.interp(rhoAA[i], refx, refy)
            predCapacity = pred * GRS.secT * GRS.Lav * cc * 1000 * GRS.nbnBns / GRS.span ** 2
            err = (CapacityA[i] - predCapacity) / CapacityA[i] * 100
            if detailOn: print('{:4d}  -  {:d}  {:.0f}  {:2.0f} {:4.1f}  - {:5.1f}:  '.format(i, GRS.span, GRS.gN,
                                                                                 GRS.span / GRS.height, GRS.Lav,
                                                                                 GRS.span / GRS.Lav), end="",
                  flush=True)
            for j in range(nm):
                if CapacityA[i, j] >= Cmin and CapacityA[i, j] <= Cmax:
                    count += 1
                    if err[j] > 0 and errMax < err[j]: errMax = err[j]
                    if err[j] < 0 and errMin > err[j]: errMin = err[j]
                    errAv = errAv + abs(err[j])
                    errV.append(err[j])
                    if detailOn: print('{:5.1f}'.format(err[j]), end="", flush=True)
                else:
                    if detailOn: print(bcolors.WARNING + '{:5.1f}'.format(err[j]) + bcolors.ENDC, end="", flush=True)
            if detailOn: print('')
    print('Max error:        {:4.1f}%'.format(errMax))
    print('Min error:        {:4.1f}%'.format(errMin))  # tulbecsult
    print('Average error:    {:4.1f}%'.format(errAv / count))
    print(count)

    data = [Histogram(x=errV,
                      xbins=dict(start=-49, end=49, size=2))]

    layout = Layout(bargap=0.05,
                    xaxis1=dict(range=(-50, 50), tick0=-50., dtick=2., title='Felirat'),
                    )

    fig=Figure(data=data, layout=layout)

    return fig


def calc_error_Omentes(GRSA,rhoAA,CapacityA,refx,refy,nm=4,runNb=30,Cmin=3, Cmax=20,detailOn=True):
    errMax = 0
    errMin = 0
    errAv = 0
    count = 0
    err = np.zeros(nm)
    errV = []
    if detailOn:
        print('  Nb  -  L   g  LpH  Lav  - L/Lav:            error')
        print(' [-]    [m] [-] [-]  [m]     [-]     sec1  sec2  sec3  sec4')
        print('-----------------------------------------------------------')
    for i in range(runNb):
        GRS = GRSA[i]
        if GRS.span / GRS.Lav > 7:
            c=-0.082 * GRS.span/GRS.height + 1.82
            if i in range(48, 54): c = -0.064 * 8 + 1.64  # L/H=8
            pred = np.interp(rhoAA[i], refx, refy)
            predCapacity = pred * GRS.secT * GRS.Lav * c * 1000 * GRS.nbnBns / GRS.span ** 2
            err = (CapacityA[i] - predCapacity) / CapacityA[i] * 100
            if detailOn: print('{:4d}  -  {:d}  {:.0f}  {:2.0f} {:4.1f}  - {:5.1f}:  '.format(i, GRS.span, GRS.gN,
                                                                                 GRS.span / GRS.height, GRS.Lav,
                                                                                 GRS.span / GRS.Lav), end="",
                  flush=True)
            for j in range(nm):
                if CapacityA[i, j] >= Cmin and CapacityA[i, j] <= Cmax:
                    count += 1
                    if err[j] > 0 and errMax < err[j]: errMax = err[j]
                    if err[j] < 0 and errMin > err[j]: errMin = err[j]
                    errAv = errAv + err[j]
                    errV.append(err[j])
                    if detailOn: print('{:5.1f}'.format(err[j]), end="", flush=True)
                else:
                    if detailOn: print(bcolors.WARNING + '{:5.1f}'.format(err[j]) + bcolors.ENDC, end="", flush=True)
            if detailOn: print('')
    print('Max error:        {:4.1f}%'.format(errMax))
    print('Min error:        {:4.1f}%'.format(errMin))  # tulbecsult
    print('Average error:    {:4.1f}%'.format(errAv / count))
    print(count)

    data = [Histogram(x=errV,
                      xbins=dict(start=-49, end=49, size=2))]

    layout = Layout(bargap=0.05,
                    xaxis1=dict(range=(-50, 50), tick0=-50., dtick=2., title='Felirat'),
                    )

    fig=Figure(data=data, layout=layout)

    return fig


def calc_error_imp(GRSA, rhoAA, CapacityA, rhoAAp, CapacityAp, refx, refy, nm=4, runNb=30, Cmin=3, Cmax=20,detailOn=True,lim=0):
    errMax = 0
    errMin = 0
    errAv = 0
    count = 0
    err = np.zeros(nm)
    errV = []
    if detailOn:
        print('  Nb  -  L   g  LpH  Lav  - L/Lav:            error')
        print(' [-]    [m] [-] [-]  [m]     [-]     sec1  sec2  sec3  sec4')
        print('-----------------------------------------------------------')

    for i in range(runNb):
        GRS = GRSA[i]
        if GRS!=0:
            if GRS.span / GRS.Lav > 5:
                perf = np.interp(rhoAA[i], rhoAAp[i], CapacityAp[i])
                decrese = (perf - CapacityA[i]) / perf
                cc = (GRS.span / GRS.height - 12) ** 2 / 51 + 0.92
                pred = np.interp(rhoAA[i], refx, refy)
                predCapacity = pred * GRS.secT * GRS.Lav * cc * 1000 * GRS.nbnBns / GRS.span ** 2
                err = (CapacityA[i] - predCapacity) / CapacityA[i] * 100
                if detailOn: print('{:4d}  -  {:d}  {:.0f}  {:2.0f} {:4.1f}  - {:5.1f}:  '.format(i, GRS.span, GRS.gN,
                                                                                     GRS.span / GRS.height, GRS.Lav,
                                                                                     GRS.span / GRS.Lav), end="",
                      flush=True)
                for j in range(nm):
                    if CapacityA[i, j] >= Cmin and CapacityA[i, j] <= Cmax and decrese[j] > lim:
                        count += 1
                        if err[j] > 0 and errMax < err[j]: errMax = err[j]
                        if err[j] < 0 and errMin > err[j]: errMin = err[j]
                        errAv = errAv + abs(err[j])
                        errV.append(err[j])
                        if detailOn: print('{:5.1f}'.format(err[j]), end="", flush=True)
                    else:
                        if detailOn: print(bcolors.WARNING + '{:5.1f}'.format(err[j]) + bcolors.ENDC, end="", flush=True)
                if detailOn: print('')
    print('Max error:        {:4.1f}%'.format(errMax))
    print('Min error:        {:4.1f}%'.format(errMin))  # tulbecsult
    print('Average error:    {:4.1f}%'.format(errAv / count))
    print(count)

    data = [Histogram(x=errV,
                      xbins=dict(start=-29, end=29, size=2))]

    layout = Layout(bargap=0.05,
                    xaxis1=dict(range=(-30, 30), tick0=-30., dtick=2., title='Felirat'),
                    )

    fig = Figure(data=data, layout=layout)

    return fig


def calc_error_Kiewitt(GRSA,rhoAA,CapacityA,refx,refy,nm=4,runNb=30,Cmin=3, Cmax=20,detailOn=True):
    errMax = 0
    errMin = 0
    errAv = 0
    count = 0
    err = np.zeros(nm)
    errV = []
    if detailOn:
        print('  Nb  -  L   nxr  LpH  Lav  - L/Lav:            error')
        print(' [-]    [m]  [-] [-]  [m]     [-]     sec1  sec2  sec3  sec4')
        print('-----------------------------------------------------------')
    for i in range(runNb):
        GRS = GRSA[i]
        if GRSA[i].gN in {6, 7, 8, 9} and GRSA[i].span / GRSA[i].Lav > 8:  # and GRSA[k].gR in {5,6,7,8,9}:
            if GRSA[i].gN == 9:
                div = 0.95
            elif GRSA[i].gN == 5:
                div = 1
            elif GRSA[i].gN == 6:
                div = 1.15
            elif GRSA[i].gN == 7:
                div = 1.08
            else:
                div = 1
            pred = np.interp(rhoAA[i], refx, refy)
            cc = (GRS.span/GRS.height - 10.3) ** 2 / 26 + 1

            predCapacity = pred * GRS.secT * GRS.Lav * cc * 1000 * GRS.nbnBns / GRS.PlanArea*div
            err = (CapacityA[i] - predCapacity) / CapacityA[i] * 100
            if detailOn: print('{:4d}  -  {:d}  {:d}x{:d}  {:2.0f} {:4.1f}  - {:5.1f}:  '.format(i, GRS.span, GRS.gN,GRS.gR,
                                                                                 GRS.span / GRS.height, GRS.Lav,
                                                                                 GRS.span / GRS.Lav), end="",
                  flush=True)
            for j in range(nm):
                if CapacityA[i, j] >= Cmin and CapacityA[i, j] <= Cmax:
                    count += 1
                    if err[j] > 0 and errMax < err[j]: errMax = err[j]
                    if err[j] < 0 and errMin > err[j]: errMin = err[j]
                    errAv = errAv + err[j]
                    errV.append(err[j])
                    if detailOn: print('{:5.1f}'.format(err[j]), end="", flush=True)
                else:
                    if detailOn: print(bcolors.WARNING + '{:5.1f}'.format(err[j]) + bcolors.ENDC, end="", flush=True)
            if detailOn: print('')
    print('Max error:        {:4.1f}%'.format(errMax))
    print('Min error:        {:4.1f}%'.format(errMin))  # tulbecsult
    print('Average error:    {:4.1f}%'.format(errAv / count))
    print(count)

    data = [Histogram(x=errV,
                      xbins=dict(start=-59, end=59, size=2))]

    layout = Layout(bargap=0.05,
                    xaxis1=dict(range=(-60, 60), tick0=-60., dtick=2., title='Felirat'),
                    )

    fig = Figure(data=data, layout=layout)

    return fig