import opensees as ops
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sc
import units as un
import Tekno as tk
import Types as tp
import TekGeom as tg
import copy
import scipy.sparse as scsp

# I M P E R F E C T I O N


def get_normal_force(GRS):

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for i in range(GRS.nbnBns):
        if GRS.LoadType == 0 \
                or (GRS.GeomType in {0, 1, 2} and GRS.LoadType == 1 and GRS.nsAll.x[GRS.nbn[i]] <= 0) \
                or (GRS.GeomType in {0, 1, 2} and GRS.LoadType == 2 and GRS.nsAll.y[GRS.nbn[i]] <= 0) \
                or (GRS.GeomType in {4} and GRS.LoadType == 1 and GRS.nsAll.x[GRS.nbn[i]] <= GRS.span / 2) \
                or (GRS.GeomType in {4} and GRS.LoadType == 2 and GRS.nsAll.y[GRS.nbn[i]] <= GRS.span / 2):
            ops.load(int(100 + GRS.nbn[i]), 0., 0., -1., 0., 0., 0.)

    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1)
    ops.analysis('Static')

    ops.analyze(1)

    EForce = np.zeros([GRS.nbElAll])
    for j in range(GRS.nbElAll):
        EForce[j] = ops.eleResponse(int(j + 1000), 'localForce')[0]
    return EForce


def get_disp(GRS):

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for i in range(GRS.nbnBns):
        if GRS.LoadType == 0 \
                or (GRS.GeomType in {0, 1, 2} and GRS.LoadType == 1 and GRS.nsAll.x[GRS.nbn[i]] <= 0) \
                or (GRS.GeomType in {0, 1, 2} and GRS.LoadType == 2 and GRS.nsAll.y[GRS.nbn[i]] <= 0) \
                or (GRS.GeomType in {4} and GRS.LoadType == 1 and GRS.nsAll.x[GRS.nbn[i]] <= GRS.span / 2) \
                or (GRS.GeomType in {4} and GRS.LoadType == 2 and GRS.nsAll.y[GRS.nbn[i]] <= GRS.span / 2):
            ops.load(int(100 + GRS.nbn[i]), 0., 0., -1., 0., 0., 0.)

    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1)
    ops.analysis('Static')

    ops.analyze(1)

    NDisp = np.zeros([GRS.nbNsAll, 3])
    for j in range(GRS.nbNsAll):
        NDisp[j, 0] = ops.nodeDisp(int(j + 100), 1)
        NDisp[j, 1] = ops.nodeDisp(int(j + 100), 2)
        NDisp[j, 2] = ops.nodeDisp(int(j + 100), 3)

    return NDisp


def k_e(GRS, Li):

    EA = GRS.Es * GRS.secA
    EI = GRS.Es * GRS.secI
    GI = GRS.Gs * GRS.secIt
    teta = 12 * EI / GRS.Gs / (GRS.secA * 2 / np.pi) / Li ** 2  # Asy=2A/pi

    KEe = np.zeros((12, 12))

    KEe[0, 0] = EA / Li
    KEe[0, 6] = -      EA / Li
    KEe[1, 1] = 12 * EI / Li ** 3 / (teta + 1)
    KEe[1, 5] = 6 * EI / Li ** 2 / (teta + 1)
    KEe[1, 7] = - 12 * EI / Li ** 3 / (teta + 1)
    KEe[1, 11] = 6 * EI / Li ** 2 / (teta + 1)
    KEe[2, 2] = 12 * EI / Li ** 3 / (teta + 1)
    KEe[2, 4] = -  6 * EI / Li ** 2 / (teta + 1)
    KEe[2, 8] = - 12 * EI / Li ** 3 / (teta + 1)
    KEe[2, 10] = -  6 * EI / Li ** 2 / (teta + 1)
    KEe[3, 3] = GI / Li
    KEe[3, 9] = - GI / Li
    KEe[4, 4] = (teta + 4) * EI / Li / (teta + 1)
    KEe[4, 8] = 6 * EI / Li ** 2 / (teta + 1)
    KEe[4, 10] = (2 - teta) * EI / Li / (teta + 1)
    KEe[5, 5] = (teta + 4) * EI / Li / (teta + 1)
    KEe[5, 7] = -6 * EI / Li ** 2 / (teta + 1)
    KEe[5, 11] = (2 - teta) * EI / Li / (teta + 1)
    KEe[6, 6] = EA / Li
    KEe[7, 7] = 12 * EI / Li ** 3 / (teta + 1)
    KEe[7, 11] = - 6 * EI / Li ** 2 / (teta + 1)
    KEe[8, 8] = 12 * EI / Li ** 3 / (teta + 1)
    KEe[8, 10] = 6 * EI / Li ** 2 / (teta + 1)
    KEe[9, 9] = GI / Li
    KEe[10, 10] = (teta + 4) * EI / Li / (teta + 1)
    KEe[11, 11] = (teta + 4) * EI / Li / (teta + 1)

    KEe[6, 0] = KEe[0, 6]
    KEe[5, 1] = KEe[1, 5]
    KEe[7, 1] = KEe[1, 7]
    KEe[11, 1] = KEe[1, 11]
    KEe[4, 2] = KEe[2, 4]
    KEe[8, 2] = KEe[2, 8]
    KEe[10, 2] = KEe[2, 10]
    KEe[9, 3] = KEe[3, 9]
    KEe[8, 4] = KEe[4, 8]
    KEe[10, 4] = KEe[4, 10]
    KEe[7, 5] = KEe[5, 7]
    KEe[11, 5] = KEe[5, 11]
    KEe[11, 7] = KEe[7, 11]
    KEe[10, 8] = KEe[8, 10]

    return KEe


def kg_e(GRS, N, Li):

    nn = N / Li
    teta = 12 * GRS.Es * GRS.secI / GRS.Gs / (GRS.secA * 2 / np.pi) / Li ** 2  # Asy=2A/pi
    kg1 = nn * (6 / 5 + 2 * teta + teta ** 2) / (1 + teta) ** 2
    kg2 = nn * (Li / 10 / (1 + teta) ** 2)
    kg3 = nn * GRS.secIt / GRS.secA  # ?
    kg4 = nn * (2 * Li ** 2 / 15 + Li ** 2 * teta / 6 + Li ** 2 * teta ** 2 / 12) / (1 + teta) ** 2
    kg5 = nn * (-Li ** 2 / 30 - Li ** 2 * teta / 6 - Li ** 2 * teta ** 2 / 12) / (1 + teta) ** 2

    KGe = np.zeros((12, 12))

    KGe[1, 1] = kg1
    KGe[1, 5] = kg2
    KGe[1, 7] = - kg1
    KGe[1, 11] = kg2
    KGe[2, 2] = kg1
    KGe[2, 4] = - kg2
    KGe[2, 8] = - kg1
    KGe[2, 10] = - kg2
    KGe[3, 3] = kg3
    KGe[3, 9] = - kg3
    KGe[4, 4] = kg4
    KGe[4, 8] = kg2
    KGe[4, 10] = kg5
    KGe[5, 5] = kg4
    KGe[5, 7] = - kg2
    KGe[5, 11] = kg5
    KGe[7, 7] = kg1
    KGe[7, 11] = - kg2
    KGe[8, 8] = kg1
    KGe[8, 10] = kg2
    KGe[9, 9] = kg3
    KGe[10, 10] = kg4
    KGe[11, 11] = kg4

    KGe[5, 1] = KGe[1, 5]
    KGe[7, 1] = KGe[1, 7]
    KGe[11, 1] = KGe[1, 11]
    KGe[4, 2] = KGe[2, 4]
    KGe[8, 2] = KGe[2, 8]
    KGe[10, 2] = KGe[2, 10]

    KGe[9, 3] = KGe[3, 9]
    KGe[8, 4] = KGe[4, 8]
    KGe[10, 4] = KGe[4, 10]
    KGe[7, 5] = KGe[5, 7]
    KGe[11, 5] = KGe[5, 11]
    KGe[11, 7] = KGe[7, 11]
    KGe[10, 8] = KGe[8, 10]

    return KGe


def transf_mx(Cx, Cy, Cz):

    Tv = np.zeros((3,3))

    Cxz = (Cx ** 2 + Cz ** 2) ** 0.5
    c = 1  # alfa=0
    s = 0

    # exception if vertical
    if Cx == 0 and Cz == 0 and Cxz == 0:
        Tv[0, 1] = Cy
        Tv[1, 0] = -Cy
        Tv[2, 2] = 1
    else:
        Tv[0, 0] = Cx
        Tv[0, 1] = Cy
        Tv[0, 2] = Cz
        Tv[1, 0] = (-Cx * Cy * c - Cz * s) / Cxz
        Tv[1, 1] = Cxz * c
        Tv[1, 2] = (-Cy * Cz * c + Cx * s) / Cxz
        Tv[2, 0] = (Cx * Cy * s - Cz * c) / Cxz
        Tv[2, 1] = -Cxz * s
        Tv[2, 2] = (Cy * Cz * s + Cx * c) / Cxz

    Te = np.zeros((3 * 4, 3 * 4))
    for i in range(4):
        Te[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = Tv

    return Te, np.transpose(Te)


def compile_mx(GRS, NA):

    nbN = GRS.nbNsAll

    KE = np.zeros((nbN * 6, nbN * 6))
    KG = np.zeros((nbN * 6, nbN * 6))

    for i in range(GRS.nbElAll):
        sID = GRS.lsAll.sID[i]
        eID = GRS.lsAll.eID[i]
        dx = GRS.nsAll.x[eID] - GRS.nsAll.x[sID]
        dy = GRS.nsAll.y[eID] - GRS.nsAll.y[sID]
        dz = GRS.nsAll.z[eID] - GRS.nsAll.z[sID]
        le = (dx**2 + dy**2 + dz**2)**0.5
        Cx = dx / le
        Cy = dy / le
        Cz = dz / le
        T, Tt = transf_mx(Cx, Cy, Cz)
        kei = Tt @ k_e(GRS, le) @ T
        kgi = Tt @ kg_e(GRS, NA[i], le) @ T
        KE[sID * 6:(sID + 1) * 6, sID * 6:(sID + 1) * 6] += kei[0:6,  0:6]
        KE[sID * 6:(sID + 1) * 6, eID * 6:(eID + 1) * 6] += kei[0:6,  6:12]
        KE[eID * 6:(eID + 1) * 6, sID * 6:(sID + 1) * 6] += kei[6:12, 0:6]
        KE[eID * 6:(eID + 1) * 6, eID * 6:(eID + 1) * 6] += kei[6:12, 6:12]
        KG[sID * 6:(sID + 1) * 6, sID * 6:(sID + 1) * 6] += kgi[0:6,  0:6]
        KG[sID * 6:(sID + 1) * 6, eID * 6:(eID + 1) * 6] += kgi[0:6,  6:12]
        KG[eID * 6:(eID + 1) * 6, sID * 6:(sID + 1) * 6] += kgi[6:12, 0:6]
        KG[eID * 6:(eID + 1) * 6, eID * 6:(eID + 1) * 6] += kgi[6:12, 6:12]

    return KE, KG


def add_support(GRS):
    # spt: all boundary points are fixed for disp., 1: oldalnyomasos, 2: oldalnyomasmentes,

    nbN = GRS.nbNsAll

    KR = np.zeros((nbN * 6, nbN * 6))
    fix = 1e10

    if GRS.SupType == 0:  # all boundary points fixed for deformations
        for i in range(GRS.nbBns):
            KR[GRS.bn[i] * 6 + 0, GRS.bn[i] * 6 + 0] = fix
            KR[GRS.bn[i] * 6 + 1, GRS.bn[i] * 6 + 1] = fix
            KR[GRS.bn[i] * 6 + 2, GRS.bn[i] * 6 + 2] = fix
    elif GRS.SupType == 1:  # oldalnyomasos
        for i in range(len(GRS.bnX)):
            KR[GRS.bnX[i] * 6 + 0, GRS.bnX[i] * 6 + 0] = fix
            KR[GRS.bnX[i] * 6 + 2, GRS.bnX[i] * 6 + 2] = fix
        for i in range(len(GRS.bnY)):
            KR[GRS.bnY[i] * 6 + 1, GRS.bnY[i] * 6 + 1] = fix
            KR[GRS.bnY[i] * 6 + 2, GRS.bnY[i] * 6 + 2] = fix
        for i in range(len(GRS.bnC)):
            KR[GRS.bnC[i] * 6 + 0, GRS.bnC[i] * 6 + 0] = fix
            KR[GRS.bnC[i] * 6 + 1, GRS.bnC[i] * 6 + 1] = fix
            KR[GRS.bnC[i] * 6 + 2, GRS.bnC[i] * 6 + 2] = fix
    elif GRS.SupType == 2:  # oldalnyomasmentes
        for i in range(len(GRS.bnX)):
            KR[GRS.bnX[i] * 6 + 2, GRS.bnX[i] * 6 + 2] = fix
        for i in range(len(GRS.bnY)):
            KR[GRS.bnY[i] * 6 + 2, GRS.bnY[i] * 6 + 2] = fix
        #for i in range(len(GRS.bnC)):
        #    KR[GRS.bnC[i] * 6 + 0, GRS.bnC[i] * 6 + 0] = fix
        #    KR[GRS.bnC[i] * 6 + 1, GRS.bnC[i] * 6 + 1] = fix
        #    KR[GRS.bnC[i] * 6 + 2, GRS.bnC[i] * 6 + 2] = fix
        KR[GRS.bnC[0] * 6 + 0, GRS.bnC[0] * 6 + 0] = fix
        KR[GRS.bnC[0] * 6 + 1, GRS.bnC[0] * 6 + 1] = fix
        KR[GRS.bnC[2] * 6 + 0, GRS.bnC[2] * 6 + 0] = fix
        KR[GRS.bnC[0] * 6 + 2, GRS.bnC[0] * 6 + 2] = fix
        KR[GRS.bnC[1] * 6 + 2, GRS.bnC[1] * 6 + 2] = fix
        KR[GRS.bnC[2] * 6 + 2, GRS.bnC[2] * 6 + 2] = fix
        KR[GRS.bnC[3] * 6 + 2, GRS.bnC[3] * 6 + 2] = fix
    elif GRS.SupType == 3:  # felmerev #NOT WORKING YET
        pass
    elif GRS.SupType == 4:  # sarkok
        for i in range(len(GRS.bnC)):
            KR[GRS.bnC[i] * 6 + 0, GRS.bnC[i] * 6 + 0] = fix
            KR[GRS.bnC[i] * 6 + 1, GRS.bnC[i] * 6 + 1] = fix
            KR[GRS.bnC[i] * 6 + 2, GRS.bnC[i] * 6 + 2] = fix
    elif GRS.SupType == 5:  # dome
        for i in range(GRS.nbBns):
            sa = 2 * GRS.nsAll.y[GRS.bn[i]] / GRS.span
            ca = 2 * GRS.nsAll.x[GRS.bn[i]] / GRS.span
            KR[GRS.bn[i] * 6 + 0, GRS.bn[i] * 6 + 0] = fix * ca
            KR[GRS.bn[i] * 6 + 1, GRS.bn[i] * 6 + 1] = fix * sa
            KR[GRS.bn[i] * 6 + 2, GRS.bn[i] * 6 + 2] = fix
    return KR


def imperfection(GRS, eigenNB, amp, imptype=0, neg=0):

    GRSimp = copy.deepcopy(GRS)
    GRSeig = copy.deepcopy(GRS)

    GRSeig.GeomNL=0
    GRSeig.MatNL=0
    tk.BuildOpsModel(GRSeig)
    NA = get_normal_force(GRSeig)
    KE, KG = compile_mx(GRSeig, NA)
    KR = add_support(GRSeig)

    alfa=-111111

    if neg==1:
        mult=-1 # imperfection in the other direction
    else: mult=1

    try:
        if imptype==0:
            w, v = scsp.linalg.eigsh(-KG, 1, KE + KR)  # sparse solver, quicker, only first eigenvalue, abs lowest - ??
            alfa = 1 / w[eigenNB]
            #mult = - mult  # required only for this solver
        elif imptype==2:
            w, v = sc.eigh(-KG, KE+KR, eigvals=(0,0)) #lowest positive eigenvalue
            alfa = 1 / w[eigenNB]
        elif imptype==3:
            w, v = sc.eigh(-KG, KE+KR) # all eigenvalues
            alfa = 1 / w[eigenNB]
            while alfa > 0 and eigenNB < 100:
                eigenNB += 1
                alfa = 1 / w[eigenNB]

        ex = np.array(v[0::6, eigenNB]) * mult
        ey = np.array(v[1::6, eigenNB]) * mult
        ez = np.array(v[2::6, eigenNB]) * mult

        maxe=np.max((ex**2+ey**2+ez**2)**0.5)
        exi = ex / maxe * amp
        eyi = ey / maxe * amp
        ezi = ez / maxe * amp

        #update GRS geometry
        GRSimp.nsAll.x += exi
        GRSimp.nsAll.y += eyi
        GRSimp.nsAll.z += ezi

    finally:
        GRSeig.GeomNL = GRS.GeomNL
        GRSeig.MatNL  = GRS.MatNL

    return alfa, GRSimp


# DISPLACED SHAPE AS IMPERFECTION

def imperfectionDisp(GRS, amp, neg=0):

    GRSimp = copy.deepcopy(GRS)
    GRSeig = copy.deepcopy(GRS)

    GRSeig.GeomNL = 0
    GRSeig.MatNL = 0
    tk.BuildOpsModel(GRSeig)
    dA = get_disp(GRSeig)

    if neg==1:
        mult=-1
    else: mult=1

    ex = np.array(dA[:,0])*mult
    ey = np.array(dA[:,1])*mult
    ez = np.array(dA[:,2])*mult

    maxe = np.max((ex ** 2 + ey ** 2 + ez ** 2) ** 0.5)
    exi = ex / maxe * amp
    eyi = ey / maxe * amp
    ezi = ez / maxe * amp

    # update GRS geometry
    GRSimp.nsAll.x += exi
    GRSimp.nsAll.y += eyi
    GRSimp.nsAll.z += ezi

    return GRSimp