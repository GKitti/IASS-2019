import Types as tp
import numpy as np
from scipy import optimize
import Tekno as tk

# C R E A T E   G R I S S H E L L   G E O M E T R Y

def Geometry1(GRS):
    # ez a racsozas NEM olyan, mint ami a doktorimban volt

    # Grid
    xV = np.array([np.arange(-GRS.span / 2, GRS.span / 2 + GRS.span / GRS.gN, GRS.span / GRS.gN)])
    yV = np.array([np.arange(-GRS.span / 2, GRS.span / 2 + GRS.span / GRS.gN, GRS.span / GRS.gN)])

    # Points
    np2 = int(GRS.gN / 2)
    idx = np.arange(0, GRS.gN + 1, 2)
    idy = np.zeros(np2 + 1, dtype=np.int16)
    for i in range(np2):
        idx = np.concatenate((idx, np.array([0])))
        idx = np.concatenate((idx, np.arange(1, GRS.gN, 2)))
        idx = np.concatenate((idx, np.array([GRS.gN])))
        idy = np.concatenate((idy, (2 * i + 1) * np.ones(np2 + 2, dtype=np.int16)))
        idx = np.concatenate((idx, np.arange(0, GRS.gN + 1, 2)))
        idy = np.concatenate((idy, (2 * i + 2) * np.ones(np2 + 1, dtype=np.int16)))
    xMx = xV[0, idx]
    yMx = yV[0, idy]

    # Height
    zMx = np.ravel(GRS.height * (1 - yMx ** 2 / (GRS.span / 2) ** 2) * (1 - xMx.T ** 2 / (GRS.span / 2) ** 2))

    # Lines
    nbP = xMx.shape[0]
    idr = np.arange(nbP)

    # diagonals n=3: 0,7,1,8,4,11,5,12
    id1 = idr[:-GRS.gN - 3:GRS.gN + 3]
    for i in range(1, np2):
        id1 = np.concatenate((id1, idr[i:-GRS.gN - 3:GRS.gN + 3]))
    for i in range(np2 + 2, GRS.gN + 2):
        id1 = np.concatenate((id1, idr[i::GRS.gN + 3]))

    # diagonals n=3: 1,8,2,9,4,11,5,12
    id1 = np.concatenate((id1, idr[1:-GRS.gN - 3:GRS.gN + 3]))
    for i in range(1, np2):
        id1 = np.concatenate((id1, idr[i + 1:-GRS.gN - 3:GRS.gN + 3]))
    for i in range(np2 + 2, GRS.gN + 2):
        id1 = np.concatenate((id1, idr[i::GRS.gN + 3]))
    id2 = nbP - 1 - id1[::-1]

    # horizontal
    id1 = np.concatenate((id1, idr[:-np2 - 1:GRS.gN + 3]))
    id1 = np.concatenate((id1, idr[np2 + 1::GRS.gN + 3]))
    id1 = np.concatenate((id1, idr[np2:-np2 - 1:GRS.gN + 3]))
    id1 = np.concatenate((id1, idr[GRS.gN + 2::GRS.gN + 3]))
    id2 = np.concatenate((id2, idr[np2 + 1::GRS.gN + 3]))
    id2 = np.concatenate((id2, idr[GRS.gN + 3::GRS.gN + 3]))
    id2 = np.concatenate((id2, idr[GRS.gN + 2::GRS.gN + 3]))
    id2 = np.concatenate((id2, idr[GRS.gN + 3 + np2::GRS.gN + 3]))

    # vertical
    nbL = id1.shape[0]
    for i in range(0, np2):
        id1 = np.concatenate((id1, idr[i::GRS.gN + 3]))
    id1 = np.concatenate((id1, idr[np2 + 1::GRS.gN + 3]))
    for i in range(0, np2):
        id1 = np.concatenate((id1, idr[np2 + 2 + i::GRS.gN + 3]))
    id2 = np.concatenate((id2, id1[nbL::] + 1))

    # Visualize
    # line1x = xMx[id1]
    # line1y = yMx[id1]
    # line2x = xMx[id2]
    # line2y = yMx[id2]
    #_ = plt.plot(np.ravel(xMx), np.ravel(yMx), 'go')  # dot plot
    # x = [line1x, line2x]
    # y = [line1y, line2y]
    #_ = plt.plot(x, y)  # line plot
    #_ = plt.gca().set_aspect('equal')

    GRS.nbNs, = xMx.shape
    GRS.nbEl, = id1.shape
    GRS.ns.x = xMx
    GRS.ns.y = yMx
    GRS.ns.z = zMx
    GRS.ls.sID = id1
    GRS.ls.eID = id2
    GRS.GetBoundaryPts()
    GRS.GetTopNode()
    GRS.PlanArea = GRS.span**2


def Geometry2(GRS):
    # ez a racsozas olyan, mint ami a doktorimban volt - de most csak egy parameteres

    # Grid
    xV = np.array([np.arange(-GRS.span / 2, GRS.span / 2 + GRS.span / GRS.gN / 2, GRS.span / GRS.gN / 2)])
    yV = np.array([np.arange(-GRS.span / 2, GRS.span / 2 + GRS.span / GRS.gN, GRS.span / GRS.gN)])

    # Points
    np2 = int(GRS.gN / 2)
    idx = np.arange(0, GRS.gN * 2 + 1, 2)
    idy = np.zeros(GRS.gN + 1, dtype=np.int16)
    for i in range(np2):
        idx = np.concatenate((idx, np.array([0])))
        idx = np.concatenate((idx, np.arange(1, GRS.gN * 2, 2)))
        idx = np.concatenate((idx, np.array([GRS.gN * 2])))
        idy = np.concatenate((idy, (2 * i + 1) * np.ones(GRS.gN + 2, dtype=np.int16)))
        idx = np.concatenate((idx, np.arange(0, GRS.gN * 2 + 1, 2)))
        idy = np.concatenate((idy, (2 * i + 2) * np.ones(GRS.gN + 1, dtype=np.int16)))
    xMx = xV[0, idx]
    yMx = yV[0, idy]
    # Height
    zMx = np.ravel(GRS.height * (1 - yMx ** 2 / (GRS.span / 2) ** 2) * (1 - xMx.T ** 2 / (GRS.span / 2) ** 2))

    # Lines
    nbP = xMx.shape[0]
    idr = np.arange(nbP)
    nbPt2r = 2 * GRS.gN + 3  # number of nodes in 2 rows

    # diagonals
    id1 = idr[:-GRS.gN - 3:nbPt2r]
    for i in range(1, GRS.gN):
        id1 = np.concatenate((id1, idr[i:-GRS.gN - 3:nbPt2r]))
    for i in range(GRS.gN + 2, GRS.gN * 2 + 2):
        id1 = np.concatenate((id1, idr[i::nbPt2r]))

    # diagonals
    id1 = np.concatenate((id1, idr[1:-GRS.gN - 3:nbPt2r]))
    for i in range(1, GRS.gN):
        id1 = np.concatenate((id1, idr[i + 1:-GRS.gN - 3:nbPt2r]))
    for i in range(GRS.gN + 2, GRS.gN * 2 + 2):
        id1 = np.concatenate((id1, idr[i::nbPt2r]))
    id2 = nbP - 1 - id1[::-1]

    # horizontal
    id1 = np.concatenate((id1, idr[:-GRS.gN - 2:nbPt2r]))
    id1 = np.concatenate((id1, idr[GRS.gN + 1::nbPt2r]))
    id1 = np.concatenate((id1, idr[GRS.gN:-GRS.gN - 1:nbPt2r]))
    id1 = np.concatenate((id1, idr[nbPt2r - 1::nbPt2r]))
    id2 = np.concatenate((id2, idr[GRS.gN + 1::nbPt2r]))
    id2 = np.concatenate((id2, idr[nbPt2r::nbPt2r]))
    id2 = np.concatenate((id2, idr[nbPt2r - 1::nbPt2r]))
    id2 = np.concatenate((id2, idr[nbPt2r + GRS.gN::nbPt2r]))

    # vertical
    for i in range(0, GRS.gN):
        id1 = np.concatenate((id1, idr[i::nbPt2r]))
        id2 = np.concatenate((id2, idr[i + 1::nbPt2r]))
    for i in range(0, GRS.gN + 1):
        id1 = np.concatenate((id1, idr[GRS.gN + 1 + i::nbPt2r]))
        id2 = np.concatenate((id2, idr[GRS.gN + 1 + i + 1::nbPt2r]))


    # Visualize
    # line1x = xMx[id1]
    # line1y = yMx[id1]
    # line2x = xMx[id2]
    # line2y = yMx[id2]
    #_ = plt.plot(np.ravel(xMx), np.ravel(yMx), 'go')  # dot plot
    # x = [line1x, line2x]
    # y = [line1y, line2y]
    #_ = plt.plot(x, y)  # line plot
    #_ = plt.gca().set_aspect('equal')

    GRS.nbNs, = xMx.shape
    GRS.nbEl, = id1.shape
    GRS.ns.x = xMx
    GRS.ns.y = yMx
    GRS.ns.z = zMx
    GRS.ls.sID = id1
    GRS.ls.eID = id2
    GRS.GetBoundaryPts()
    GRS.GetTopNode()
    GRS.PlanArea = GRS.span**2


def Geometry3(GRS):
    # Kiewitt
    shape = 1  # 0: sphere, 1: catenary

    # Nodes
    nbNs = int(1 + GRS.gN * GRS.gR * (GRS.gR + 1) / 2)
    xMx = np.zeros(nbNs)
    yMx = np.zeros(nbNs)
    zMx = np.zeros(nbNs)
    nbR = int(GRS.gN * GRS.gR * (GRS.gR + 1) / 2)  # number of ring lines
    nbT = GRS.gN * (GRS.gR - 1) * GRS.gR  # number of triangle lines
    nbM = GRS.gN * GRS.gR  # number of meridian lines
    nbEl = nbR + nbT + nbM  # number of lines
    sID = np.zeros(nbEl, dtype=np.int16)
    eID = np.zeros(nbEl, dtype=np.int16)
    zMx[0] = GRS.height  # base top node

    # nodes on meridians
    if shape == 0:
        R = GRS.span ** 2 / 8 / GRS.height + GRS.height / 2
        beta = np.arcsin(GRS.span / 2 / R)
        beta = beta / GRS.gR / 2
    else:
        xxL, zzL = catenary(GRS.span, GRS.height, GRS.gR)
        #print(xxL)
        #print(zzL)

    k = 0
    for i in range(GRS.gR):
        if shape == 0:
            s = np.sin(beta * (i + 1)) * R * 2
            xx = np.cos(beta * (i + 1)) * s
            zz = GRS.height - np.sin(beta * (i + 2)) * s
        else:
            xx = xxL[i]
            zz = zzL[i]
        alpha = 2 * np.pi / GRS.gN / (i + 1)
        for j in range(GRS.gN * (i + 1)):
            # rotation
            k += 1
            xMx[k] = -np.sin(alpha * j) * xx
            yMx[k] = np.cos(alpha * j) * xx
            zMx[k] = zz

            # rings
            sID[k - 1] = k
            if j < GRS.gN * (i + 1) - 1:
                eID[k - 1] = k + 1
            else:
                eID[k - 1] = k - GRS.gN * (i + 1) + 1

            # triangles
            if i < GRS.gR - 1:
                ind = int(nbR + (i + 1) / 2 * i * GRS.gN + j)
                sID[ind] = (i + 1) * i * GRS.gN / 2 + j + 1
                eID[ind] = (i + 2) * (i + 1) * GRS.gN / 2 + j + 2 + int(j / (i + 1))
                # print(sID[ind], eID[ind])
                ind = ind + int(nbT / 2)
                sID[ind] = (i + 1) * i * GRS.gN / 2 + j + 1
                if j == GRS.gN * (i + 1) - 1:
                    eID[ind - GRS.gN * (i + 1) + 1] = (i + 2) * (i + 1) * GRS.gN / 2 + j + 2 + int(j / (i + 1))
                else:
                    eID[ind + 1] = (i + 2) * (i + 1) * GRS.gN / 2 + j + 2 + int(j / (i + 1))

        # meridians/ribs
        for j in range(GRS.gN):
            ind = nbR + nbT + i * GRS.gN + j
            if i == 0:
                sID[ind] = 0
            else:
                sID[ind] = i * j + (i - 1) * i * GRS.gN / 2 + 1
            eID[ind] = (i + 1) * j + i * (i + 1) * GRS.gN / 2 + 1

    GRS.nbNs, = xMx.shape
    GRS.nbEl, = sID.shape
    GRS.ns.x = xMx
    GRS.ns.y = yMx
    GRS.ns.z = zMx
    GRS.ls.sID = sID
    GRS.ls.eID = eID
    GRS.GetBoundaryPts()
    GRS.GetTopNode()
    GRS.PlanArea = np.pi / 4 * GRS.span**2


def Geometry4(GRS, geompath):

    filename = geompath+'\\{:d}x{:d}\\D{:d}{:d}\\h{:d}\\{:d}\\geometry{:s}.txt'.format(GRS.span, GRS.span, GRS.gN, GRS.gN, int(GRS.lperh), GRS.shape, GRS.grid)

    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    #Height = float(content[38])  # m
    #nb = int(content[44]) + 1  # nbMidPt
    nPt = int(content[25])  # osszes csp szama
    #nPtB = int(content[26])  # osszes perem csp szama
    #nPtAll = int(content[23])  # osszes pont szama
    #Area = int(content[27])  # surface area [m2]
    GRS.Lav = float(content[36])  # m
    MinL = float(content[34])  # m
    MaxL = float(content[35])  # m
    GRS.LSum = float(content[45])  # m

    px = [float(i) / 1000 for i in content[0].split(",")]  # nPtAll
    py = [float(i) / 1000 for i in content[1].split(",")]  #
    pz = [float(i) / 1000 for i in content[2].split(",")]  #

    GRS.ns.x = px[0:nPt]
    GRS.ns.y = py[0:nPt]
    GRS.ns.z = pz[0:nPt]
    GRS.nbNs = nPt
    GRS.ls.sID=[]
    GRS.ls.eID=[]

    GRS_an = tp.GridShell(GRS.span, GRS.height, 100, 5, GRS.gN, 1, 2)  # analytic
    tk.CreateGeom(GRS_an)
    for k in range(GRS_an.nbEl):
        ks = GRS_an.ls.sID[k]
        ke = GRS_an.ls.eID[k]
        ii=-1
        dsmin=1e10
        for i in range(GRS.nbNs):
            dxs = GRS.ns.x[i] - GRS_an.ns.x[ks] - GRS.span / 2
            dys = GRS.ns.y[i] - GRS_an.ns.y[ks] - GRS.span / 2
            dzs = GRS.ns.z[i] - GRS_an.ns.z[ks]
            ls = (dxs ** 2 + dys ** 2 + dzs ** 2) ** 0.5
            if dsmin > ls:
                dsmin = ls
                ii = i
        jj = -1
        demin = 1e10
        for j in range(GRS.nbNs):
            dxe = GRS.ns.x[j] - GRS_an.ns.x[ke] - GRS.span / 2
            dye = GRS.ns.y[j] - GRS_an.ns.y[ke] - GRS.span / 2
            dze = GRS.ns.z[j] - GRS_an.ns.z[ke]
            le = (dxe ** 2 + dye ** 2 + dze ** 2) ** 0.5
            if demin > le:
                demin = le
                jj = j

        GRS.ls.sID.append(ii)
        GRS.ls.eID.append(jj)

    GRS.nbEl = len(GRS.ls.sID)
    GRS.GetBoundaryPts()
    GRS.GetTopNode()
    GRS.PlanArea = GRS.span ** 2


def SplitBeams(GRS):
    wn = GRS.Nb
    if wn == 1:
        wNs = GRS.nbNs - 1
        wEl = 0
        GRS.nbNsAll = GRS.nbNs
        GRS.nbElAll = GRS.nbEl
        GRS.nsAll.x = GRS.ns.x.copy()  # copy the existing nodes
        GRS.nsAll.y = GRS.ns.y.copy()  # copy the existing nodes
        GRS.nsAll.z = GRS.ns.z.copy()  # copy the existing nodes
        GRS.lsAll.sID = GRS.ls.sID.copy()  # copy the existing lines
        GRS.lsAll.eID = GRS.ls.eID.copy()  # copy the existing lines
    else:
        wNs = GRS.nbNs
        wEl = 0
        GRS.nbNsAll = GRS.nbNs + (wn-1) * GRS.nbEl
        GRS.nbElAll = wn * GRS.nbEl
        wx = np.zeros(GRS.nbNsAll)
        wy = np.zeros(GRS.nbNsAll)
        wz = np.zeros(GRS.nbNsAll)
        wx[:GRS.nbNs:] = GRS.ns.x[::].copy()  # first we copy the existing nodes
        wy[:GRS.nbNs:] = GRS.ns.y[::].copy()  # first we copy the existing nodes
        wz[:GRS.nbNs:] = GRS.ns.z[::].copy()  # first we copy the existing nodes   #----------------------????????
        GRS.lsAll.sID = np.zeros(GRS.nbElAll, dtype=np.int16)
        GRS.lsAll.eID = np.zeros(GRS.nbElAll, dtype=np.int16)
        for i in range(GRS.nbEl):
            sPtID = GRS.ls.sID[i]
            ePtID = GRS.ls.eID[i]
            sPt = tp.GSnodes(GRS.ns.x[sPtID], GRS.ns.y[sPtID], GRS.ns.z[sPtID])
            ePt = tp.GSnodes(GRS.ns.x[ePtID], GRS.ns.y[ePtID], GRS.ns.z[ePtID])
            for j in range(wn):
                if j != wn-1:  # in every step we create a new node (the new end node), except in the last step
                    # new nodes
                    wx[wNs] = sPt.x + (j+1) * (ePt.x - sPt.x) / wn
                    wy[wNs] = sPt.y + (j+1) * (ePt.y - sPt.y) / wn
                    wz[wNs] = sPt.z + (j+1) * (ePt.z - sPt.z) / wn
                    wNs += 1
                # new lines
                if j == 0:
                    GRS.lsAll.sID[wEl] = sPtID
                elif j == wn - 1:
                    GRS.lsAll.sID[wEl] = wNs - 1  # the node we created in the step before this
                else:
                    GRS.lsAll.sID[wEl] = wNs - 2  # the node we created in the step before this
                if j == wn-1:
                    GRS.lsAll.eID[wEl] = ePtID
                else:
                    GRS.lsAll.eID[wEl] = wNs-1  # the node we have just created
                wEl += 1
        GRS.nsAll.x = wx
        GRS.nsAll.y = wy
        GRS.nsAll.z = wz


def catenary(L, H, n):
    # catenary shape
    x = [0, L / 2]
    y = [H, 0]
    params, params_covariance = optimize.curve_fit(cate, x, y)

    # arc-length
    s = 0
    div = 1000
    for i in range(0, div):
        s = s + np.sqrt(1 + (np.cosh((L / 2 / div * i) / params[1])) ** 2) * L / 2 / div

    # roughly divide
    for i in range(10):
        d = s / (n + i)
        px0, py0, px1, py1 = 0, H, L / 2, 0
        dlistr, xxr, zzr = slice_cate(d, n, px0, py0, px1, py1, params[0], params[1], 1000, 1000, 0.01, 0.1)
        if len(dlistr) == n: break
    for i in range(n):
        print('{:.3f} '.format(dlistr[i]), end="", flush=True)
    print()

    # refine division
    try:
        d = sum(dlistr) / len(dlistr)
        dlist, xx, zz = slice_cate(d, n, px0, py0, px1, py1, params[0], params[1], 1000, 10000, 0.00001, 0.01)
        for i in range(n):
            print('{:.3f} '.format(dlist[i]), end="", flush=True)
        print()
    except:
        dlist, xx, zz = dlistr, xxr, zzr


    # visualize
    # x=np.linspace(px0,px1,100)
    # _ = plt.gca().set_aspect('equal')
    # plt.plot(x,cate(x,params[0],params[1]),'r--')
    # plt.plot(xx, zz,'ro')

    return xx, zz


def cate(x, a1, a2):
    return a1 - a2 * np.cosh(x/a2)


def slice_cate(d, n, px0, py0, px1, py1, params0, params1, l1, l2, delta, ok):
    for i in range(l1):
        x0 = px0
        y0 = py0
        k = 0
        count = 0
        dlist = []
        xlist = []
        ylist = []
        for x in np.linspace(px0, px1, l2):
            y = cate(x, params0, params1)
            dist = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
            if dist < d:
                k += 1
            else:
                count += 1
                xlist.append(x)
                ylist.append(y)
                dlist.append(dist)
                x0 = x
                y0 = y
        rem = ((px1 - x0) ** 2 + (py1 - y0) ** 2) ** 0.5
        if count > n:
            d += delta
        else:
            d -= delta
        if abs(rem) < ok: break
        if count > n + 1: break

    rem = ((px1 - x0) ** 2 + (py1 - y0) ** 2) ** 0.5
    if rem < 0:
        dlist.append(dist)
        xlist.append(px1)
        ylist.append(py1)
    else:
        xlist[-1] = px1
        ylist[-1] = py1

    return dlist, xlist, ylist