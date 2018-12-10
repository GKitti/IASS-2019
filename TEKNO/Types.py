import units as un
import numpy as np


class GSnodes:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class GSlines:
    def __init__(self, startPointID, endPointID):
        self.sID = startPointID
        self.eID = endPointID


class GridShell:
    def __init__(self, span=20, height=3, secD=80, secT=3, gN=10, nb=4, geomtype=1, gR=5, shape=410, grid=''):
        # GEOMETRY SETTINGS
        self.span = span  # m
        self.height = height  # m
        self.lperh = span/height
        self.secD = secD * un.mm
        self.secT = secT * un.mm
        self.secA = np.pi * (secD ** 2 - (secD - 2 * secT) ** 2) / 4 * un.mm**2
        self.secI = np.pi * (secD ** 4 - (secD - 2 * secT) ** 4) / 64 * un.mm**4
        self.secIt = np.pi * (secD ** 4 - (secD - 2 * secT) ** 4) / 32 * un.mm**4
        self.gN = gN  # grid density // Kiewitt sections
        self.gR = gR # relevant only for Kiewitt: number of rings
        self.shape = shape  # relevant only for relaxed shape (Grasshopper)
        self.grid = grid  # relevant only for relaxed shape (Grasshopper) ''=projected grid, '2'=uniform grid
        self.Nb = nb  # number of finite elements per beam
        self.ns = GSnodes(0, 0, 0)  # array of nodes
        self.ls = GSlines(0, 0)  # array of line node IDs
        self.nsAll = GSnodes(0, 0, 0)  # array of all FE nodes
        self.lsAll = GSlines(0, 0)  # array of all FE line node IDs
        self.bn = []  # boundary node IDs
        self.bnX = []  # boundary node IDs that may be fixed in X direction
        self.bnY = []  # boundary node IDs that may be fixed in X direction
        self.bnC = []  # boundary node IDs in corners
        self.nbn = []  # non boundary node IDs
        self.nbNs = 0  # number of nodes in grid shell
        self.nbBns = 0  # number of boundary nodes in grid shell
        self.nbnBns = 0  # number of non-boundary nodes in grid shell
        self.nbEl = 0  # number of elements in grid shell
        self.nbNsAll = 0  # number of all nodes in grid shell
        self.nbElAll = 0  # number of all elements in grid shell
        self.maxNsID = 0  # ID of highest node
        self.Lav = 0  # average element lengths (not finite element lengths)
        self.LSum = 0  # total element lengths
        self.lmax = 0  # maximum element length
        self.covL = 0 # coefficient of variation
        self.GeomType = geomtype  # grid pattern/layout
        self.PlanArea=0
        self.delNodes = []
        # ANALYSIS SETTINGS
        self.GeomNL = 1 # geometric non-linearity, default on
        self.Es = 210. * un.GPa
        self.Gs = self.Es / (2 * 1.3)
        self.MatNL = 0           # elastic material
        self.Steps = 50          # load steps
        self.MinStepSize = 0.1   # [kN] Runiteration stops if load step is smaller than this
        self.reRunIfNotConv = 1  # rerun analysis if load was too high and analysis did not converge
        self.SupType = 1  # all boundary points are fixed for disp., 1: oldalnyomasos, 2: oldalnyomasmentes, 3: elastic
        self.eqB = 0  # equivalent shell bending stiffness
        self.eqT = 0  # equivalent shell membrane stiffness
        self.teq = 0  # equivalent shell thickness
        self.Eeq = 0  # equivalent shell modulus
        self.Ncr = [] # Euler critical load
        self.LoadType = 0  # 0: full, 1,2: half

    def GetBoundaryPts(self):
        for i in range(self.nbNs):
            if self.ns.z[i] < 0.1 * un.mm:
                self.bn.append(i)
                self.nbBns += 1
                if self.GeomType in {1,2}:
                    if abs(self.ns.x[i]) > self.span/2 - 0.1 * un.mm:
                        if abs(self.ns.y[i]) > self.span/2 - 0.1 * un.mm:
                            self.bnC.append(i)
                        else:
                            self.bnX.append(i)
                    elif abs(self.ns.y[i]) > self.span/2 - 0.1 * un.mm:
                        self.bnY.append(i)
                else:
                    if (self.ns.x[i] > self.span - 0.1 * un.mm) or (self.ns.x[i] < 0.1 * un.mm):
                        if (self.ns.y[i] > self.span - 0.1 * un.mm) or (self.ns.y[i] < 0.1 * un.mm):
                            self.bnC.append(i)
                        else:
                            self.bnX.append(i)
                    elif (self.ns.y[i] > self.span - 0.1 * un.mm) or (self.ns.y[i] < 0.1 * un.mm):
                        self.bnY.append(i)
            else:
                self.nbn.append(i)
                self.nbnBns += 1

    def GetTopNode(self):
        for i in range(self.nbNsAll):
            self.maxNsID = np.argmax(self.nsAll.z)


class IterObj:
    def __init__(self, Span, Height, gN, nb, LoadType, qz, MatNL=False, GeomNL=0, Steps=50, n=1, m=1, DStart=100, DStep=20, tStart=10, tStep=5, printon=0):
        self.Span = Span  # m
        self.Height = Height  # m
        self.gN = gN  # grid density
        self.nb = nb  # number of finite elements per beam
        self.LoadType = LoadType
        self.qz = qz
        self.MatNL = MatNL
        self.GeomNL = GeomNL
        self.Steps = Steps
        self.n = n
        self.m = m
        self.DStart = DStart
        self.DStep = DStep
        self.tStart = tStart
        self.tStep = tStep
        self.printon = printon
