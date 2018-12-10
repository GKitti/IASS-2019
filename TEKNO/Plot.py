from plotly.graph_objs import Layout, Data, Figure, Marker
from plotly.graph_objs import Scatter3d, Scatter
from plotly import tools
import units as un
import numpy as np


# F I G U R E S


def PlotGeom(GRS, ID, savepic=0):
    if savepic:
        wline=5
        pheight = 2000
        pwidth = 4000
        nwidth=5
    else:
        wline=3
        pheight = 500
        pwidth = 1000
        nwidth=2

    edge_trace = Scatter3d(
        x=[],
        y=[],
        z=[],
        # hoverinfo='none',
        mode='lines',
        line = dict(color='blue', width=wline),
    )

    for i in range(GRS.nbElAll):
        sID = GRS.lsAll.sID[i]
        eID = GRS.lsAll.eID[i]
        x0 = GRS.nsAll.x[sID]
        y0 = GRS.nsAll.y[sID]
        z0 = GRS.nsAll.z[sID]
        x1 = GRS.nsAll.x[eID]
        y1 = GRS.nsAll.y[eID]
        z1 = GRS.nsAll.z[eID]
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]
        edge_trace['z'] += [z0, z1, None]

    node_trace = Scatter3d(
        x=[],
        y=[],
        z=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            color=['red'],
            size=5,
        ),
        line=dict(width=nwidth))

    node_trace['x'].append(GRS.nsAll.x[ID])
    node_trace['y'].append(GRS.nsAll.y[ID])
    node_trace['z'].append(GRS.nsAll.z[ID])

    fig = Figure(data=Data([edge_trace, node_trace]),
                 layout=Layout(
                     showlegend=False,
                     hovermode='closest',
                     margin=dict(b=20, l=5, r=5, t=40),
                     height=pheight,
                     width=pwidth,
                     scene=dict(
                         aspectmode='data',
                         xaxis=dict(
                             showgrid=False,
                             zeroline=False),
                         yaxis=dict(
                             showgrid=False,
                             zeroline=False),
                         zaxis=dict(
                             showgrid=False,
                             zeroline=False), ),
                 ))

    return fig


def PlotDef(GRS, ID, NDisp, scale, small=False, deflOnly=False, mode=0):
    edge_trace1 = Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='lines',
        line=dict(color='grey'),
    )

    edge_trace2 = Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='lines',
        line=dict(color='blue'),
    )

    for i in range(GRS.nbElAll):
        sID = GRS.lsAll.sID[i]
        eID = GRS.lsAll.eID[i]
        x0 = 0
        y0 = 0
        x1 = 0
        y1 = 0
        z0 = 0
        z1 = 0
        if mode in {0, 2, 3}:
            x0 = GRS.nsAll.x[sID]
            x1 = GRS.nsAll.x[eID]
        if mode in {0, 1, 3}:
            y0 = GRS.nsAll.y[sID]
            y1 = GRS.nsAll.y[eID]
        if mode in {0, 1, 2}:
            z0 = GRS.nsAll.z[sID]
            z1 = GRS.nsAll.z[eID]
        edge_trace1['x'] += [x0, x1, None]
        edge_trace1['y'] += [y0, y1, None]
        edge_trace1['z'] += [z0, z1, None]
        if mode in {0, 2, 3}:
            x0 -= scale * NDisp[sID, 0] * un.mm
            x1 -= scale * NDisp[eID, 0] * un.mm
        if mode in {0, 1, 3}:
            y0 -= scale * NDisp[sID, 1] * un.mm
            y1 -= scale * NDisp[eID, 1] * un.mm
        if mode in {0, 1, 2}:
            z1 -= scale * NDisp[eID, 2] * un.mm
            z0 -= scale * NDisp[sID, 2] * un.mm
        edge_trace2['x'] += [x0, x1, None]
        edge_trace2['y'] += [y0, y1, None]
        edge_trace2['z'] += [z0, z1, None]

    if not small:
        for i in range(GRS.nbElAll-1):
            sID = GRS.lsAll.sID[i]
            eID = GRS.lsAll.eID[i]
            x0 = 0
            y0 = 0
            x1 = 0
            y1 = 0
            z0 = 0
            z1 = 0
            if mode in {0, 2, 3}:
                x0 = GRS.nsAll.x[sID]
                x1 = GRS.nsAll.x[eID]
                x0i = x0 - scale * NDisp[sID, 0] * un.mm
                x1i = x1 - scale * NDisp[eID, 0] * un.mm
            if mode in {0, 1, 3}:
                y0 = GRS.nsAll.y[sID]
                y1 = GRS.nsAll.y[eID]
                y0i = y0 - scale * NDisp[sID, 1] * un.mm
                y1i = y1 - scale * NDisp[eID, 1] * un.mm
            if mode in {0, 1, 2}:
                z0 = GRS.nsAll.z[sID]
                z1 = GRS.nsAll.z[eID]
                z0i = z0 - scale * NDisp[sID, 2] * un.mm
                z1i = z1 - scale * NDisp[eID, 2] * un.mm
            edge_trace1['x'] += [x0, x0i, None]
            edge_trace1['y'] += [y0, y0i, None]
            edge_trace1['z'] += [z0, z0i, None]
            edge_trace2['x'] += [x1, x1i, None]
            edge_trace2['y'] += [y1, y1i, None]
            edge_trace2['z'] += [z1, z1i, None]

    node_trace = Scatter3d(
        x=[],
        y=[],
        z=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            color=['red'],
            size=5,
        ),
        line=dict(width=2))

    if mode==1:     node_trace['x'].append(0)
    else:     node_trace['x'].append(GRS.nsAll.x[ID])
    if mode==2:         node_trace['y'].append(0)
    else: node_trace['y'].append(GRS.nsAll.y[ID])
    if mode==3:     node_trace['z'].append(0)
    else:     node_trace['z'].append(GRS.nsAll.z[ID])

    v=[edge_trace1, edge_trace2, node_trace]
    if deflOnly:     v=[edge_trace2, node_trace]

    if mode==1:
        camx=1
        camy=0
        camz=0
    elif mode==2:
        camx=0
        camy=1
        camz=0
    elif mode==3:
        camx=0
        camy=0
        camz=1
    else:
        camx=-5
        camy=-5
        camz=5
    fig = Figure(data=Data(v),
                 layout=Layout(
                     showlegend=False,
                     hovermode='closest',
                     margin=dict(b=20, l=5, r=5, t=40),
                     height=1000,
                     width=2000,
                     scene=dict(
                         camera=dict(eye=dict(x=camx, y=camy, z=camz)),
                         aspectmode='data',
                         xaxis=dict(
                             showgrid=False,
                             zeroline=False,
                             showline=False,
                             ticks='',
                             showticklabels=False
                         ),
                         yaxis=dict(
                             showgrid=False,
                             zeroline=False,
                             showline=False,
                             ticks='',
                             showticklabels=False),
                         zaxis=dict(
                             showgrid=False,
                             zeroline=False,
                             showline=False,
                             ticks='',
                             showticklabels=False),
                     ),
                 ))

    return fig


def PlotShape(GRS, GRS2, ID1, ID2, savepic=False, shift=True, mode=3):
    if savepic:
        wline=5
        pheight = 2000
        pwidth = 4000
        nwidth=5
    else:
        wline=3
        pheight = 500
        pwidth = 1000
        nwidth=2

    edge_trace1 = Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='lines',
        line=dict(color='grey', width=wline),
    )

    edge_trace2 = Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='lines',
        line=dict(color='blue', width=wline),
    )

    if shift: ss=GRS.span/2
    else: ss=0

    for i in range(GRS.nbElAll):
        sID = GRS.lsAll.sID[i]
        eID = GRS.lsAll.eID[i]
        x0 = 0
        y0 = 0
        x1 = 0
        y1 = 0
        z0 = 0
        z1 = 0
        if mode in {0,1,3}:
            y0 = GRS.nsAll.y[sID] - ss
            y1 = GRS.nsAll.y[eID] - ss
        if mode in {0, 2, 3}:
            x0 = GRS.nsAll.x[sID] - ss
            x1 = GRS.nsAll.x[eID] - ss
        if mode in {0, 1, 2}:
            z0 = GRS.nsAll.z[sID]
            z1 = GRS.nsAll.z[eID]
        edge_trace1['x'] += [x0, x1, None]
        edge_trace1['y'] += [y0, y1, None]
        edge_trace1['z'] += [z0, z1, None]

    for i in range(GRS2.nbElAll):
        sID = GRS2.lsAll.sID[i]
        eID = GRS2.lsAll.eID[i]
        x0 = 0
        y0 = 0
        x1 = 0
        y1 = 0
        z0 = 0
        z1 = 0
        if mode in {0, 1, 3}:
            y0 = GRS2.nsAll.y[sID]
            y1 = GRS2.nsAll.y[eID]
        if mode in {0, 2, 3}:
            x0 = GRS2.nsAll.x[sID]
            x1 = GRS2.nsAll.x[eID]
        if mode in {0, 1, 2}:
            z0 = GRS2.nsAll.z[sID]
            z1 = GRS2.nsAll.z[eID]
        edge_trace2['x'] += [x0, x1, None]
        edge_trace2['y'] += [y0, y1, None]
        edge_trace2['z'] += [z0, z1, None]

    node_trace1 = Scatter3d(
        x=[],
        y=[],
        z=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            color=['red'],
            size=10,
        ),
        line=dict(width=nwidth))

    if mode not in {1, 2, 3}:
        node_trace1['x'].append(GRS.nsAll.x[ID1]-ss)
        node_trace1['y'].append(GRS.nsAll.y[ID1]-ss)
        node_trace1['z'].append(GRS.nsAll.z[ID1])

    node_trace2 = Scatter3d(
        x=[],
        y=[],
        z=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            color=['green'],
            size=10,
        ),
        line=dict(width=nwidth))

    if mode not in {1,2,3}:
        node_trace2['x'].append(GRS2.nsAll.x[ID2])
        node_trace2['y'].append(GRS2.nsAll.y[ID2])
        node_trace2['z'].append(GRS2.nsAll.z[ID2])

    if mode==1:
        camx=1
        camy=0
        camz=0
    elif mode==2:
        camx=0
        camy=1
        camz=0
    elif mode==3:
        camx=0
        camy=0
        camz=1
    else:
        camx=-5
        camy=-5
        camz=5

    scene = dict(camera=dict(eye=dict(x=camx, y=camy, z=camz)), aspectmode='data',
                    xaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        ticks='',
                        showticklabels=False
                    ),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        ticks='',
                        showticklabels=False
                    ),
                    zaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        ticks='',
                        showticklabels=False
                    ), )

    fig = Figure(data=Data([edge_trace1, edge_trace2, node_trace1, node_trace2]),
                 layout=Layout(
                     showlegend=False,
                     hovermode='closest',
                     margin=dict(b=20, l=5, r=5, t=40),
                     height=pheight,
                     width=pwidth,
                     scene=scene,
                 ))

    return fig


def PlotNF(GRS, NForce, minid, maxid, scale):

    Nstart = -NForce[:, 0]
    Nend = NForce[:, 6]
    scale = scale * 1 / np.max(np.abs(np.append(Nstart, Nend)))

    edge_trace1 = Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='lines',
        line=dict(color='black'),
    )

    edge_trace2 = Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='lines',
        line=dict(color='grey'),
    )

    edge_trace3 = Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='lines',
        line=dict(color='red'),
    )

    edge_trace4 = Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='lines',
        line=dict(color='blue'),
    )

    for i in range(GRS.nbElAll):
        sID = GRS.lsAll.sID[i]
        eID = GRS.lsAll.eID[i]
        x0 = GRS.nsAll.x[sID]
        y0 = GRS.nsAll.y[sID]
        z0 = GRS.nsAll.z[sID]
        x1 = GRS.nsAll.x[eID]
        y1 = GRS.nsAll.y[eID]
        z1 = GRS.nsAll.z[eID]
        edge_trace1['x'] += [x0, x1, None]
        edge_trace1['y'] += [y0, y1, None]
        edge_trace1['z'] += [z0, z1, None]
        z0 += scale * Nstart[i]
        z1 += scale * Nend[i]
        if i == maxid:
            edge_trace3['x'] += [x0, x1, None]
            edge_trace3['y'] += [y0, y1, None]
            edge_trace3['z'] += [z0, z1, None]
        elif i == minid:
            edge_trace4['x'] += [x0, x1, None]
            edge_trace4['y'] += [y0, y1, None]
            edge_trace4['z'] += [z0, z1, None]
        else:
            edge_trace2['x'] += [x0, x1, None]
            edge_trace2['y'] += [y0, y1, None]
            edge_trace2['z'] += [z0, z1, None]

    for i in range(GRS.nbElAll):
        sID = GRS.lsAll.sID[i]
        eID = GRS.lsAll.eID[i]
        x0 = GRS.nsAll.x[sID]
        y0 = GRS.nsAll.y[sID]
        z0 = GRS.nsAll.z[sID]
        x1 = GRS.nsAll.x[eID]
        y1 = GRS.nsAll.y[eID]
        z1 = GRS.nsAll.z[eID]
        z0i = z0 + scale * Nstart[i]
        z1i = z1 + scale * Nend[i]
        if i == maxid:
            edge_trace3['x'] += [x0, x0, None]
            edge_trace3['y'] += [y0, y0, None]
            edge_trace3['z'] += [z0, z0i, None]
            edge_trace3['x'] += [x1, x1, None]
            edge_trace3['y'] += [y1, y1, None]
            edge_trace3['z'] += [z1, z1i, None]
        elif i == minid:
            edge_trace4['x'] += [x0, x0, None]
            edge_trace4['y'] += [y0, y0, None]
            edge_trace4['z'] += [z0, z0i, None]
            edge_trace4['x'] += [x1, x1, None]
            edge_trace4['y'] += [y1, y1, None]
            edge_trace4['z'] += [z1, z1i, None]
        else:
            edge_trace2['x'] += [x0, x0, None]
            edge_trace2['y'] += [y0, y0, None]
            edge_trace2['z'] += [z0, z0i, None]
            edge_trace2['x'] += [x1, x1, None]
            edge_trace2['y'] += [y1, y1, None]
            edge_trace2['z'] += [z1, z1i, None]

    fig = Figure(data=Data([edge_trace1, edge_trace2, edge_trace3, edge_trace4]),
                 layout=Layout(
                     showlegend=False,
                     hovermode='closest',
                     #  margin=dict(b=20, l=5, r=5, t=40),
                     height=500,
                     width=1000,
                     scene=dict(
                         aspectmode='data',
                         xaxis=dict(
                             showgrid=False,
                             zeroline=False),
                         yaxis=dict(
                             showgrid=False,
                             zeroline=False),
                         zaxis=dict(
                             showgrid=False,
                             zeroline=False), ),
                 ))

    return fig


def PlotMom(GRS, MForce, scale):

    Mstart = -MForce[:, 4]
    Mend = MForce[:, 10]
    scale = scale * 1 / np.max(np.abs(np.append(Mstart, Mend)))

    edge_trace1 = Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='lines',
        line=dict(color='grey'),
    )

    edge_trace2 = Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='lines',
        line=dict(color='blue'),
    )

    for i in range(GRS.nbElAll):
        sID = GRS.lsAll.sID[i]
        eID = GRS.lsAll.eID[i]
        x0 = GRS.nsAll.x[sID]
        y0 = GRS.nsAll.y[sID]
        z0 = GRS.nsAll.z[sID]
        x1 = GRS.nsAll.x[eID]
        y1 = GRS.nsAll.y[eID]
        z1 = GRS.nsAll.z[eID]
        edge_trace1['x'] += [x0, x1, None]
        edge_trace1['y'] += [y0, y1, None]
        edge_trace1['z'] += [z0, z1, None]
        z0 += scale * Mstart[i]
        z1 += scale * Mend[i]
        edge_trace2['x'] += [x0, x1, None]
        edge_trace2['y'] += [y0, y1, None]
        edge_trace2['z'] += [z0, z1, None]

    for i in range(GRS.nbElAll):
        sID = GRS.lsAll.sID[i]
        eID = GRS.lsAll.eID[i]
        x0 = GRS.nsAll.x[sID]
        y0 = GRS.nsAll.y[sID]
        z0 = GRS.nsAll.z[sID]
        x1 = GRS.nsAll.x[eID]
        y1 = GRS.nsAll.y[eID]
        z1 = GRS.nsAll.z[eID]
        z0i = z0 + scale * Mstart[i]
        z1i = z1 + scale * Mend[i]
        edge_trace1['x'] += [x0, x0, None]
        edge_trace1['y'] += [y0, y0, None]
        edge_trace1['z'] += [z0, z0i, None]
        edge_trace2['x'] += [x1, x1, None]
        edge_trace2['y'] += [y1, y1, None]
        edge_trace2['z'] += [z1, z1i, None]

    fig = Figure(data=Data([edge_trace1, edge_trace2]),
                 layout=Layout(
                     showlegend=False,
                     hovermode='closest',
                     margin=dict(b=20, l=5, r=5, t=40),
                     height=1000,
                     width=2000,
                     scene=dict(
                         aspectmode='data',
                         xaxis=dict(
                             showgrid=False,
                             zeroline=False,
                             showline=False,
                             ticks='',
                             showticklabels=False
                         ),
                         yaxis=dict(
                             showgrid=False,
                             zeroline=False,
                             showline=False,
                             ticks='',
                             showticklabels=False
                         ),
                         zaxis=dict(
                             showgrid=False,
                             zeroline=False,
                             showline=False,
                             ticks='',
                             showticklabels=False
                         ), ),
                 ))

    return fig


def PlotImp(GRS,GRSimp, scale, savepic=False):

    if savepic:
        wline=5
        pheight = 2000
        pwidth = 4000
    else:
        wline=3
        pheight = 500
        pwidth = 1000

    edge_trace = Scatter3d(
        x=[],
        y=[],
        z=[],
        # hoverinfo='none',
        line=dict(color='grey',width=wline),
        mode='lines')

    edge_trace2 = Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='lines',
        line=dict(color='blue', width=wline),
    )

    for i in range(GRS.nbElAll):
        sID = GRS.lsAll.sID[i]
        eID = GRS.lsAll.eID[i]
        x0 = GRS.nsAll.x[sID]
        y0 = GRS.nsAll.y[sID]
        z0 = GRS.nsAll.z[sID]
        x1 = GRS.nsAll.x[eID]
        y1 = GRS.nsAll.y[eID]
        z1 = GRS.nsAll.z[eID]
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]
        edge_trace['z'] += [z0, z1, None]
        sID = GRSimp.lsAll.sID[i]
        eID = GRSimp.lsAll.eID[i]
        x0 = GRS.nsAll.x[sID] + (GRSimp.nsAll.x[sID]-GRS.nsAll.x[sID])*scale
        y0 = GRS.nsAll.y[sID] + (GRSimp.nsAll.y[sID]-GRS.nsAll.y[sID])*scale
        z0 = GRS.nsAll.z[sID] + (GRSimp.nsAll.z[sID]-GRS.nsAll.z[sID])*scale
        x1 = GRS.nsAll.x[eID] + (GRSimp.nsAll.x[eID]-GRS.nsAll.x[eID])*scale
        y1 = GRS.nsAll.y[eID] + (GRSimp.nsAll.y[eID]-GRS.nsAll.y[eID])*scale
        z1 = GRS.nsAll.z[eID] + (GRSimp.nsAll.z[eID]-GRS.nsAll.z[eID])*scale
        edge_trace2['x'] += [x0, x1, None]
        edge_trace2['y'] += [y0, y1, None]
        edge_trace2['z'] += [z0, z1, None]

    fig = Figure(data=Data([edge_trace,edge_trace2]),
                 layout=Layout(
                     showlegend=False,
                     hovermode='closest',
                     margin=dict(b=20, l=5, r=5, t=40),
                     height=pheight,
                     width=pwidth,
                     scene=dict(
                         aspectmode='data',
                         xaxis=dict(
                             showgrid=False,
                             zeroline=False,
                             showline=False,
                             ticks='',
                             showticklabels=False),
                         yaxis=dict(
                             showgrid=False,
                             zeroline=False,
                             showline=False,
                             ticks='',
                             showticklabels=False),
                         zaxis=dict(
                             showgrid=False,
                             zeroline=False,
                             showline=False,
                             ticks='',
                             showticklabels=False), ),
                 ))

    return fig


def plot_res(GRSA, rhoAA, CapacityFA, ref_rho1, ref_f1, Data, nb):
    traces = []
    LperHAord = np.zeros(6)
    for i in range(6):
        LperHAord[i] = GRSA[i].span / GRSA[i].height
    colors = ['rgb(205, 12, 24)', 'rgb(34,139,34)', 'rgb(240,230,140)', 'rgb(22, 96, 167)', 'rgb(255,0,255)',
              'rgb(0,206,209)']
    c = np.zeros(6)
    nb1=nb
    nb2=nb+1
    if nb==6:
        nb1=0
        nb2=6
    for i in range(6):
        c[i] = (LperHAord[i] - 12) ** 2 / 51 + 0.91
    for i in range(len(GRSA)%6):
        for j in range(nb1,nb2):
            k = i * 6 + j
            D = Data[k, 3]
            T = GRSA[k].secT
            i2min = 32 * (D * 2 + D * T + 2 * T ** 2)
            D = Data[k, 3]+Data[k, 3]*3
            i2max = 32 * (D * 2 + D * T + 2 * T ** 2)
            minload = CapacityFA[k,0] / GRSA[k].span / GRSA[k].span * GRSA[k].nbnBns
            maxload = CapacityFA[k,3] / GRSA[k].span / GRSA[k].span * GRSA[k].nbnBns
            traces.append(Scatter(
                x=rhoAA[k] * 1e6,
                y=CapacityFA[k] / GRSA[k].secT / 1000 / GRSA[k].Lav / c[j],
                text=['#{:d} L={:d}m g={:d} L/H={:.0f} Lav={:.1f}m L/Lav={:.1f} i={:.0f}-{:.0f}mm p={:.0f}-{:.0f}kN/m2 {}'.format(k, GRSA[k].span, GRSA[k].gN,
                                                                                                 GRSA[k].span / GRSA[k].height,
                                                                                                 GRSA[k].Lav,
                                                                                                 GRSA[k].span / GRSA[k].Lav,
                                                                                                 i2min, i2max,
                                                                                                 minload,maxload, ijk) for ijk in rhoAA[k]],
                hoverinfo='text',
                name=k,
                showlegend = k<6,
                legendgroup = j,
                line=dict(color=colors[j]),
            ))

    traces.append(Scatter(
        x=ref_rho1 * 1e6,
        y=ref_f1,
        text='ref',
        hoverinfo='text',
        line=dict(color='rgb(0,0,0)', dash='dash'),
    ))

    fig = Figure(data=traces,
                 layout=Layout(
                     showlegend=True,
                     hovermode='closest',
                     height=600,
                     width=800, ),
                 )
    return fig


def plot_res_sep(GRSA, rhoAA, CapacityFA, ref_rho1, ref_f1):

    traces = []
    LperHAord = np.zeros(6)
    for i in range(6):
        LperHAord[i] = GRSA[i].span / GRSA[i].height
    colors = ['rgb(205, 12, 24)', 'rgb(34,139,34)', 'rgb(240,230,140)', 'rgb(22, 96, 167)', 'rgb(255,0,255)',
              'rgb(0,206,209)']
    c = np.zeros(6)
    for i in range(6):
        c[i] = (LperHAord[i] - 12) ** 2 / 51 + 0.91

    fig = tools.make_subplots(rows=3, cols=2)

    for i in range(5):
        for j in range(6):
            k = i * 6 + j
            traces.append(Scatter(
                x=rhoAA[k] * 1e6,
                y=CapacityFA[k] / GRSA[k].secT / 1000 / GRSA[k].Lav / c[j],
                text='#{:d} L={:d}m g={:d} L/H={:.0f} Lav={:.1f}m L/Lav={:.1f}'.format(k, GRSA[k].span, GRSA[k].gN,
                                                                                   GRSA[k].span / GRSA[k].height,
                                                                                   GRSA[k].Lav,
                                                                                   GRSA[k].span / GRSA[k].Lav),
                hoverinfo='text',
                line=dict(color=colors[j]),
                xaxis=dict(
                    range=[0, 15]
                ),
                yaxis=dict(
                    range=[0, 8]
                ),
            ))
            fig.append_trace(traces[k], 1+int(j/2), 1+j%2)

    traces.append(Scatter(
        x=ref_rho1 * 1e6,
        y=ref_f1,
        text='ref',
        hoverinfo='text',
        line=dict(color='rgb(0,0,0)', dash='dash'),
    ))
    fig.append_trace(traces[30], 1, 1)
    fig.append_trace(traces[30], 1, 2)
    fig.append_trace(traces[30], 2, 1)
    fig.append_trace(traces[30], 2, 2)
    fig.append_trace(traces[30], 3, 1)
    fig.append_trace(traces[30], 3, 2)

    fig['layout'].update(
                 showlegend=False,
                 hovermode='closest',
                 height=800,
                 width=1000,
             )

    return fig