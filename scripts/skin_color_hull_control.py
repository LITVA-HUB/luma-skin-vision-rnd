"""Same empirical color-support control; projection does not use image truth."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize,nnls


def make_hull(palette):
    center=palette.mean(0);scale=np.maximum(palette.std(0),1e-6)
    equations=ConvexHull((palette-center)/scale).equations
    return {'center':center,'scale':scale,'a':equations[:,:3],'b':equations[:,3]}


def project_hull(prediction,hull):
    a,b=hull['a'],hull['b'];out=prediction.copy();feasibility=stationarity=0.;changed=0
    for i,p in enumerate(prediction):
        z0=(p-hull['center'])/hull['scale']
        if np.max(a@z0+b)<=1e-10:continue
        result=minimize(lambda z:.5*np.sum((z-z0)**2),np.zeros(3),jac=lambda z:z-z0,
            constraints={'type':'ineq','fun':lambda z:-(a@z+b),'jac':lambda z:-a},
            method='SLSQP',options={'ftol':1e-12,'maxiter':300})
        z=result.x;slack=a@z+b;violation=max(0.,float(slack.max()))
        active=slack>=-1e-7
        if not active.any():raise ValueError('Outside input without active projection face')
        multipliers,residual=nnls(a[active].T,z0-z,maxiter=1000)
        relative=float(residual/max(1.,np.linalg.norm(z0-z)))
        if violation>1e-7 or relative>1e-7:raise ValueError(f'Projection KKT failed: {result.message}, {violation}, {relative}')
        feasibility=max(feasibility,violation);stationarity=max(stationarity,relative)
        out[i]=z*hull['scale']+hull['center'];changed+=1
    return out,{'projected_images':changed,'max_feasibility_violation':feasibility,'max_relative_stationarity':stationarity}
