# -*- coding: utf-8 -*-
#*************************************************************************
#***File Name: constants.py
#***Author: Zhonghai Zhao
#***Mail: zhaozhonghi@126.com 
#***Created Time: 2018年03月25日 星期日 14时39分05秒
#*************************************************************************
class bubble_mr(object):
    "define come useful physical constants."
    import math
# light speed
    c = 3e8
# alfven speed
    va = c/100.0
# particle number density
    n0 = 0.2187e26
# boltzmann constant
    kb = 1.38e-23
# electron mass
    me = 9.11e-31
# charge
    qe = 1.602e-19
    qi = 1*qe
# ion mass
    ratio = 100
    mi = ratio*me
# permittivity and permeability
    epsilon0 = 8.854e-12
    mu0 = 4*3.14*1e-7
# plasma frequency
    wpe = math.sqrt(n0*qe*qe/epsilon0/me)
    wpi = math.sqrt(n0*qi*qi/epsilon0/mi)
# magnetic field
    B0 = va*math.sqrt(mu0*mi*n0)
# temperature and velocity
    e = 0.025*me*c*c
    Te = e/kb
    Ti = Te
    ve = math.sqrt(2*e/me)
    vi = math.sqrt(2*e/mi)
#acoustic veloccity
    gamma = 5.0/3.0
    cs = math.sqrt(gamma*e/mi)
# cyclotron frequency
    ome = B0*qe/me
    omi = B0*qi/mi
# skin depth
    de = c/wpe
    di = c/wpi
# characteristic length
    re = me*ve/qe/B0
    ri = mi*vi/qi/B0
# characteristic time
    tec = 1/ome
    tic = 1/omi
    tep = 1/wpe
    tip = 1/wpi 
# normalization
    E0 = va*B0
    J0 = n0*qe*va
    T0 = Te
    V0 = va
    N0 = n0
    t0 = tic
    P0 = n0*e
    F0 = B0*di
    D0 = di
    Pe0 = me*va
class laser_mr(object):
    "define some constants in laser plasma reconnection."
    #constantis
    import math
    me = 9.11e-31
    mi = 1836.0*me
    c = 3e8
    qe = 1.602e-19
    pi = 3.1415926
    epsilon0 = 8.854e-12
    mu0 = 4*pi*1e-7
    #laser
    a = 1
    la = 1e-6
    omega = 2*pi*c/la
    T0 = la/c
    E0 = a*me*c*omega/qe
    #plasma
    nc = me*omega*omega*epsilon0/qe/qe
    n0 = 20*nc
    B0 = E0/c
    J0 = nc*qe*c
    P0 = me*c
    T0 = 1.16045e7
    Pa0 = 1e19
    sigma0 = 1.0/float(2*mu0*me*c*c)
    je0 = E0*J0
    en0 = me*c*c
    #calculation
    te = 1e3*qe
    de = math.sqrt(epsilon0*te/n0/qe/qe)
    col = math.log(4*3.1415926*n0*de*de*de)
    mfp = 6*3.1415926*math.sqrt(6*3.1415926)*(epsilon0*epsilon0*te*te/n0/qe/qe/qe/qe/col)
    wpe = math.sqrt(n0*qe*qe/epsilon0/me)
    wpi = math.sqrt(n0*qe*qe/epsilon0/mi)
class proton_radiography:
    '''
    define some constants in proton radiography case.
    '''
    import numpy as np
    #constants
    me = 9.11e-31
    mi = 100.0*me
    c = 3e8
    qe = 1.602e-19
    pi = 3.1415926
    epsilon0 = 8.854e-12
    mu0 = 4*pi*1e-7
    n0 = 0.2187e26
    #length
    wpi = np.sqrt(n0*qe*qe/epsilon0/mi)
    di = c/wpi
    #dimensionless
    B0 = 150
    E0 = c * B0
    J0 = qe * n0 * c
    T0 = me * c * c
    ek = 2.0e6 * qe
    v0 = np.sqrt(2*ek/mi)
    r0 = mi*v0/qe/B0
class proton_benchmark:
    '''
    define some constants in proton benchmark case.
    '''
    import numpy as np
    #constants
    me = 9.11e-31
    mass_ratio = 100
    mi = mass_ratio*me
    c = 3e8
    qe = 1.602e-19
    pi = 3.1415926
    epsilon0 = 8.854e-12
    mu0 = 4*pi*1e-7
    n0 = 0.2187e26
    #length
    wpi = np.sqrt(n0*qe*qe/epsilon0/mi)
    di = 1e-5
    a = 100e-6
    b = 300e-6
    source = 1e-2
    W = 14.7
    #dimensionless
    B0 = 0.1903*(a/b)*np.sqrt(W*mass_ratio)/(source*100)
    E0 = c * B0
    J0 = qe * n0 * c
    T0 = me * c * c
    ek = 2.0e6 * qe
    v0 = np.sqrt(2*ek/mi)
    r0 = mi*v0/qe/B0
