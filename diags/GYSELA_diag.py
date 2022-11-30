import getopt
import glob
import matplotlib.pyplot as mpp
import numpy as np
import sys

import imports.HDF5utils as H5ut
from imports.diag_utils import fourier_diag_to_tensor

# from compression_datagen import GYSELA_diag

# -----------------------------------------------
# Read the arguments given in the command line
# -----------------------------------------------
def read_args(argv):
    inputDir = ""

    try:
        opts, args = getopt.getopt(argv, "hi:", ["idir="])
    except getopt.GetoptError:
        print("python GYSELA_waveletdiag.py -i <inputDir>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("python GYSELA_waveletdiag.py -i <inputDir>")
            sys.exit()
        elif opt in ("-i", "--idir"):
            inputDir = arg

    return inputDir


# --------------------------------------------------
# Compute rms
# --------------------------------------------------
def rms(W):

    rmsW = np.sqrt(np.mean((W - np.mean(W)) * (W - np.mean(W))))

    return rmsW


# -------------------------------------------------
# Personal FFT2D function
# -------------------------------------------------
def Fourier2D(F0, y0, x0):

    """ Personal FFT2D function"""

    nx0 = len(x0)
    nx = 2 * int(nx0 / 2)
    hnx = int(nx / 2)
    ny0 = len(y0)
    ny = 2 * int(ny0 / 2)
    hny = int(ny / 2)

    x = x0[0:nx]
    y = y0[0:ny]
    F = F0[0:ny, 0:nx]

    Lx = x[nx - 1] - x[0]
    dx = x[1] - x[0]
    dkx = 2.0 * np.pi / (Lx + dx)
    kx = np.zeros(nx)
    temp = -dkx * np.r_[1 : hnx + 1]
    kx[0:hnx] = temp[::-1]
    kx[hnx:nx] = dkx * np.r_[0:hnx]

    Ly = y[ny - 1] - y[0]
    dy = y[1] - y[0]
    dky = 2.0 * np.pi / (Ly + dy)
    ky = np.zeros(ny)
    temp = -dky * np.r_[1 : hny + 1]
    ky[0:hny] = temp[::-1]
    ky[hny:ny] = dky * np.r_[0:hny]

    TFF = np.zeros((ny, nx), dtype=complex)
    AA = np.zeros((ny, nx), dtype=complex)
    var = np.conjugate(np.fft.fft2(np.conjugate(F))) / float((nx * ny))

    AA[:, 0:hnx] = var[:, hnx:nx]
    AA[:, hnx:nx] = var[:, 0:hnx]
    TFF[0:hny, :] = AA[hny:ny, :]
    TFF[hny:ny, :] = AA[0:hny, :]

    return TFF, kx, ky


# -------------------------------------------------
# First Derivative
#   Input: F        = function to be derivate
#          dx       = step of the variable
#                      for derivative
#          periodic = 1 if F is periodic
#                   = 0 otherwise (by default)
#   Output dFdx = first derivative of F
# -------------------------------------------------
def Derivee1(F, dx, periodic=False, axis=0):

    """
    First Derivative
       Input: F        = function to be derivate
              dx       = step of the variable for derivative
              periodic = 1 if F is periodic
       Output: dFdx = first derivative of F
    """
    if axis != 0:
        F = np.swapaxes(F, axis, 0)
        F = Derivee1(F, dx, periodic=periodic, axis=0)
        F = np.swapaxes(F, axis, 0)
        return F

    nx = np.size(F, 0)
    dFdx = np.empty_like(F)

    c0 = 2.0 / 3.0
    dFdx[2:-2] = c0 / dx * (F[3:-1] - F[1:-3] - (F[4:] - F[:-4]) / 8)

    c1 = 4.0 / 3.0
    c2 = 25.0 / 12.0
    c3 = 5.0 / 6.0
    if not periodic:
        dFdx[0] = (-F[4] / 4.0 + F[3] * c1 - F[2] * 3.0 + F[1] * 4.0 - F[0] * c2) / dx
        dFdx[-1] = (
            +F[-5] / 4.0 - F[-4] * c1 + F[-3] * 3.0 - F[-2] * 4.0 + F[-1] * c2
        ) / dx
        dFdx[1] = (+F[4] / 12.0 - F[3] / 2.0 + F[2] / c0 - F[1] * c3 - F[0] / 4.0) / dx
        dFdx[-2] = (
            -F[-5] / 12.0 + F[-4] / 2.0 - F[-3] / c0 + F[-2] * c3 + F[-1] / 4.0
        ) / dx
    else:
        # Here, we need to take care of the ghost point on the edge
        dFdx[1] = c0 / dx * (F[2] - F[-1] - (F[3] - F[-2]) / 8.0)
        dFdx[0] = c0 / dx * (F[1] - F[-2] - (F[2] - F[-3]) / 8)
        dFdx[-1] = dFdx[0]
        dFdx[-2] = c0 / dx * (F[0] - F[-3] - (F[1] - F[-4]) / 8.0)

    return dFdx


# --------------------------------------------------
# search index position
# --------------------------------------------------
def search_pos(x_value, x_array):
    """ Search index position"""

    x_indx = int(np.searchsorted(x_array, x_value, side="left"))

    return x_indx


# -----------------------------------------------------
# Define graphic axis
# -----------------------------------------------------
def setup_polsection_plot(ax, xx, yy, i_buffL=None, i_buffR=None):
    """
    Parameters
    ----------
    ax: object *axes*
    species: object *GYSspecies*

    Returns
    -------
    None
    """
    ax.axis("equal")
    ax.axis("off")
    ax.axis((np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)))
    ax.plot(xx[:, 0], yy[:, 0], "k")
    ax.plot(xx[:, -1], yy[:, -1], "k")
    if (i_buffL is not None) and (i_buffR is not None):
        ax.plot(xx[:, i_buffL], yy[:, i_buffL], "k--")
        ax.plot(xx[:, i_buffR], yy[:, i_buffR], "k--")
    return None


# -----------------------------------------------
# Plot a Phi2D cross-section
# -----------------------------------------------
def PlotPhi2D(H5conf, H5rprofP, H5Phi2D, outputDir):

    Nr = H5conf.Nr
    Ntheta = H5conf.Ntheta
    rg = H5conf.rg
    thetag = H5conf.thetag
    xx = H5conf.R - H5conf.R0
    yy = np.copy(H5conf.Z)

    Ntime = len(H5Phi2D.time_diag)
    itimeDiag = Ntime - 1
    timePhi2D = H5Phi2D.time_diag[itimeDiag][0]
    Phi2D_rtheta = H5Phi2D.Phirth[itimeDiag, :, :]
    Phi00 = H5rprofP.Phi00[itimeDiag, :]
    Phi00_rtheta = np.dot(np.ones((Ntheta + 1, 1)), Phi00.reshape(1, Nr + 1))
    deltaPhi_rtheta = Phi2D_rtheta - Phi00
    rmsDeltaPhi = rms(deltaPhi_rtheta)

    mpp.ioff()
    fig = mpp.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    mpp.suptitle(
        r"$\Phi-\Phi_{{00}}$ at time = ${}/\omega_c$".format(timePhi2D), size=18
    )
    p1 = ax1.pcolormesh(
        xx,
        yy,
        deltaPhi_rtheta,
        vmin=-3 * rmsDeltaPhi,
        vmax=3 * rmsDeltaPhi,
        shading="gouraud",
        cmap="bwr",
    )
    setup_polsection_plot(ax1, xx, yy)

    if "LIMITER_shape" in H5conf.keys:
        ll = np.ma.masked_array(H5conf.LIMITER_shape, H5conf.LIMITER_shape < 0.5)
        l_shp = ax1.pcolormesh(xx, yy, ll, shading="gouraud", cmap="binary")

    figName = outputDir + "/Phi2Drtheta_t" + str(round(timePhi2D)) + ".png"
    print("Save figure:", figName)
    mpp.savefig(figName)
    mpp.ion()


def GetPhi2D_thphi_timeseries(H5Phi2D):
    return H5Phi2D.Phithphi


def GetPhi2Dmostunstable(H5conf, H5Phi2D):
    """
    We extract the unstable fourier modes time series 
    (without necessity of plotting) 
    """

    Nr = H5conf.Nr
    Ntheta = H5conf.Ntheta
    Nphi = H5conf.Nphi
    rg = H5conf.rg
    thetag = H5conf.thetag
    phig = H5conf.phig

    timeg = H5Phi2D.time_diag
    Ntime = len(timeg)
    itimeEnd = Ntime - 1
    timeEnd = timeg[itimeEnd][0]

    Phi2D_thphi = H5Phi2D.Phithphi

    # --> For modes m=0 to m_max with n=0
    itimeLook = itimeEnd
    [TFPhi_mn, m2d, n2d] = Fourier2D(
        Phi2D_thphi[itimeLook, 0:Nphi, 0:Ntheta], phig[0:Nphi], thetag[0:Ntheta]
    )
    m_max = 4
    ktheta0 = int(Ntheta // 2)
    kphi0 = int(Nphi // 2)
    dic_kthetam = {}
    dic_strm0 = {}
    str_n0 = str(int(n2d[kphi0]))
    for im in np.arange(0, m_max + 1):
        kthetam = ktheta0 + im
        dic_kthetam[im] = kthetam
        dic_strm0[im] = "(m,n) = {m},0".format(m=int(m2d[kthetam]))
    # end for
    Abs_modes_m0 = np.zeros((m_max + 1, itimeEnd + 1))
    for it in np.r_[0 : itimeEnd + 1]:
        [TFPhi_mn, m2d, n2d] = Fourier2D(
            Phi2D_thphi[it, 0:Nphi, 0:Ntheta], phig[0:Nphi], thetag[0:Ntheta]
        )
        for im in np.arange(0, m_max + 1):
            Abs_modes_m0[im, it] = np.abs(TFPhi_mn[kphi0, dic_kthetam[im]])
        # end for
    # end for

    # --> For the nb_mn_unstable most unstable mode for n different of 0
    [TFPhi_mn, m2d, n2d] = Fourier2D(
        Phi2D_thphi[itimeLook, 0:Nphi, 0:Ntheta], phig[0:Nphi], thetag[0:Ntheta]
    )
    # --> The search of maximum is only done on half the spectrum
    # -->  (due to symmetry)
    # --> iphi_min excludes modes such that: |n| < iphi_min
    iphi_min = 1
    Abs_TFPhi = np.abs(TFPhi_mn[0 : Nphi // 2 + 1 - iphi_min, :])
    max_abs_TFPhi = np.sort(np.amax(Abs_TFPhi, axis=0))

    nb_mn_unstable = 7
    dic_ktheta_unstable = {}
    dic_kphi_unstable = {}
    dic_str_mn_unstable = {}
    nb_max_found = 0
    for imax in np.arange(0, nb_mn_unstable + 1):
        max_imax = max_abs_TFPhi[-1 - nb_max_found]
        k_imax = np.nonzero(Abs_TFPhi == max_imax)
        max_found = len(k_imax[0])
        nb_max_found = nb_max_found + max_found
        ktheta_imax = k_imax[1][-1]
        kphi_imax = k_imax[0][-1]
        dic_ktheta_unstable[imax] = ktheta_imax
        dic_kphi_unstable[imax] = kphi_imax
        dic_str_mn_unstable[imax] = "(m,n) = {m},{n}".format(
            m=int(m2d[ktheta_imax]), n=int(n2d[kphi_imax])
        )
    # end for
    Abs_modes_mn_unstable = np.zeros((nb_mn_unstable + 1, itimeEnd + 1))
    for it in np.r_[0 : itimeEnd + 1]:
        [TFPhi_mn, m2d, n2d] = Fourier2D(
            Phi2D_thphi[it, 0:Nphi, 0:Ntheta], phig[0:Nphi], thetag[0:Ntheta]
        )
        for imax in np.arange(0, nb_mn_unstable + 1):
            Abs_modes_mn_unstable[imax, it] = np.abs(
                TFPhi_mn[dic_kphi_unstable[imax], dic_ktheta_unstable[imax]]
            )
        # end for
    # end for
    return Abs_modes_m0, Abs_modes_mn_unstable


# ------------------------------------------------------
# Plot the time evolution of the
#  eight most unstable modes + (0,0) + (1,0)
# ------------------------------------------------------
def PlotPhi2Dmostunstable(H5conf, H5Phi2D, outputDir):
    """
    Plot the time evolution of the eight most 
    unstable modes + (0,0) + (1,0)
    """

    Nr = H5conf.Nr
    Ntheta = H5conf.Ntheta
    Nphi = H5conf.Nphi
    rg = H5conf.rg
    thetag = H5conf.thetag
    phig = H5conf.phig

    timeg = H5Phi2D.time_diag
    Ntime = len(timeg)
    itimeEnd = Ntime - 1
    timeEnd = timeg[itimeEnd][0]

    Phi2D_thphi = H5Phi2D.Phithphi

    # --> For modes m=0 to m_max with n=0
    itimeLook = itimeEnd
    [TFPhi_mn, m2d, n2d] = Fourier2D(
        Phi2D_thphi[itimeLook, 0:Nphi, 0:Ntheta], phig[0:Nphi], thetag[0:Ntheta]
    )
    m_max = 4
    ktheta0 = int(Ntheta // 2)
    kphi0 = int(Nphi // 2)
    dic_kthetam = {}
    dic_strm0 = {}
    str_n0 = str(int(n2d[kphi0]))
    for im in np.arange(0, m_max + 1):
        kthetam = ktheta0 + im
        dic_kthetam[im] = kthetam
        dic_strm0[im] = "(m,n) = {m},0".format(m=int(m2d[kthetam]))
    # end for
    Abs_modes_m0 = np.zeros((m_max + 1, itimeEnd + 1))
    for it in np.r_[0 : itimeEnd + 1]:
        [TFPhi_mn, m2d, n2d] = Fourier2D(
            Phi2D_thphi[it, 0:Nphi, 0:Ntheta], phig[0:Nphi], thetag[0:Ntheta]
        )
        for im in np.arange(0, m_max + 1):
            Abs_modes_m0[im, it] = np.abs(TFPhi_mn[kphi0, dic_kthetam[im]])
        # end for
    # end for

    # --> For the nb_mn_unstable most unstable mode for n different of 0
    [TFPhi_mn, m2d, n2d] = Fourier2D(
        Phi2D_thphi[itimeLook, 0:Nphi, 0:Ntheta], phig[0:Nphi], thetag[0:Ntheta]
    )
    # --> The search of maximum is only done on half the spectrum
    # -->  (due to symmetry)
    # --> iphi_min excludes modes such that: |n| < iphi_min
    iphi_min = 1
    Abs_TFPhi = np.abs(TFPhi_mn[0 : Nphi // 2 + 1 - iphi_min, :])
    max_abs_TFPhi = np.sort(np.amax(Abs_TFPhi, axis=0))

    nb_mn_unstable = 7
    dic_ktheta_unstable = {}
    dic_kphi_unstable = {}
    dic_str_mn_unstable = {}
    nb_max_found = 0
    for imax in np.arange(0, nb_mn_unstable + 1):
        max_imax = max_abs_TFPhi[-1 - nb_max_found]
        k_imax = np.nonzero(Abs_TFPhi == max_imax)
        max_found = len(k_imax[0])
        nb_max_found = nb_max_found + max_found
        ktheta_imax = k_imax[1][-1]
        kphi_imax = k_imax[0][-1]
        dic_ktheta_unstable[imax] = ktheta_imax
        dic_kphi_unstable[imax] = kphi_imax
        dic_str_mn_unstable[imax] = "(m,n) = {m},{n}".format(
            m=int(m2d[ktheta_imax]), n=int(n2d[kphi_imax])
        )
    # end for
    Abs_modes_mn_unstable = np.zeros((nb_mn_unstable + 1, itimeEnd + 1))
    for it in np.r_[0 : itimeEnd + 1]:
        [TFPhi_mn, m2d, n2d] = Fourier2D(
            Phi2D_thphi[it, 0:Nphi, 0:Ntheta], phig[0:Nphi], thetag[0:Ntheta]
        )
        for imax in np.arange(0, nb_mn_unstable + 1):
            Abs_modes_mn_unstable[imax, it] = np.abs(
                TFPhi_mn[dic_kphi_unstable[imax], dic_ktheta_unstable[imax]]
            )
        # end for
    # end for

    mpp.ioff()
    fig = mpp.figure(figsize=(8, 8))
    color_str = ["k", "k--", "r", "b", "g", "m", "k", "y", "c", "r--", "b--"]
    # --> Plot (m,n)=(0,0)
    if np.min(Abs_modes_m0[0, :]) > 0.0:
        mpp.semilogy(
            timeg[0 : itimeEnd + 1],
            Abs_modes_m0[0, :],
            color_str[0],
            label=dic_strm0[0],
        )
    # --> Plot (m,n)=(1,0)
    if np.min(Abs_modes_m0[1, :]) > 0.0:
        mpp.semilogy(
            timeg[0 : itimeEnd + 1],
            Abs_modes_m0[1, :],
            color_str[1],
            label=dic_strm0[1],
        )
    # --> Plot the nb_mn_unstable most unstable (m,n) modes with n different from 0
    for imax in np.arange(0, nb_mn_unstable + 1):
        str_legend_mn = dic_str_mn_unstable[imax]
        mpp.semilogy(
            timeg[0 : itimeEnd + 1],
            Abs_modes_mn_unstable[imax, :],
            color_str[2 + imax],
            label=str_legend_mn,
        )
    # end for
    mpp.xlabel("time")
    mpp.ylabel(r"$abs(\Phi_{mn})(r)$")
    mpp.legend(loc=4)
    figName = outputDir + "/Fig_mn_mostunstable_t" + str(timeEnd) + ".png"
    print("Save figure:", figName)
    mpp.savefig(figName, transparent=False, format="png")
    mpp.ion()


# ------------------------------------------------------
# Compute the time evolution of the local charge
#  density conservation
# ------------------------------------------------------
def Compute_local_charge_density(H5conf, H5rprofGC):

    """ Local charge density computation """

    rg = H5conf.rg
    dr = rg[1] - rg[0]
    iota = 1.0 / H5conf.safety_factor

    timeg = H5rprofGC.time_diag
    dt = timeg[1] - timeg[0]

    # --> Computation of d<nGC>/dt
    dnGC_dt = Derivee1(H5rprofGC.densGC_FSavg, axis=0, dx=dt)

    intdthetadphi_Js = H5conf.intdthetadphi_Js
    # --> Computation of dGamma/dpsi
    r_on_q = rg * iota
    dGamma_vE_dpsi = Derivee1(
        r_on_q * H5rprofGC.GammaGC_vE_r_FSavg * intdthetadphi_Js, axis=1, dx=dr
    ) / (r_on_q * intdthetadphi_Js)
    dGamma_vD_dpsi = Derivee1(
        r_on_q * H5rprofGC.GammaGC_vD_r_FSavg * intdthetadphi_Js, axis=1, dx=dr
    ) / (r_on_q * intdthetadphi_Js)
    dGamma_vE_dr = Derivee1(
        H5rprofGC.GammaGC_vE_r_FSavg * intdthetadphi_Js, axis=1, dx=dr
    ) / (intdthetadphi_Js)
    dGamma_vD_dr = Derivee1(
        H5rprofGC.GammaGC_vD_r_FSavg * intdthetadphi_Js, axis=1, dx=dr
    ) / (intdthetadphi_Js)

    # --> Computation of divergence of radial diffusion flux
    #    -1/r d/dr [ r D(r) d/dr (<nGC>-<nGC(t=0)>) ]
    dGamma_diffus_dr = (
        -1.0
        / rg
        * Derivee1(
            rg
            * H5conf.coefDr
            * Derivee1(
                H5rprofGC.densGC_FSavg - H5rprofGC.densGC_FSavg[0], axis=1, dx=dr
            ),
            axis=1,
            dx=dr,
        )
    )

    return [
        dnGC_dt,
        dGamma_vE_dpsi,
        dGamma_vD_dpsi,
        dGamma_vE_dr,
        dGamma_vD_dr,
        dGamma_diffus_dr,
    ]


# ------------------------------------------------------
# Local charge density conservation at a specific time
# ------------------------------------------------------
def PlotConservDensity(H5conf, H5rprofGC, speciesName, outputDir):
    """
    Local charge density conservation
    """

    # --> Compute the different terms of the local charge
    # -->  density for each species
    [
        dnGC_dt,  # A compresser -> pas vraiment maintenant, c'est en 1D. On attend Virginie
        dGamma_vE_dpsi,
        dGamma_vD_dpsi,
        dGamma_vE_dr,
        dGamma_vD_dr,
        dGamma_diffus_dr,
    ] = Compute_local_charge_density(H5conf, H5rprofGC)

    # --> Definition of rhomin and rhomax
    Nr = H5conf.Nr
    rhog = H5conf.rhostar * H5conf.rg
    imin_rho = 0
    imax_rho = Nr
    irho_range = np.r_[imin_rho : imax_rho + 1]

    # --> Ask for initial and end time
    timeg = H5rprofGC.time_diag
    itime_init = 0
    itime_end = len(timeg) - 1
    timeEnd = timeg[itime_end][0]

    it0 = itime_init
    it1 = itime_end + 1

    # *** Figure 1 ***
    fig = mpp.figure(figsize=(8, 8))
    mpp.ioff()
    mpp.suptitle(" for {}".format(speciesName))

    Sce2D_dens = np.zeros((1, Nr + 1))
    if "S_rshape_dens" in H5conf.keys:
        Sce2D_dens = H5conf.Sce_dens * H5conf.S_rshape_dens[np.newaxis, :]

    # Determine the min/max limit of the colorbars
    v1 = np.min(dnGC_dt[:, irho_range])
    v2 = np.max(dnGC_dt[:, irho_range])
    vlim = max(-v1, v2)

    # Error: dnGC/dt+div(Gamma_tot)-Sce_n for each species
    ax1 = fig.add_subplot(1, 1, 1)
    p1 = ax1.pcolormesh(
        rhog[irho_range],
        timeg,
        dnGC_dt[:, irho_range]
        + dGamma_vE_dr[:, irho_range]
        + dGamma_vD_dr[:, irho_range]
        + dGamma_diffus_dr[:, irho_range]
        - Sce2D_dens[:, irho_range],
        vmin=-0.1 * vlim,
        vmax=0.1 * vlim,
        shading="gouraud",
        cmap="bwr",
    )
    ax1.set_xlabel("$\\rho=r/a$", size=18, labelpad=20)
    ax1.set_ylabel("time", size=18, labelpad=20)
    ax1.axis("tight")
    ax1.set_title(r"$dn_{GC}/dt+div(\Gamma_{Tot.})-Sce_n$", size=16)
    fig.colorbar(p1)
    figName = (
        outputDir + "/ConservDensity_" + speciesName + "_t" + str(timeEnd) + ".png"
    )
    print("Save figure:", figName)
    mpp.savefig(figName, transparent=False, format="png")


# -----------------------------------------------------------
#  MAIN PROGRAM
# -----------------------------------------------------------
if __name__ == "__main__":

    # Read command line
    inputDir = read_args(sys.argv[1:])

    DirSp0 = inputDir
    # DirSp1 = inputDir+'/sp1/'

    # Reading of initstate
    H5conf = H5ut.loadHDF5(DirSp0 + "init_state/init_state_r001.h5")
    H5magnet = H5ut.loadHDF5(DirSp0 + "init_state/magnet_config_r001.h5")
    H5mesh = H5ut.loadHDF5(DirSp0 + "init_state/mesh5d_r001.h5")
    H5conf.append(H5magnet)
    H5conf.append(H5mesh)

    # Reading of the Phi2D files
    Phi2dFileNames = DirSp0 + "Phi2D/Phi2D_d009*.h5"
    Phi2dFileList = glob.glob(Phi2dFileNames)
    Phi2dFileList.sort()
    H5Phi2D = H5ut.loadHDF5(Phi2dFileList)

    # Reading of the files rprof/rprof_part_d*.h5 for species 0
    # rprofpSp0FileNames = DirSp0 + "rprof/rprof_part_d009*.h5"
    # rprofpSp0FileList = glob.glob(rprofpSp0FileNames)
    # rprofpSp0FileList.sort()
    # H5rprofSp0P = H5ut.loadHDF5(rprofpSp0FileList)

    # Reading of the files rprof/rprof_GC_d*.h5 for species 0
    # rprofgcSp0FileNames = DirSp0 + 'rprof/rprof_GC_d009*.h5'
    # rprofgcSp0FileList  = glob.glob(rprofgcSp0FileNames)
    # rprofgcSp0FileList.sort()
    # H5rprofSp0GC = H5ut.loadHDF5(rprofgcSp0FileList)

    # Reading of the files rprof/rprof_GC_d*.h5 for species 1
    # rprofgcSp1FileNames = DirSp1 + 'rprof/rprof_GC_d*.h5'
    # rprofgcSp1FileList  = glob.glob(rprofgcSp1FileNames)
    # rprofgcSp1FileList.sort()
    # H5rprofSp1GC = H5ut.loadHDF5(rprofgcSp1FileList)

    # Plot Phi2D
    # outputDir = "./GYSELA_figures"
    # PlotPhi2D(H5conf, H5rprofSp0P, H5Phi2D, outputDir)
    # PlotPhi2D_wavelets(0.3, H5conf, H5rprofSp0P, H5Phi2D, outputDir)
    # PlotPhi2D_zfp(3, H5conf, H5rprofSp0P, H5Phi2D, outputDir)
    # PlotPhi2D_EZW(25, H5conf, H5rprofSp0P, H5Phi2D, outputDir)
    # PlotPhi2Dmostunstable(H5conf, H5Phi2D, outputDir)
    modes_m0, modes_mn = GetPhi2Dmostunstable(H5conf, H5Phi2D)
    modes_tensor = fourier_diag_to_tensor(modes_m0, modes_mn)
    print(modes_tensor)
    print(modes_tensor.shape)
    # PlotPhi2Dmostunstable_wavelets(0.3, H5conf, H5Phi2D, outputDir)
    # PlotPhi2Dmostunstable_zfp(3, H5conf, H5Phi2D, outputDir)
    # PlotPhi2Dmostunstable_EZW(10, H5conf, H5Phi2D, outputDir)
    # PlotConservDensity(H5conf,H5rprofSp0GC,"species0",outputDir)
    # PlotConservDensity(H5conf,H5rprofSp1GC,"species1",outputDir)
