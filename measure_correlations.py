"""
Measure correlation functions from generated fields using TreeCorr.

TreeCorr handles spin-weighted correlation functions properly:
    - GGCorrelation: spin-2 × spin-2 (shear)
    - KKCorrelation: spin-0 × spin-0 (convergence)
    - NNCorrelation: number counts (for position correlations)
"""

import numpy as np
import treecorr


def create_catalog_from_field(field1, field2, box_size_deg, field_type="shear"):
    """
    Create a TreeCorr catalog from a 2D field.

    For spin-2 (shear): field1=γ₁, field2=γ₂ → uses g1, g2
    For spin-1 (velocity): field1=v_x, field2=v_y → uses v1, v2
    For spin-0 (scalar): field1=κ, field2=None → uses k

    Parameters
    ----------
    field1, field2 : array
        Field components on 2D grid
    box_size_deg : float
        Physical size of the box in degrees
    field_type : str
        "shear" (spin-2), "velocity" (spin-1), or "scalar" (spin-0)

    Returns
    -------
    treecorr.Catalog
    """
    nx, ny = field1.shape
    x = np.linspace(0, box_size_deg, nx, endpoint=False)
    y = np.linspace(0, box_size_deg, ny, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    kwargs = {"x": xx.flatten(), "y": yy.flatten(), "x_units": "deg", "y_units": "deg"}

    if field_type == "shear":
        kwargs["g1"], kwargs["g2"] = field1.flatten(), field2.flatten()
    elif field_type == "velocity":
        kwargs["v1"], kwargs["v2"] = field1.flatten(), field2.flatten()
    else:  # scalar
        kwargs["k"] = field1.flatten()

    return treecorr.Catalog(**kwargs)


def measure_gg_correlation(g1, g2, box_size_deg, nbins=20, min_sep=1.0, max_sep=300.0):
    """
    Measure shear-shear correlation function using TreeCorr.

    Parameters
    ----------
    g1, g2 : array
        Shear components (γ₁, γ₂) on 2D grid
    box_size_deg : float
        Physical size of the box in degrees
    nbins : int
        Number of angular bins
    min_sep, max_sep : float
        Min/max separation in arcminutes

    Returns
    -------
    dict with keys:
        'theta': mean separation in arcmin
        'xi_plus': ξ₊ correlation
        'xi_minus': ξ₋ correlation
        'sigma_plus': error on ξ₊
        'sigma_minus': error on ξ₋
        'npairs': number of pairs per bin
    """
    cat = create_catalog_from_field(g1, g2, box_size_deg, field_type="shear")
    gg = treecorr.GGCorrelation(
        nbins=nbins,
        min_sep=min_sep,
        max_sep=max_sep,
        sep_units="arcmin",
        bin_slop=0.1,
    )
    gg.process(cat)

    return {
        "theta": np.exp(gg.meanlogr),  # arcmin
        "xi_plus": gg.xip,
        "xi_minus": gg.xim,
        "sigma_plus": np.sqrt(gg.varxip),
        "sigma_minus": np.sqrt(gg.varxim),
        "npairs": gg.npairs,
    }


def measure_kk_correlation(kappa, box_size_deg, nbins=20, min_sep=1.0, max_sep=300.0):
    """
    Measure convergence auto-correlation using TreeCorr.

    Parameters
    ----------
    kappa : array
        Convergence field on 2D grid
    box_size_deg : float
        Physical size of the box in degrees
    nbins : int
        Number of angular bins
    min_sep, max_sep : float
        Min/max separation in arcminutes

    Returns
    -------
    dict with keys:
        'theta': mean separation in arcmin
        'xi': correlation function
        'sigma': error estimate
        'npairs': number of pairs per bin
    """
    cat = create_catalog_from_field(kappa, None, box_size_deg, field_type="scalar")
    kk = treecorr.KKCorrelation(
        nbins=nbins,
        min_sep=min_sep,
        max_sep=max_sep,
        sep_units="arcmin",
        bin_slop=0.1,
    )
    kk.process(cat)

    return {
        "theta": np.exp(kk.meanlogr),
        "xi": kk.xi,
        "sigma": np.sqrt(kk.varxi),
        "npairs": kk.npairs,
    }


def measure_gv_correlation(g1, g2, v1, v2, box_size_deg, nbins=20, min_sep=1.0, max_sep=300.0):
    """
    Measure shear-velocity cross-correlation using TreeCorr GVCorrelation.

    GVCorrelation computes spin-2 × spin-1 correlation:
        ξ₊ = ⟨γ v*⟩  (spin |2-1| = 1, uses J₁ Bessel function)
        ξ₋ = ⟨γ v⟩   (spin 2+1 = 3, uses J₃ Bessel function)

    Parameters
    ----------
    g1, g2 : array
        Shear components (spin-2)
    v1, v2 : array
        Velocity components (spin-1)
    box_size_deg : float
        Physical size of the box in degrees

    Returns
    -------
    dict with cross-correlation results
    """
    cat_g = create_catalog_from_field(g1, g2, box_size_deg, field_type="shear")
    cat_v = create_catalog_from_field(v1, v2, box_size_deg, field_type="velocity")

    gv = treecorr.GVCorrelation(
        nbins=nbins,
        min_sep=min_sep,
        max_sep=max_sep,
        sep_units="arcmin",
        bin_slop=0.1,
    )
    gv.process(cat_g, cat_v)

    return {
        "theta": np.exp(gv.meanlogr),
        "xi_plus": gv.xip,
        "xi_minus": gv.xim,
        "sigma_plus": np.sqrt(gv.varxip),
        "sigma_minus": np.sqrt(gv.varxim),
        "npairs": gv.npairs,
    }


def measure_vv_correlation(v1, v2, box_size_deg, nbins=20, min_sep=1.0, max_sep=300.0):
    """
    Measure velocity-velocity auto-correlation using TreeCorr VVCorrelation.

    VVCorrelation computes spin-1 × spin-1 correlation:
        ξ₊ = ⟨v v*⟩  (spin 0, uses J₀ Bessel function)
        ξ₋ = ⟨v v⟩   (spin 2, uses J₂ Bessel function)

    Parameters
    ----------
    v1, v2 : array
        Velocity components (spin-1)
    box_size_deg : float
        Physical size of the box in degrees

    Returns
    -------
    dict with correlation results
    """
    cat = create_catalog_from_field(v1, v2, box_size_deg, field_type="velocity")
    vv = treecorr.VVCorrelation(
        nbins=nbins,
        min_sep=min_sep,
        max_sep=max_sep,
        sep_units="arcmin",
        bin_slop=0.1,
    )
    vv.process(cat)

    return {
        "theta": np.exp(vv.meanlogr),
        "xi_plus": vv.xip,
        "xi_minus": vv.xim,
        "sigma_plus": np.sqrt(vv.varxip),
        "sigma_minus": np.sqrt(vv.varxim),
        "npairs": vv.npairs,
    }


def measure_all_correlations(fields, box_size_deg, nbins=20, min_sep=1.0, max_sep=300.0):
    """
    Measure all relevant correlation functions from generated fields.

    Parameters
    ----------
    fields : dict
        Dictionary with keys 'kappa', 'g1', 'g2', 'v1', 'v2'
    box_size_deg : float
        Physical size of the box in degrees
    nbins : int
        Number of angular bins
    min_sep, max_sep : float
        Min/max separation in arcminutes

    Returns
    -------
    dict with correlation function results
    """
    results = {}
    results["gg"] = measure_gg_correlation(
        fields["g1"], fields["g2"], box_size_deg, nbins, min_sep, max_sep
    )
    results["kk"] = measure_kk_correlation(
        fields["kappa"], box_size_deg, nbins, min_sep, max_sep
    )
    if "v1" in fields and "v2" in fields:
        results["gv"] = measure_gv_correlation(
            fields["g1"], fields["g2"],
            fields["v1"], fields["v2"],
            box_size_deg, nbins, min_sep, max_sep
        )
        results["vv"] = measure_vv_correlation(
            fields["v1"], fields["v2"],
            box_size_deg, nbins, min_sep, max_sep
        )
    return results
