import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
import astropy.units as u


def plot_features(table, ax=None, **kw):
    """
    Plot some planetary features.

    Parameters
    ----------
    table : astropy.table.Table
        A table containing wavelengths, features, uncertainties.
    """

    lower = table["wavelength"] - table["wavelength_lower"]
    upper = table["wavelength_upper"] - table["wavelength"]

    errorkw = dict(marker="o", linewidth=0, elinewidth=1)
    errorkw.update(**kw)

    if ax is None:
        plt.figure(figsize=(8, 3), dpi=300)
    else:
        plt.sca(ax)
    plt.errorbar(
        table["wavelength"],
        table["depth"],
        yerr=table["uncertainty"],
        xerr=[lower, upper],
        **errorkw,
    )
    plt.xlabel("Wavelength (micron)")
    plt.ylabel("Depth (unitless)")


def load_feature_file(filename="example-file-of-features.txt", **kw):
    """
    Read a fitted planetary features file.

    Parameters
    ----------
    filename : str
        The filename of the example file.
    kw : dict
        Extra keywords will be passed along
        as options to astropy.io.ascii.read
    """

    # load the table
    t = ascii.read(filename, **kw)

    # make sure the values and uncertainties are set
    for k in ["depth", "uncertainty"]:
        if k not in t.columns:
            raise RuntimeError(f'"{k}" not found in {filename}')

    if "wavelength" not in t.columns:
        t["wavelength"] = 0.5 * (t["wavelength_lower"] + t["wavelength_upper"])

    if ("wavelength_lower" not in t.columns) and ("wavelength_upper" not in t.columns):
        bin_widths = np.gradient(t["wavelength"])
        t["wavelength_lower"] = t["wavelength"] - bin_widths / 2
        t["wavelength_upper"] = t["wavelength"] + bin_widths / 2
    return t


def write_example_file(
    filename="example-file-of-features.txt",
    include_edges=False,
    include_units=False,
    N=10,
):
    """
    Write out an example fitted planetary features file.

    Parameters
    ----------
    filename : str
        The filename of the example file.
    include_edges : bool
        Should the edges of the bins be explicitly stated?
    include_unts : bool
        Should astropy units be included?
    N : int
        The number of data points in the fake file.
    """

    # create an empty dictionary
    columns = {}

    # define a constant R wavelength grid
    wavelength_in_microns = np.logspace(0, 1, N)

    if include_units:
        wavelength_in_microns *= u.micron

    # (approximate) definition of bin edges if necessary
    if include_edges:
        dw = np.gradient(wavelength_in_microns)
        columns["wavelength_lower"] = wavelength_in_microns - dw / 2
        columns["wavelength_upper"] = wavelength_in_microns + dw / 2
    else:
        columns["wavelength"] = wavelength_in_microns

    # set up some (not-constant) depth uncertainties
    depth_uncertainties = np.linspace(0.001, 0.003, N)

    # create some fake depths
    depth = np.random.normal(0.02, depth_uncertainties)
    columns["depth"] = depth
    columns["uncertainty"] = depth_uncertainties

    # store these three columns in an astropy table
    t = Table(columns)

    # write out this table
    t.write(filename, format="ascii.ecsv", overwrite=True)


