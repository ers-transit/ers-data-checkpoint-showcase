import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
import astropy.units as u
import glob, warnings, os
from astropy.visualization import quantity_support


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


def standardise_table_format(submitted_data):
    # module to standardize the astropy table format (a bit hacky at the moment)
    colnames = submitted_data.colnames

    if "wavelength" not in colnames:
        submitted_data.rename_column(colnames[0], "wavelength")

    if "depth (ppt)" in colnames:
        submitted_data.rename_column("depth (ppt)", "depth")
        submitted_data["depth"] = submitted_data["depth"] / 1000
    elif "Transit Depth (ppm)" in colnames:
        submitted_data.rename_column("Transit Depth (ppm)", "depth")
        submitted_data["depth"] = submitted_data["depth"] / 1000000
    elif "depth" not in colnames:
        submitted_data.rename_column(colnames[1], "depth")

    if "depth err (ppt)" in colnames:
        submitted_data.rename_column("depth err (ppt)", "uncertainty")
        submitted_data["uncertainty"] = submitted_data["uncertainty"] / 1000
    elif "Transit Depth error (ppm)" in colnames:
        submitted_data.rename_column("Transit Depth error (ppm)", "uncertainty")
        submitted_data["uncertainty"] = submitted_data["uncertainty"] / 1000000
    elif "uncertainty" not in colnames:
        submitted_data.rename_column(colnames[2], "uncertainty")

    return submitted_data


class PlanetarySpectrum:
    @property
    def wavelength(self):
        return self.table["wavelength"].quantity

    @property
    def depth(self):
        return self.table["depth"].quantity

    @property
    def uncertainty(self):
        if ("uncertainty_lower" in self.table.columns) and (
            "uncertainty_upper" in self.table.columns
        ):
            return np.array(
                [
                    self.table["uncertainty_lower"].quantity,
                    self.table["uncertainty_upper"].quantity,
                ]
            )
        else:
            return self.table["uncertainty"].quantity

    def __init__(self, table=None, label=None):
        """
        Initialize planetary spectrum object.

        Parameters
        ----------
        table : astropy.table.Table
            A table of depths (or other wavelength-dependent features).
            It should contain at least:
                + `wavelength` should represent the central wavelength of the
                   wavelength bin. Alternatively, there could be two columns
                   labeled `wavelength_lower` and `wavelength_upper` to represent
                   the lower and upper bounds of each wavelength bin. The units
                   should be in microns.
                + `depth` should be the transit depth $(R_p/R_\star)^2$ or the
                   eclipse depth ($F_p/F_\star$). This quantity should be unitless;
                   for example, a transit depth of 1% should be written as `0.01`.
                + `uncertainty` should be the uncertainty on the depth (if this
                   is data). This quantity should have the same units as depth
                   (so also be unitless).
             Planet parameters can/should be included as `meta` in this
             initializing astropy table.
         label : str
             A string labeling this planet spectrum. It might appear in the
             the display name for the object, plot legends, filenames, ...
        """

        # store the original data inputs
        self.table = Table(table)

        # store additional information that might be handy
        self.label = label

        # make sure the data format works
        self._validate_data()

    def _validate_data(self):
        """
        Make sure the data table has the right format.
        """

        # validate each core component
        self._validate_wavelengths()
        self._validate_depths()
        self._validate_uncertainties()

    def _validate_wavelengths(self):
        """
        Make sure wavelengths are usable.
        """

        # set centers from edges, if necessary
        if "wavelength" not in self.table.columns:
            self.table["wavelength"] = 0.5 * (
                self.table["wavelength_lower"] + self.table["wavelength_upper"]
            )

        # set edges from centers, if necessary
        if ("wavelength_lower" not in self.table.columns) and (
            "wavelength_upper" not in self.table.columns
        ):
            bin_widths = np.gradient(self.table["wavelength"])
            self.table["wavelength_lower"] = self.table["wavelength"] - bin_widths / 2
            self.table["wavelength_upper"] = self.table["wavelength"] + bin_widths / 2

        # make sure the units are good
        for k in ["wavelength", "wavelength_lower", "wavelength_upper"]:
            try:
                self.table[k] = self.table[k].to(u.micron)
            except (AttributeError, u.UnitConversionError):
                self.table[k] = self.table[k] * u.micron
                # warnings.warn(f"Assuming units for {k} are micron.")

        assert "wavelength" in self.table.columns

    def _validate_depths(self):
        """
        Make sure depths are usable.
        """
        if np.all(self.depth > 1):
            messages = """
            🪐 All depths are >1, implying the planet is
            bigger than the star. Depths, should be unitless,
            so a 1% transit should have a depth of 0.01.
            """
            warnings.warn(message)

    def _validate_uncertainties(self):
        """
        Make sure uncertainties are usable.
        """
        pass

    def __repr__(self):
        if "uncertainty" in self.table.columns:
            extra = " with uncertainties!"
        else:
            extra = ""
        return f"<🪐PlanetarySpectrum({len(self.wavelength)}w{extra})>"


class PlanetarySpectrumModel(PlanetarySpectrum):
    def plot(self, ax=None, **kw):
        """
        Plot the model.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes into which the plot should be drawn.
        kw : dict
            Extra keywords will be passed along to `plt.plot`
        """
        if ax is None:
            plt.figure(figsize=(8, 6), dpi=300)
            ax = plt.gca()
        else:
            plt.sca(ax)

        plot_kw = dict(zorder=-1, alpha=0.5, linewidth=3, label=self.label)
        plot_kw.update(**kw)
        plt.plot(self.table["wavelength"], self.table["depth"], **plot_kw)
        plt.xlabel("Wavelength (micron)")
        plt.ylabel("Depth (unitless)")
        return ax


class PlanetarySpectrumData(PlanetarySpectrum):
    def plot(self, ax=None, **kw):
        """
        Plot some planetary features.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes into which the plot should be drawn.
        """
        #         table = Table(self)

        lower = (self.table["wavelength"] - self.table["wavelength_lower"]).quantity
        upper = (self.table["wavelength_upper"] - self.table["wavelength"]).quantity

        errorkw = dict(marker="o", linewidth=0, elinewidth=1, label=self.label)
        errorkw.update(**kw)

        if ax is None:
            plt.figure(figsize=(8, 6), dpi=300)
            ax = plt.gca()
        else:
            plt.sca(ax)
        plt.errorbar(
            self.wavelength.value,
            self.depth.value,
            yerr=self.uncertainty.value,
            xerr=[lower.value, upper.value],
            **errorkw,
        )
        plt.xlabel("Wavelength (micron)")
        plt.ylabel("Depth (unitless)")
        return ax


def load_data(filename):

    # import each of the data files
    submitted_data = ascii.read(filename)

    # format the columns + column names into the correct format:
    formatted_data = standardise_table_format(submitted_data)

    # create a Planetary Spectrum object:
    data = PlanetarySpectrumData(formatted_data, label=filename.split("/")[-1])

    return data
