from utils import *
import pickle

x = pickle.load(open("data_challenge_spectra_v01.pickle", "rb"))


def summarize_instrument(d):
    for x in d:
        print(f"{x} = {np.shape(d[x])}")


def summarize_all_models():
    for k in model_dictionaries:
        print(k)
        print(model_dictionaries[k])
        print()


models = {}
for k in ["NIRSpec", "NIRCam"]:
    planet = x[f"WASP39b_{k}"]
    table = Table(
        dict(wavelength=planet["wl"], depth=planet["transmission"], stellar_flux=planet["stellar_flux"]),
        meta=x["WASP39b_parameters"],
    )
    models[k] = PlanetarySpectrumModel(table=table, label="injected model")
