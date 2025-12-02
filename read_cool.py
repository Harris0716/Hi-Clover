import cooler

c = cooler.Cooler("GSM2453281_TAM-R1.10kb.cool.HDF5")
print(c.info)
print(c.shape)
