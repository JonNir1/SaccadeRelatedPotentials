from mne.datasets import sample
from mne.io import read_raw_fif

import matplotlib
matplotlib.use('TkAgg')     # or 'Qt5Agg'


EEG_REF = "EEG 010"

fname = sample.data_path() / "MEG" / "sample" / "sample_audvis_raw.fif"
raw = read_raw_fif(fname, preload=False)
raw.crop(0, 60).pick("eeg")
raw.load_data()
raw.filter(1., 40.)

# raw.set_eeg_reference("average")
# spectrum = raw.compute_psd()

raw.set_eeg_reference(ref_channels=[EEG_REF])
spectrum = raw.copy().compute_psd(n_fft=512, verbose=False, exclude=[EEG_REF])
print((spectrum.data.min(), spectrum.data.max()))

spectrum.plot(average=False, picks=['eeg', 'eog'])

