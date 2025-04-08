import matplotlib
import mne
from mne.datasets import sample
from mne.io import read_raw_fif
import easygui_qt.easygui_qt as gui
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication

matplotlib.use('Qt5Agg')
matplotlib.interactive(False)
mne.viz.set_browser_backend('qt')   # or "matplotlib"

EEG_REF = "EEG 010"

fname = sample.data_path() / "MEG" / "sample" / "sample_audvis_raw.fif"
raw = read_raw_fif(fname, preload=False, verbose=False)
raw.crop(0, 60).pick("eeg")
raw.load_data(verbose=False)
raw.filter(1., 40., verbose=False)

# raw.set_eeg_reference("average")
raw.set_eeg_reference(ref_channels=[EEG_REF], verbose=False)

# spectrum = raw.compute_psd(n_fft=512, verbose=False, exclude=[EEG_REF])
# spectrum.plot(average=False, picks=['eeg', 'eog'])

input1 = gui.get_continue_or_cancel(title="Input 1", message="", continue_button_text="T", cancel_button_text="F")
print("1) QT Running: ", QApplication.instance())

fig = raw.plot(verbose=False, show=False)    # can set show=False with same downstream effect
plt.show(block=True)
plt.close('all')
plt.ioff()                        # this has no effect
# matplotlib.interactive(False)     # this has no effect

app = QApplication.instance()
print("fig) QT Running: ", app)
if app is not None:
    app.processEvents()
    # app.quit()

input2 = gui.get_continue_or_cancel(title="Input 2", message="", continue_button_text="T", cancel_button_text="F")
print("2) QT Running: ", QApplication.instance())
print(f"In 1: {input1}\t\tIn 2: {input2}")
