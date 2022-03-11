import glob, os

import numpy as np
import astropy.io.fits as fits
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

class Data:
	'''
	A simple class to hold data

	Using **kwargs to keep things future proof, see parameters in load_data 
	'''
	def __init__(self, filepath, **kwargs):
		# Load all relevant arguments that have been provided, set to defaults if not. 
		self.sci_ext = kwargs.get('sci_ext', 'SCI')
		self.err_ext = kwargs.get('err_ext', 'ERR')
		self.suffix = kwargs.get('suffix', 'calints.fits')
		self.x_ax = kwargs.get('x_ax', 2)
		self.y_ax = kwargs.get('y_ax', 1)
		self.t_ax = kwargs.get('t_ax', 0)

		# Load the data 
		self.load_data(filepath, self.sci_ext, self.err_ext, self.suffix, self.x_ax, self.y_ax, self.t_ax)

		# Load a seed image if possible 
		# if (self.instrument == 'NIRCAM') or (self.instrument == 'MIRI'):
		# 	self.load_seed()
		# else:
		# 	self.seed = None

		# Create a dictionary that we can save outputs to
		self.save_dict = {}

	def load_data(self, filepath, sci_ext='SCI', err_ext='ERR', suffix='calints.fits', x_ax=2, y_ax=1, t_ax=0):
		'''
		Function to load JWST pipeline Stage 2 equivalent data (e.g. calints.fits files)
		'''

		# Check if filepath is a directory, or a file
		if os.path.isdir(filepath):
			all_files = sorted(glob.glob(filepath+'*'+suffix))
		elif os.path.isfile(filepath):
			all_files = [filepath]
		else:
			raise ValueError('File path not recognised as a file or directory.')

		# Raise error if no files found.
		if not all_files:
			raise ValueError('No files found at provided filepath!')

		# Loop over provided files
		for i, file in enumerate(all_files):
			# Open FITS file
			with fits.open(file) as hdul:
				# Read in science and error data using default / provided extensions. 
				sci = hdul[sci_ext].data
				err = hdul[err_ext].data

				if i == 0:
					# If this is the first file, let's initialise some arrays to start
					# saving data into 
					all_sci = sci
					all_err = err

					#Also, let's take this moment to get other information from the headers
					phead = hdul[0].header

					self.instrument = phead['INSTRUME'].upper()
				else:
					# Append data to the existing arrays (probably a smarter way to do this)
					all_sci = np.append(all_sci, sci, axis=t_ax)
					all_err = np.append(all_err, err, axis=t_ax)

		# If necessary, transpose array to a general format that the rest of the code will understand.
		if (x_ax != 2) or (y_ax != 1) or (t_ax != 0):
			all_sci = np.transpose(all_sci, axes=(t_ax, y_ax, x_ax))
			all_err = np.transpose(all_err, axes=(t_ax, y_ax, x_ax))

		# We can't use all the integrations (it would take too long), 
		# so pick out specific ones depending on the dataset. 
		if self.instrument == 'NIRCAM':
			self.use_ints = [1,20,32,47,-1]
			plot_aspect = 10
			vl, vh = 0.01, 0.6
			self.px, self.py = 1024, 32
		elif self.instrument == 'NIRISS':
			self.use_ints = [1,2,3,4,-1]
			plot_aspect=2
			vl, vh = 0.001, 0.05
			self.px, self.py = 1600, 30
		elif self.instrument == 'NIRSPEC':
			self.use_ints = [1,10,15,20,-1]
			plot_aspect = 5
			vl, vh = 0.01, 0.6
			self.px, self.py = 256, 16
		elif self.instrument == 'MIRI':
			self.use_ints = [1,2,3,4,-1]
			plot_aspect=2
			vl, vh = 0.01, 0.6
			self.px, self.py = 37, 220
		else:
			raise ValueError('Instrument:{} not recognised!'.format(self.instrument))

		# Want to track a single pixel across all integrations for the dataset
		self.single_pixel = all_sci[:,self.py,self.px]

		# Save those specific integrations to the class
		self.sci = all_sci[self.use_ints,:,:]
		self.err = all_err[self.use_ints,:,:]

		# Explicitly free the memory being used to hold all of the data.
		# This might not be necessary, but who knows.
		del all_sci
		del all_err

		# Make a quick plot of the first integration image 
		plt.figure(figsize=(24,6))
		ax = plt.gca()

		ax.imshow(self.sci[0], aspect=plot_aspect, norm=LogNorm(vmin=vl*np.nanmax(self.sci[0]), vmax=vh*np.nanmax(self.sci[0])))
		ax.set_title('Wow, what a lovely {} spectrum!'.format(self.instrument), fontsize=16)
		ax.tick_params(axis='both', labelsize=16)

		return

	def basic_properties(self):
		# Function to calculate a range of basic properties for the images. 

		# Plot some profiles of the dispersion/cross-dispersion axes
		self.profiles(self.sci)

		# Make some plots based on the image pixels
		self.pixelplots(self.sci)

		# Get some simple quantitative measures from the images
		self.quantitatives(self.sci)

		return

	def seed_comparison(self):
		# Function to calculate comparisons to the seed images. 

		# Can only do this for NIRCam and NIRISS
		if (self.instrument == 'NIRSPEC') or (self.instrument == 'MIRI'):
			print('There is no seed image for {}, skipping this step!'.format(self.instrument))
		else:
			# We are using NIRCam or NIRISS
			seed_image = load_seed()

			# Now subtract the seed image from the science images
			subtracted = self.sci - seed_image

			# Now run the functions from basic_properties again
			# Plot some profiles of the dispersion/cross-dispersion axes
			self.profiles(self.sci)

			# Make some plots based on the image pixels
			self.pixelplots(self.sci)

			# Get some simple quantitative measures from the images
			self.quantitatives(self.sci)

		return

	def load_seed(self):
		# Function to load the seed image in as part of the class.

		return


	def profiles(self, images):
		# Create 1D profiles from a given image
		fig, axs = plt.subplots(1, 2, figsize=(24,6), gridspec_kw={'width_ratios': [2, 1]})
		axs[0].set_title('Summed Dispersion Profile', fontsize=24)
		axs[1].set_title('Summed X-Dispersion Profile', fontsize=24)

		#Loop over the images we are interested in
		for i, image in enumerate(images):
			disp = np.nansum(image, axis=0)
			xdisp = np.nansum(image, axis=1)

			axs[0].plot(range(len(disp)), disp, label='Integration {}'.format(self.use_ints[i]))
			axs[1].plot(range(len(xdisp)), xdisp, label='Integration {}'.format(self.use_ints[i]))

		plt.subplots_adjust(wspace=0.1)
		for ax in axs:
			ax.yaxis.get_offset_text().set_fontsize(18)
			ax.tick_params(axis='both', labelsize=18)
			ax.set_xlabel('Pixels', fontsize=20)
			ax.legend(prop={'size': 16})

		axs[0].set_ylabel('Counts', fontsize=20)
		plt.show()

		return 

	def pixelplots(self, images):
		# Create some plots from the pixel values
		fig, axs = plt.subplots(1, 2, figsize=(24,6))
		axs[0].set_title('Histogram of Pixel Values', fontsize=24)
		axs[1].set_title('Pixel X={}, Y={}'.format(self.px, self.py), fontsize=24)

		#Plot the histogram of pixel values for the specific integrations
		for i, image in enumerate(images):
			image_1d = image.flatten()

			bins = np.logspace(np.log10(1e-3), np.log10(np.nanmax(image_1d)), 100)

			axs[0].hist(image_1d, bins=bins, histtype='step', label='Integration {}'.format(self.use_ints[i]))

		#Use the single pixel data from the load_data function to plot it's flux over time
		axs[1].plot(range(len(self.single_pixel)), self.single_pixel)

		axs[0].legend(prop={'size': 16})
		axs[0].set_xscale('log')
		axs[0].set_xlabel('Pixel Flux / Counts', fontsize=20)
		axs[0].set_ylabel('Number of Pixels', fontsize=20)

		axs[1].set_xlabel('Integration #', fontsize=20)
		axs[1].set_ylabel('Pixel Flux / Counts', fontsize=20)

		for ax in axs:
			ax.tick_params(axis='both', labelsize=18)

		plt.show()

	def quantitatives(self, images):
		# Set up arrays to save things into 
		means = np.empty(len(self.use_ints))
		medians = np.empty(len(self.use_ints))
		vmaxs = np.empty(len(self.use_ints))
		vmins = np.empty(len(self.use_ints))
		stds = np.empty(len(self.use_ints))

		# Loop over the images we are interested in
		for i, image in enumerate(images):

			means[i] = np.nanmean(image)
			medians[i] = np.nanmedian(image)
			vmaxs[i] = np.nanmax(image)
			vmins[i] = np.nanmin(image)
			stds[i] = np.nanstd(image)

		# Print things out for people to see
		print('Integration #\'s: ', self.use_ints)
		print('Mean Values: ',means)
		print('Median Values: ',medians)
		print('Min Values: ',vmins)
		print('Max Values: ',vmaxs)
		print('Standard Dev Values: ',stds)

		# Now save to the global dictionary
		self.save_dict['means'] = means
		self.save_dict['medians'] = medians
		self.save_dict['vmins'] = vmins
		self.save_dict['vmaxs'] = vmaxs
		self.save_dict['stds'] = stds

		return

	def residual_analysis():
		# TODO
		return

	def bad_pixels(self, sigma=10):

		# Get number of pixels in a single image
		npixels = self.sci[0].shape[0] * self.sci[0].shape[1]

		# Create an empty array to assign the number of outliers
		nints = self.sci.shape[0]
		all_n_outliers = np.empty(nints)

		# Function to calculate median and standard deviation of pixel values.  
		def identify_std(values):
			std = np.nanstd(values)
			return std
		def identify_median(values):
			median = np.nanmedian(values)
			return median

		# Describe a footprint to apply function (i.e. neighbouring pixels)
		footprint =  np.array([[1,1,1],[1,0,1],[1,1,1]])

		# Run filter functions
		for i, image in enumerate(self.sci):
			# Get standard deviation and median images
			stds = ndimage.generic_filter(image, identify_std, footprint=footprint, mode='constant', cval=np.nan)
			meds = ndimage.generic_filter(image, identify_median, footprint=footprint, mode='constant', cval=np.nan)

			# Identify outliers
			outlier_image = np.divide(np.subtract(image,meds), stds)

			# Count outliers
			n_outliers = np.count_nonzero(outlier_image > 10)

			# Save to array
			all_n_outliers[i] = n_outliers

		plt.figure()
		ax = plt.gca()
		ax.scatter(range(len(all_n_outliers)), all_n_outliers/npixels*100)
		ax.set_xlabel('Integration #', fontsize=16)
		ax.set_ylabel('Bad Pixel % (10$\sigma$)', fontsize=16)
		ax.set_xticks(range(len(self.use_ints)), labels=self.use_ints)

		#Save to global dict
		self.save_dict['n_outliers'] = all_n_outliers

		return

	def background_trend():
		# TODO

		return

	def summary():
		# TODO

		return





