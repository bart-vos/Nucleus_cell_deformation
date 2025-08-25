import numpy as np
import matplotlib.pyplot as plt
import seaborn
from PIL import Image

seaborn.set_context("poster", font_scale=1.0)
seaborn.set_style("white")
seaborn.set_style("ticks")

from fluorescence_oocyte_deformation_functions import analyse_contour_FFT
from fluorescence_oocyte_deformation_functions import plot_reconstruction


plt.close('all')


#%% Loading the files

contour_coords_cell_def = np.loadtxt('Contour_coords_cell_def.txt')
contour_coords_nucl_def = np.loadtxt('Contour_coords_nucl_def.txt')
contour_coords_cell_undef = np.loadtxt('Contour_coords_cell_undef.txt')
contour_coords_nucl_undef = np.loadtxt('Contour_coords_nucl_undef.txt')


im_undef = Image.open('cell_undef.tif')
im_undef = np.array(im_undef)

im_def = Image.open('cell_def.tif')
im_def = np.array(im_def)


# Now we analyse the contour with a FFT. Note that we create a complex number in the FFT
FFT_cell_array = np.zeros([2,1000])*1j
FFT_nucl_array = np.zeros([2,1000])*1j

xcoords_cell_undef, ycoords_cell_undef, r_cell_undef, rf_ratio_cell_undef, FFT_cell_array[0,:], center_cell_undef = analyse_contour_FFT(contour_coords_cell_undef)
xcoords_nucl_undef, ycoords_nucl_undef, r_nucl_undef, rf_ratio_nucl_undef, FFT_nucl_array[0,:], center_nucl_undef = analyse_contour_FFT(contour_coords_nucl_undef)
xcoords_cell_def, ycoords_cell_def, r_cell_def, rf_ratio_cell_def, FFT_cell_array[1,:], center_cell_def = analyse_contour_FFT(contour_coords_cell_def)
xcoords_nucl_def, ycoords_nucl_def, r_nucl_def, rf_ratio_nucl_def, FFT_nucl_array[1,:], center_nucl_def = analyse_contour_FFT(contour_coords_nucl_def)


# SNAR: F2-19 divided by F0
SNAR_cell_undef = np.sum(np.abs(FFT_cell_array[0,2:20])) / np.abs(FFT_cell_array[0,0])
SNAR_nucl_undef = np.sum(np.abs(FFT_nucl_array[0,2:20])) / np.abs(FFT_nucl_array[0,0])
SNAR_cell_def   = np.sum(np.abs(FFT_cell_array[1,2:20])) / np.abs(FFT_cell_array[1,0])
SNAR_nucl_def   = np.sum(np.abs(FFT_nucl_array[1,2:20])) / np.abs(FFT_nucl_array[1,0])

# NCDC
NCDC = (SNAR_nucl_undef - SNAR_nucl_def) / (SNAR_cell_undef - SNAR_cell_def)

print('NCDC = '+str(NCDC))



# Show the contour outline together with the original image
plt.figure()
plt.imshow(im_undef/np.max(im_undef)*255, cmap='gray')

plot_reconstruction(FFT_cell_array[0,:], center_cell_undef, 'red')
plot_reconstruction(FFT_nucl_array[0,:], center_nucl_undef, 'blue')

plt.axis('off')
plt.gca().set_aspect('equal')
plt.show()


plt.figure()
plt.imshow(im_def/np.max(im_def)*255, cmap='gray')

plot_reconstruction(FFT_cell_array[1,:], center_cell_def, 'red')
plot_reconstruction(FFT_nucl_array[1,:], center_nucl_def, 'blue')

plt.axis('off')
plt.gca().set_aspect('equal')
plt.show()

