# %%

#Package installation in terminal needed for deepbrain package (mask generation)

#pip install tensorflow==1.13.1
#pip install tensorflow-gpu==1.13.1
#pip install deepbrain

# %%

# Part II
# Importing libraries

import nibabel as nib
import os.path
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt



#Image loading

interictal = 'images/prINTERICTAL.nii'
ictal = 'images/pICTAL.nii'
rm = 'images/RM.nii'

def load_image(image):
    if os.path.isfile(image):
        #loading the image
        img_in = nib.nifti1.load(image)
        #extracting the data

        #get_fdata() gets the matrix-like info of the data, the values of the actual voxels of the 3D volume
        data_in = img_in.get_fdata()
        #shape gets the dimensions of this 3D image
        dim_in = img_in.shape
        #header gives the complementary info of the image saved on the "head"
        header_in = img_in.header
        #affine gives the transformations applied to the 3D matrix to actually generate the final image (translations, rotations etc.)
        affine_in = img_in.affine

        return img_in, data_in, dim_in, header_in, affine_in
    
    else:
        print("No such archive in path: "+image)
    
#Loading the Interictal, Ictal and RM respectively.
interictal_in, data_interictal_in, dim_interictal_in, header_interictal_in, affine_interictal_in = load_image(interictal)
ictal_in, data_ictal_in, dim_ictal_in, header_ictal_in, affine_ictal_in = load_image(ictal)
rm_in, data_rm_in, dim_rm_in, header_rm_in, affine_rm_in = load_image(rm)





# Plotting the three images


# cut in x
sagittal = -45
# cut in y
coronal = -5
# cut in z
axial = -52

# Coordinates assigned to the plots
coords = [sagittal, coronal, axial]

#plot de les tres imatges
plotting.plot_epi(interictal, cut_coords = coords)
plotting.plot_epi(ictal, cut_coords = coords)
plotting.plot_anat(rm, cut_coords = coords)

# %%
#Generating a mask with Extractor() function from deepbrain library. This library needs tensorflow version 1.13.1
from deepbrain import Extractor
ext = Extractor()

# `prob` will be a 3D image containing probability of being brain tissue for each of the voxels in `img`
prob = ext.run(data_rm_in) 

# mask obtained as voxels with probabiliti higher than 50% of being brain tissue

mask = prob > 0.5



#%%
#MASK IMPLEMENTATION MRI

#Assigning the image value to each voxel with brain tissue to obtain the image with mask applied
rm_masked = data_rm_in*mask #put the image value if the voxel is part of the brain

rm_mask_path = "generated_images/rm_mask.nii" #path of new image
rmmask_image = nib.Nifti1Image(rm_masked, affine_rm_in) #nii creation
nib.save(rmmask_image, rm_mask_path) #save

# Plot of new masked rm image
plotting.plot_anat(rm_mask_path, cut_coords = coords)




# %%
#MASK IMPLEMENTATION SPECT

#Applying mask to SPECT. Warning: if image shape and mask does not match, there must be a problem with image path or loading
#as mask has been directly created from rm image, which was used as reference image for spm12.
data_interictal_masked = data_interictal_in[(mask==True)] #generate a new data matrix just taking values from in-brain voxels 
data_ictal_masked = data_ictal_in[(mask==True)] #same for ictal



