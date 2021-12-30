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



# %%

# NORMALISATION

# Intensity histogram for SPECTs without mask
plt.hist(data_ictal_in.flatten(), bins = 'auto')
plt.xlabel("Intensity")
plt.ylabel("Frequency (in counts)")
plt.ylim(0,1000000)
plt.title("Histogram SPECT Ictal")
plt.show()

plt.hist(data_interictal_in.flatten(), bins = 'auto')
plt.xlabel("Intensity")
plt.ylabel("Frequency (in counts)")
plt.title("Histogram SPECT Interictal")
plt.show()


# %%

# Intensity histogram for SPECTs with mask
freqs_ictal_masked, intensities_ictal_masked,_  = plt.hist(data_ictal_masked.flatten(), bins = 'auto')
plt.xlabel("Intensity")
plt.ylabel("Frequency (in counts)")
plt.title("Histogram SPECT Ictal masked")
plt.show()

freqs_interictal_masked, intensities_interictal_masked,_ = plt.hist(data_interictal_masked.flatten(), bins = 'auto')

plt.xlabel("Intensity")
plt.ylabel("Frequency (in counts)")
plt.title("Histogram SPECT Interictal with mask")
plt.show()



# %%
# Ictal clearly has a wider range of intensities, the point is wether the high intensity values are related to the epileptogenic zone or not.

# Option 1:
# Computing the normalisation factor as the ratio between the total intensity of the ictal image ans the total intensity of the interictal image

norm_factor_meanimask = data_ictal_masked.sum()/data_interictal_masked.sum() #calculem el factor de normalització com la suma de les intensitats de l'ictal entre la suma de les intensitats de l'interictal
print("Normalisation factor for voxels inside the mark: "+str(norm_factor_meanimask))
norm_factor_meani = data_ictal_in.sum()/data_interictal_in.sum()
print("Normalisation factor of whole image: "+str(norm_factor_meani))
# We observ that the ratios are different, probably meaning that the focus Epileptogenic focus determining this difference and that another approach should be implemented

# Option 2:
# Computing the same ratio but with the voxels outside the image, which do not contain the epileptogenic focus

data_interictal_outmask = data_interictal_in[(mask==False)] #agafem aquells valors de interictal NO inclosos dins la màscara
data_ictal_outmask = data_ictal_in[(mask==False)] #agafem aquells valors de ictal NO inclosos dins la màscara
norm_factor_meani_outmask = data_ictal_outmask.sum()/data_interictal_outmask.sum() #calculem el factor de normalització com la suma de les intensitats de l'ictal entre la suma de les intensitats de l'interictal
print("Normalisation factor for voxels outside the mask: "+str(norm_factor_meani_outmask))
#Other approaches are due to be studied before choosing the best option


#Option 3:
#Normalising with the quotient of the most frequent intensity of ictal and interictal masked images
max_interictal = float (data_interictal_masked[np.where(freqs_interictal_masked == max(freqs_interictal_masked))])
max_ictal = float (data_ictal_masked[np.where(freqs_ictal_masked == max(freqs_ictal_masked))])
normfactor = max_ictal/max_interictal
print('Normalisation factor of quotient of the most frequent intensity of ictal and interictal masked images:'+str(normfactor))



# %%
#OPTION 3 WAS CHOSEN

#Applying normalisation factor
data_norminterictal = data_interictal_in*mask*normfactor #apply the normalisation factor
fnamenorm = "generated_images/norminterictal.nii" #path
affinenorm = affine_interictal_in #the affine space of the image and the normalised is the same
image_normalized = nib.Nifti1Image(data_norminterictal, affinenorm) #Generating the image
image_normalized.set_data_dtype(np.float32) 
nib.save(image_normalized, fnamenorm) #save


#Import of interictal normalised image
interictal_norm, data_interictal_norm, dim_interictal_norm, header_interictal_norm, affine_interictal_norm = load_image(fnamenorm)

#Comparing histograms (normalised and template)
plt.hist(data_ictal_masked.flatten(), bins = 150, color='b')
plt.hist(data_norminterictal.flatten(), bins = 100, color = 'y')
plt.xlabel("Intensity")
plt.ylabel("Frequency (in counts)")
plt.xlim(20,180)
plt.ylim(0,75000)
plt.title("Ictal histogram vs interictal normalised histogram")
plt.legend({'Ictal':'blue','Normalised interictal':'yellow'})
plt.text(112, 55000,'Normalisation factor: ' + str(np.round(normfactor,2)))
plt.show()



# %%

#Imatge of the difference
#Generation of sustraction matrix from interictal normalized (mask was already applied) and ictal only with mask applied (as the normalisation is from interictal to ictal)
sus =  data_ictal_in*mask  - data_interictal_norm


sus_path = "generated_images/subtraction.nii" #file path of image
subtraction_image = nib.Nifti1Image(sus, affine_ictal_in) #craete image
nib.save(subtraction_image, sus_path) #save    


# Importing image and its data
subtraction, data_subtraction, dim_subtraction, header_subtraction, affine_subtraction = load_image(sus_path)



# %%
# ### Locating EZ


mean = data_subtraction.mean() #mean of diff Intensity
sd = data_subtraction.std() #sd of diff intensity

llindar = mean + 2*sd 
fusion = np.zeros([dim_subtraction[0], dim_subtraction[1], dim_subtraction[2]], dtype = 'float32') #matrius de 0s

fusion = data_subtraction*(data_subtraction>=llindar)

fusion_name = "generated_images/fusion.nii" #path
fusion_image = nib.Nifti1Image(fusion, affine_subtraction) #Create image
nib.save(fusion_image, fusion_name) #save

