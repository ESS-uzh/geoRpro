The files in this directory are examples of workflows to be used

The indir (data input) is -> /home/diego/work/dev/data/test_data
The outdir (data output) is -> /home/diego/work/dev/data/test_data_out


- 1. driver_example_windows.json

#### RPropro ####
 Takes input sentinel-2 bands
 Define a commune resolution of 20 m
 Extract pixel values from all bands for a define window (AOI), e.g. [(0, 5), (0, 10)] # 5 first rows, 10 first columns

#### RMask ####
 This build a raster mask, i.e. a binary mask 0s and 1s
 Values -> specify the values that will be convert into 1s, other values will be converted to 0s
 
#### RReplace ####
 This uses the raster mask created in the previous step (scl_mask) to:
  - find target positions in input bands (they correspond to the 1s postion of the mask file)
  - replace whathever values is in the target postions with a value specified in Replace, i.e. 9999
 
 The input band in this case are:
 
 B02
 B04
 B08
 B12
 
 This step produces output bands:
 
 B02_masked
 B04_masked
 B08_masked
 B12_masked

#### Calculate ndvi and nbr indexes ####
 Calculate the indexes on the masked bands

### Stacks bands together ###
 Create rasters 



- driver_example_from_stack.json

### Extract a band from a stack ###
  The parameters Bands is a list that can accept one or more comma separate values (bands number)



- driver_example_normalize.json 

### Extract bands from a stack ###
  In this example we select the RGB bands of an hyperspectral cude (aviris-ng)

### Normalize ###
  This step takes the RGB from the previous step and normalize the values between 1 - 255
  setting no data values to 0. It produces an 8-bit georaster 
  




 
  