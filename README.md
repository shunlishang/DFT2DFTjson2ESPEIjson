# DFT2DFTjson2ESPEIjson
Thermochemical properties from DFT data to DFT json files to ESPEI json files

Before start: 
0: You need a sublattice model for CALPHAD modeling later 

1: You need a "phase_name" file in each folder, includling the following contents (only the first line is important, others are comments)
C14_194
ENDMEMBER

2. You need a yaml file "sub_conf_occ.yaml" to describe elements and their compositions in each sublattice, such as:  
sub_configurations: ## List the elements in each sublattice, one indicates fully occupied, "list" is for interaction case 
  - Hf
  - [Cr, Hf]  
sub_occupancies:    ## mole fraction. If not a list, the code will not ignore it, so, you can put any value
  - 1
  - [0.25, 0.75]

After DFT calculations, use the following three code get DFTjson and ESPEIjson files
"01short_get_info_v1" is a Linux script to prepare input file, the output file is "vasp_info" to be read by python file
"02DFTdata2_JSON_v2.py" is a python code to convert DFT data to DFTjson file 
"0303JSON2_espei_JSON_v3.py" is a python code to convert DFTjson file to ESPEIjson file 
