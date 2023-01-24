# About data simulation
The codes in this folder are for simulating pulse waveforms.

## main.m
This code is a main code that runs the entire simulation process, and saves the simulated data. It defines inter-individualities and a range of severity level of train and test dataset, and calls a function "generate_massive_data" from "generate_massive_data.m".

## generate_massive_data.m
This code iterately simulates waveforms and integrates them into single object. It calls a function "generate_single_data" from "generated_single_data.m" when simulates an waveform. The main function "generate_massive_data" takes three input objects and returns two output objects
### input
- Inter_individuality: It defines inter_individuality value for l,r,E,h, and Rp.
- severity_level: It defines a disease severity level.
- sample_num
### output
- data
- label

## generate_single_data.m
This code is an essential code for simulation. The main function is "generate_single_data", and some important values are defined in this function. For example, "COV_" things are a CoV(Coefficient of Variation) for each variables related with an individuality.
The disease (Stenosis) is defined from line 322 to 326. In short, The cross-sectional area of 33rd blood vessel (line 322) becomes smaller as much as defined severity level (SL, line 324) by thickening vessel wall (line 325). By changing the blood vessel number(line 322), the stenosis can be re-located to other vessel.