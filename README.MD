# SlicerMuscleMap: An Open-Source, Community-Supported, Wholebody Segmentation Model for Muscle in 3D Slicer.

The official documentation for **MuscleMap**, an open-source framework for automated **muscle segmentation** and quantitative analysis of **muscle volume** and **intramuscular fat infiltration**, can be found at https://musclemap.github.io/MuscleMap/.

![][MuscleMap_Slicer3D.png]

When citing MuscleMap, please cite the following publication:

> McKay MJ, Weber KA 2nd, Wesselink EO, Smith ZA, Abbott R, Anderson DB, Ashton-James CE, Atyeo J, Beach AJ, Burns J, Clarke S, Collins NJ, Coppieters MW, Cornwall J, Crawford RJ, De Martino E, Dunn AG, Eyles JP, Feng HJ, Fortin M, Franettovich Smith MM, Galloway G, Gandomkar Z, Glastras S, Henderson LA, Hides JA, Hiller CE, Hilmer SN, Hoggarth MA, Kim B, Lal N, LaPorta L, Magnussen JS, Maloney S, March L, Nackley AG, O'Leary SP, Peolsson A, Perraton Z, Pool-Goudzwaard AL, Schnitzler M, Seitz AL, Semciw AI, Sheard PW, Smith AC, Snodgrass SJ, Sullivan J, Tran V, Valentin S, Walton DM, Wishart LR, Elliott JM. MuscleMap: An Open-Source, Community-Supported Consortium for Whole-Body Quantitative MRI of Muscle. J Imaging. 2024;10(11):262. <https://doi.org/10.3390/jimaging10110262> 

> Wesselink EO, Elliott JM, Coppieters MW, Hancock MJ, Cronin B, Pool-Goudzwaard A, Weber II KA.Convolutional neural networks for the automatic segmentation of lumbar paraspinal muscles in people with low back pain. Sci Rep. 2022;12(1):13485. <https://doi.org/10.1038/s41598-022-16710-5>

##  Set up
If you have a GPU available and installed then whole-body segmentation can be computed in a few minutes. Therefore, it is recommended to set up GPU acceleration as described in this section. CUDA installation is not necessary when a suitable GPU is not available, but will   take more time. By default, the Slicer3D Pytorch module will be installed alongside the MuscleMap installation.

**To use a GPU,** you need a compatible NVIDIA GPU with [CUDA](https://developer.nvidia.com/cuda-toolkit) installed or a compatible AMD GPU with [ROCM](https://www.amd.com/en/products/software/rocm.html) installed. You then need to install the corresponding GPU-compatible version of [PyTorch v2.4.0](https://pytorch.org/get-started/previous-versions/). 

### Usage
- Start 3D slicer
- Go to MuscleMap module
- Install important MuscleMap depencies by clicking on the button in the main module.
- Load an imput image into the Slicer3D console. 
    - Example open-source to test the model can be found --for example-- at [MyoSegmentum thigh ](https://osf.io/svwa7/?view_only=c2c980c17b3a40fca35d088a3cdd83e2) or [MyoSegmentum Spine ](https://osf.io/3j54b/?view-only=f5089274d4a449cda2fef1d2df0ecc56). Dependent on the availability of a GPU and image size, segmentation may a few seconds up to a minute to segment the image. 
- When this module is used for the first time, please first download and install Pytorch and MuscleMap dependencies by clicking on the button ('Install MuscleMap dependencies'). This can take several minutes. 
    - You may get an error popup: Failed to compute results ... Command ... 'pip', 'install' ... returned non-zero exit status 1. This may be normal, see what to do in Troubleshooting section
- Select the model under *Inputs*:
    - **Model**: `wholebody` (default) or a regional model (`abdomen`, `pelvis`, `thigh`, `leg`, `forearm`). The list is read from the installed MuscleMap package.
    - **Model version**: `latest` (default) always uses the newest version published on Zenodo; a specific version can be selected for reproducibility.
    - The first run of a model downloads it from Zenodo and caches it; this can take a while.
- When dependencies are succesfully installed, click on the button ('Run MuscleMap Segmentation'). The run can be cancelled from the progress dialog.
    - Expected computation time:
        - With CUDA-capable GPU: 3 seconds to 1 minute, dependent on image resolution and size. 
        - Without GPU: 1 minute to 20 minutes, dependent on image resolution and size. 
- The resulting segmentation has the MuscleMap muscle names and colours.
- **Export segmentation...** (under *Outputs*) writes three files next to each other (the `_dseg` suffix is always added):
    - `<name>_dseg.nii.gz`: labelmap with the original MuscleMap label values (e.g. 1101 for levator scapulae left), not renumbered 1, 2, 3, ...
    - `<name>_dseg_labels.csv`: label value, region, anatomy and side.
    - `<name>_dseg.ctbl`: 3D Slicer color table with names and colours.
- *Advanced*: force CPU, lower sliding-window overlap (faster), and **Remember settings** to restore the selected model, version and options on the next start of Slicer.

### Version
The extension version is shown in the module help panel (SlicerMuscleMap version). Please mention it when reporting an issue. Releases are tagged in this repository (`v<version>`); the Slicer Extensions Manager builds from the `main` branch.

### Contact

Please post any questions to the [MuscleMap issue page](https://github.com/MuscleMap/MuscleMap/issues).

## Contributing

Contributions to this extensions are welcome. Please send a pull request with any suggested changes. 

[MuscleMap_Slicer3D.png]: MuscleMap_Slicer3D.png
