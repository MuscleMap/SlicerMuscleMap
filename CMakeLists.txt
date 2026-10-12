cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)
project(MuscleMap)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_NAME "MuscleMap")
set(EXTENSION_HOMEPAGE "https://github.com/MuscleMap/SlicerMuscleMap")
set(EXTENSION_CONTRIBUTORS "Eddo Wesselink (Stanford University), Kenneth Arnold Weber II (Stanford University)")
set(EXTENSION_DESCRIPTION "A 3D Slicer extension for the MuscleMap muscle segmentation models (whole-body and regional).")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/MuscleMap/SlicerMuscleMap/main/MuscleMap.png")
set(EXTENSION_SCREENSHOTURLS
  "https://raw.githubusercontent.com/MuscleMap/SlicerMuscleMap/main/MuscleMap_Slicer3D.png")
set(EXTENSION_DEPENDS "PyTorch")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_STATUS "Experimental")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(WholeBodyMuscleSegmentation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
