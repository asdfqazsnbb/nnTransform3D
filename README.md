nnTransform3D (CUDA 12.X required)

Usage: nnTransform3D.exe input.tbc [activeVideoStart-2] [activeVideoEnd+2]

activeVideoEnd = activeVideoStart + activeVideoWidth

This will output two tbc files into the source directory: input_Y.tbc (luma) and input_Y_chroma.tbc (chroma).

Existing JSON or DB files from the input.tbc can be reused.

v2.0: Significant processing speed uplift at the cost of partial accuracy; model weights are not backward compatible with prior versions.




