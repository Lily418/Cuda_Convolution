To build simply run make

The binary takes 2 parameters

The first indicates which implementation to run which can be
0 - CPU Implementation
1 - Naive GPU
2 - Optimised GPU

The second indicates which file to read the input vector from.

e.g. to run the CPU implementation on the small sample
bin/release/smb 0 sample_small.txt

e.g. to run the Naive GPU version on the larger sample
bin/release/smb 1 sample.txt
