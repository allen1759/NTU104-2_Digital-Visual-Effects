# vlfeat-0.9.20

These instructions explain how to setup VLFeat in MATLAB (at least 2009B) using the binary distribution (it is also possible to compile the library and toolbox from source, including running on earlier MATLAB versions by disabling some features such as OpenMP support).


### One-time setup
Download and unpack the latest VLFeat binary distribution in a directory of your choice (e.g. ~/src/vlfeat). Let VLFEATROOT denote this directory. VLFeat must be added to MATLAB search path by running the vl_setup command found in the VLFEATROOT/toolbox directory. From MATLAB prompt enter

```
>> run('VLFEATROOT/toolbox/vl_setup')
VLFeat 0.9.17 ready.
```

To check that VLFeat is sucessfully installed, try to run the vl_version command:
```
>> vl_version verbose
VLFeat version 0.9.17
    Static config: X64, little_endian, GNU C 40201 LP64, POSIX_threads, SSE2, OpenMP
    4 CPU(s): GenuineIntel MMX SSE SSE2 SSE3 SSE41 SSE42
    OpenMP: max threads: 4 (library: 4)
    Debug: yes
    SIMD enabled: yes
```

### Getting started
All commands embed interface documentation that can be viewed with the builtin help command (e.g. help vl_sift).

VLFeat bundles a large number of demos. To use them, add the demo path with vl_setup demo. For example, a sift demo vl_demo_sift_basic can be run using the following:
```
>> vl_setup demo

>> vl_demo_sift_basic
```
To see a list of demos TAB-complete vl_demo at MATLAB prompt, after running vl_setup demo.