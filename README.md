# Basic Non-Local Means (NLM) Implementation in CUDA
Very rudimentary NLM[^1] implementation in CUDA with exposed Python bindings compatible with numpy arrays.

## Compile
```bash
nvcc --shared non_local_means.cu -o non_local_means.so -Xcompiler -fPIC
```

## References
[^1]: A. Buades, B. Coll and J. . -M. Morel, "A non-local algorithm for image denoising," 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), San Diego, CA, USA, 2005, pp. 60-65 vol. 2, doi: 10.1109/CVPR.2005.38.