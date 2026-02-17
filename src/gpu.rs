use cudarc::driver::{CudaContext, CudaFunction, CudaStream};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

pub struct GpuContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
}

impl GpuContext {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream })
    }

    /// Compile a CUDA kernel from a source file, returning the named function.
    ///
    /// If the source contains `#include "common.cuh"`, the contents of
    /// `kernels/common.cuh` are inlined before compilation.
    pub fn compile_kernel(
        &self,
        source_path: &str,
        function_name: &str,
    ) -> Result<CudaFunction, Box<dyn std::error::Error>> {
        let mut source = std::fs::read_to_string(source_path)?;

        // Manually inline #include "common.cuh"
        if source.contains("#include \"common.cuh\"") {
            let common = std::fs::read_to_string("kernels/common.cuh")?;
            source = source.replace("#include \"common.cuh\"", &common);
        }

        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            },
        )?;

        let module = self.ctx.load_module(ptx)?;
        let func = module.load_function(function_name)?;
        Ok(func)
    }

    /// Compile a CUDA source file once and return multiple named functions from
    /// the same module.  Avoids redundant NVRTC compilations when a single .cu
    /// file exports several entry points.
    pub fn compile_kernel_multi(
        &self,
        source_path: &str,
        function_names: &[&str],
    ) -> Result<Vec<CudaFunction>, Box<dyn std::error::Error>> {
        let mut source = std::fs::read_to_string(source_path)?;

        // Manually inline #include "common.cuh"
        if source.contains("#include \"common.cuh\"") {
            let common = std::fs::read_to_string("kernels/common.cuh")?;
            source = source.replace("#include \"common.cuh\"", &common);
        }

        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            },
        )?;

        let module = self.ctx.load_module(ptx)?;
        let mut funcs = Vec::with_capacity(function_names.len());
        for name in function_names {
            funcs.push(module.load_function(name)?);
        }
        Ok(funcs)
    }

    /// Allocate a zero-initialized device buffer of `len` elements.
    pub fn alloc_zeros<T: cudarc::driver::ValidAsZeroBits + cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>, cudarc::driver::DriverError> {
        self.stream.alloc_zeros::<T>(len)
    }

    /// Copy a host slice to a new device buffer.
    pub fn htod<T: cudarc::driver::DeviceRepr + Unpin>(
        &self,
        data: &[T],
    ) -> Result<cudarc::driver::CudaSlice<T>, cudarc::driver::DriverError> {
        self.stream.clone_htod(data)
    }

    /// Copy a device buffer back to host.
    pub fn dtoh<T: cudarc::driver::DeviceRepr + Unpin>(
        &self,
        buf: &cudarc::driver::CudaSlice<T>,
    ) -> Result<Vec<T>, cudarc::driver::DriverError> {
        self.stream.clone_dtoh(buf)
    }
}
