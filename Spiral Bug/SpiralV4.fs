// Basic reverse mode AD on the GPU. This v4 of Spiral is focused on removing the global state.
// Goto line 1165 and uncomment the gemm function to find the testing bug.

namespace SpiralV4

module Main =
    #if INTERACTIVE
    #r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/ManagedCuda.dll"
    #r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/NVRTC.dll"
    #r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/CudaBlas.dll"
    #r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/CudaRand.dll"
    #r "../packages/ManagedCuda-75-x64.7.5.7/lib/net45/x64/NPP.dll"
    #r "../packages/ManagedCuda-CudaDNN.5.0.1/lib/net45/CudaDNN.dll"
    #r "../packages/MathNet.Numerics.3.13.0/lib/net40/MathNet.Numerics.dll"
    #r "../packages/MathNet.Numerics.FSharp.3.13.0/lib/net40/MathNet.Numerics.FSharp.dll"
    #endif

    // Open up the namespaces.
    open ManagedCuda
    open ManagedCuda.BasicTypes
    open ManagedCuda.VectorTypes
    open ManagedCuda.CudaBlas
    open ManagedCuda.CudaRand
    open ManagedCuda.NVRTC
    open ManagedCuda.CudaDNN

    open System
    open System.Collections.Generic
    open System.Runtime.InteropServices

    // Initialize the context. Analogous to a CPU process. Cuda tries to offload as much as possible during context creation so there aren't
    // any unexpected delays later.
    let ctx = new CudaContext()
    let numSm = ctx.GetDeviceInfo().MultiProcessorCount // The number of streaming multiprocessors on the device.

    // Set the Cuda libraries handles to the above stream.
    let cublas = CudaBlas(PointerMode.Host,AtomicsMode.Allowed) // Better performance for some solver functions with atomics allowed. The Spiral library does not use them though.
    let cudnn = new CudaDNNContext()
    let cudaRandom = new CudaRand.CudaRandDevice(GeneratorType.PseudoDefault)

    type unit_to_unit_delegate = delegate of unit -> unit
    let add_callback_to_stream (str : CudaStream) (callback : unit -> unit) =
        let callb (str : CUstream) (res : CUResult) (p : nativeint) =
            let t : unit_to_unit_delegate = Runtime.InteropServices.Marshal.GetDelegateForFunctionPointer(p)
            t.Invoke()

        let aux = new unit_to_unit_delegate (callback)
        let ptr_to_aux = Marshal.GetFunctionPointerForDelegate aux

        let cuda_callback = CUstreamCallback(callb)
        str.AddCallback(cuda_callback,ptr_to_aux,CUStreamAddCallbackFlags.None)

    // Helper functions
    let inline dispose v =
        (^b: (member Dispose: unit -> unit) v)

    /// Copies a host array to device.
    let inline to_dev (host_ar: 't []) =
        let d_a = new CudaDeviceVariable<'t>(SizeT host_ar.Length)    
        d_a.CopyToDevice(host_ar)
        d_a

    /// Copies a device array to host.
    let inline to_host (dev_ar: CudaDeviceVariable<'t>) =
        let h_a = Array.zeroCreate<'t> (int dev_ar.Size)
        dev_ar.CopyToHost(h_a)
        ctx.Synchronize()
        h_a

    /// Copies the device array to host. Extends the CudaDeviceVariable class.
    type CudaDeviceVariable<'t when 't: struct and 't: (new: unit -> 't) and 't:> System.ValueType> with
        member inline this.Gather() =
            to_host this

    /// The float scalar type
    type Df = 
        {
        P : Lazy<float32> ref // primal
        A : float32 ref // adjoint
        }

        static member inline create P =
            {P=ref (lazy P);A=ref 0.0f}

    type DeadFlagType =
    | Undefined
    | Dead
    | Alive

    type AutoDiffType =
        | PrimalOnly of CudaDeviceVariable<float32> // Does no AD
        | PrimalAndAdjoint of CudaDeviceVariable<float32> // Does first order AD

        static member CreatePrimalOnly(size: int) =
            new CudaDeviceVariable<float32>(SizeT size)
            |> fun x -> x.Memset(0u); x // Zeroes the variable (saves me from having to zero out the adjoint)
            |> PrimalOnly

        static member CreatePA(size: int) =
            new CudaDeviceVariable<float32>(SizeT (size*2))
            |> fun x -> x.Memset(0u); x // Zeroes the variable (saves me from having to zero out the adjoint)
            |> PrimalAndAdjoint

        member private t.Resize(size: int) =
            match t with
            | PrimalOnly v | PrimalAndAdjoint v ->
                if size < int v.Size then v
                else
                    v.Dispose()
                    new CudaDeviceVariable<float32>(SizeT size)
        
        member t.ResizePrimalOnly(size: int) =
            t.Resize(size) |> PrimalOnly

        member t.ResizePrimalAndAdjoint(size: int) =
            t.Resize(size*2) |> PrimalAndAdjoint

        member t.P =
            match t with
            | PrimalOnly v -> v
            | PrimalAndAdjoint v ->
                let half = (int v.SizeInBytes) / 2 |> SizeT
                new CudaDeviceVariable<float32>(v.DevicePointer,false,half)

        member t.A =
            match t with
            | PrimalOnly v -> failwith "No adjoint!"
            | PrimalAndAdjoint v ->
                let half = (int v.SizeInBytes) / 2 |> SizeT
                new CudaDeviceVariable<float32>(v.DevicePointer + half,false,half)

        member t.PA = t.P, t.A

        member t.CopyToPrimal(host_ar: float32[]) =
            let x = t.P
            if int x.Size <> host_ar.Length then failwithf "int x.Size(%i) <> host_ar.Length(%i)" (int x.Size) (host_ar.Length)
            x.CopyToDevice(host_ar)

        member t.HasAdjoint =
            match t with
            | PrimalOnly v -> false
            | _ -> true

        member t.Dispose() =
            match t with
            | PrimalOnly v | PrimalAndAdjoint v -> v.Dispose()

    let defaultLayout = cudnnTensorFormat.NCHW
    let defaultType = cudnnDataType.Float
    let defaultMaxPoolingNanOption = cudnnNanPropagation.PropagateNan
    let defaultReluNanOption = cudnnNanPropagation.PropagateNan

    type TensorDescriptor with
        /// Extended method that works according to the bound defaultLayout and defaultType variables.
        member inline t.SetTensor4dDescriptor(n,c,h,w) = t.SetTensor4dDescriptor(defaultLayout,defaultType,n,c,h,w)

    type FilterDescriptor with
        /// Extended method that works according to the bound defaultType variable.
        member inline t.SetFilter4dDescriptor(n,c,h,w) = t.SetFilter4dDescriptor(defaultType,defaultLayout,n,c,h,w)

    type ConvolutionParameters = {
        pad_h : int
        pad_w : int
        stride_h : int
        stride_w : int
        upscale_h : int
        upscale_w : int
        mode : cudnnConvolutionMode
        }

    type PoolingParameters =
        {
        mode : cudnnPoolingMode
        windowHeight : int
        windowWidth : int
        verticalPadding : int
        horizontalPadding : int
        verticalStride : int
        horizontalStride : int
        }

    type PoolingDescriptor with
        member inline t.SetPooling2dDescriptor (p : PoolingParameters) =
            t.SetPooling2dDescriptor(p.mode,defaultMaxPoolingNanOption,p.windowHeight,p.windowWidth,p.verticalPadding,p.horizontalPadding,p.verticalStride,p.horizontalStride)

        member inline t.GetPooling2dForwardOutputDim s =
            let mutable n,c,h,w = 0,0,0,0
            t.GetPooling2dForwardOutputDim(s,&n,&c,&h,&w)
            n,c,h,w

    let defaultConvPar = 
        {
        pad_h = 0
        pad_w = 0
        stride_h = 1
        stride_w = 1
        upscale_h = 1
        upscale_w = 1
        mode = cudnnConvolutionMode.Convolution
        }

    type ConvolutionDescriptor with
        member inline t.SetConvolution2dDescriptor (p : ConvolutionParameters) =
            t.SetConvolution2dDescriptor(p.pad_h,p.pad_w,p.stride_h,p.stride_w,p.upscale_h,p.upscale_w,p.mode, defaultType)
        member inline t.GetConvolution2dForwardOutputDim (s,f) =
            let mutable n,c,h,w = 0,0,0,0
            t.GetConvolution2dForwardOutputDim(s,f,&n,&c,&h,&w)
            n,c,h,w

    type Workspace() = 
        let mutable workspace: CudaDeviceVariable<byte> = CudaDeviceVariable.Null

        /// Resizes the workspace if it is less than size and returns it. The size is in bytes.
        member t.ResizeIf(size: int) =
            if size < int workspace.Size then workspace
            else
                workspace.Dispose()
                workspace <- new CudaDeviceVariable<byte>(SizeT size)
                workspace

        /// Resizes the workspace if it is less than size and returns it. The size is in float32s.
        member t.ResizeIfF32(size: int) =
            let toF32(workspace: CudaDeviceVariable<byte>) = new CudaDeviceVariable<float32>(workspace.DevicePointer,false)
            if size < int workspace.Size then toF32 workspace
            else
                workspace.Dispose()
                workspace <- new CudaDeviceVariable<byte>(SizeT (size * sizeof<float32>))
                toF32 workspace

        member __.Dispose() = workspace.Dispose()
        interface IDisposable with
            member t.Dispose() = t.Dispose()

    and d2M =
        {
        mutable rc : int * int
        mutable diff: AutoDiffType
        mutable is_dead : DeadFlagType // flag to skip backprop
        }

        static member private size_rc (row,col) = row*col

        /// Add the rows and column of the 2d matrix.
        member t.AddDims = t.rc |> fun (r,c) -> r+c
    
        static member create' (size : (int * int), is_constant) =
            let diff = 
                let s = d2M.size_rc size
                match is_constant with
                | true -> AutoDiffType.CreatePrimalOnly s
                | false -> AutoDiffType.CreatePA s
            {rc = size; diff=diff; is_dead=Undefined}

        static member create' (size : (int * int), host_data : float32[], is_constant) =
            let t = d2M.create' (size, is_constant)
            t.diff.CopyToPrimal(host_data)
            t

        // Constructors for the singular d2M records.
        static member inline create (ar : int * int) = d2M.create'(ar, false)
        static member inline create (row : int, col : int) = d2M.create'((row, col), false)
        static member inline create (row : int, col : int, ar_data : float32[]) = d2M.create'((row,col),ar_data, false)
        static member inline create (size : int * int, ar_data : float32[]) = d2M.create'(size,ar_data, false)
        static member inline createConstant (size : int * int) = d2M.create'(size, true)
        static member inline createConstant (row : int, col : int, ar_data : float32[]) = d2M.create'((row,col),ar_data, true)
        static member inline createConstant (size : int * int, ar_data : float32[]) = d2M.create'(size,ar_data, true)

        /// Number of rows
        member t.Rows = t.rc |> fst
        /// Number of columns
        member t.Columns = t.rc |> snd  
        /// Returns whether the function has an adjoint
        member t.HasAdjoint = t.diff.HasAdjoint
  
        /// Returns the size of matrix
        member t.Size = d2M.size_rc t.rc

        /// Get Adjoint Pointer
        member t.GAP = t.diff.A.DevicePointer

        /// Get Primal Pointer
        member t.GPP = t.diff.P.DevicePointer

        /// Get Adjoint Variable
        member t.GAV = t.diff.A

        /// Get Primal Variable
        member t.GPV = t.diff.P

        /// Get row and column
        member t.RC = t.rc
    
        /// Get the deadness flag
        member t.IsDead = t.is_dead

        /// Update the deadness flag
        member t.DeadIs v = t.is_dead <- v

        /// CUDNN has a bug where it is ridicously slow if the dimensions are not set up right.
        /// So this function is to get nchw of the matrix for the tensor_add function.
        member t.NCHW =
            (t.Columns,1,t.Rows,1)
            // (t.Columns,t.Rows,1,1) is 10x slower than the above
            // There are other fast configurations, but it is unfortunate that I picked the
            // Wrong one for SpiralV3. Now that I know duck typing, writing generic code will be
            // much easier.

        /// Returns the nchw (for the buggy tensor_add function)
        /// The stupid cuDNN function will throw an exception if I put in the parameters for the fast version.
        member t.NCHWBiasAdd = (t.Columns,t.Rows,1,1)

        /// Throws an exception if the sizes are not all equal
        static member GuardSizes(x:d2M, o: d2M) =
            if x.rc <> o.rc then failwithf "x.rc(%A) <> o.rc(%A)" x.rc o.rc

        /// Throws an exception if the sizes are not all equal
        static member GuardSizes(x:d2M, y:d2M, o: d2M) =
            if x.rc <> y.rc then failwithf "x.rc(%A) <> y.rc(%A)" x.rc y.rc
            if y.rc <> o.rc then failwithf "y.rc(%A) <> o.rc(%A)" y.rc o.rc

        /// Throws an exception if the sizes are not all equal
        static member GuardSizes(x:d2M, y:d2M, z: d2M, o: d2M) =
            if x.rc <> y.rc then failwithf "x.rc(%A) <> y.rc(%A)" x.rc y.rc
            if y.rc <> z.rc then failwithf "y.rc(%A) <> z.rc(%A)" y.rc z.rc
            if z.rc <> o.rc then failwithf "z.rc(%A) <> o.rc(%A)" z.rc o.rc

        /// Resizes the object. Does not free memory when resizing downwards.
        member t.ReplaceIf (ar : int * int, is_constant) =
            t.rc <- ar
            let new_size = d2M.size_rc ar
            match is_constant with
            | true -> t.diff <- t.diff.ResizePrimalOnly new_size
            | false -> t.diff <- t.diff.ResizePrimalAndAdjoint new_size

        /// Gets an object the same size as self from the object pool
        member inline t.GetFromObjectPool(obj_pool: ObjectPool, is_constant, is_inference_only, str: CudaStream) =
            obj_pool.Getd2M(is_constant,t.rc,is_inference_only,str)

        /// Copies the object by using the memory from the object pool.
        member inline t.CopyUsingObjectPool(obj_pool: ObjectPool, is_constant, is_inference_only, str: CudaStream) =
            let x = obj_pool.Getd2M(is_constant,t.rc,is_inference_only,str)
            x.GPV.AsyncCopyToDevice(t.GPV,str.Stream)
            x

        member t.Dispose() = t.diff.Dispose()

        /// Sets the adjoint to a value.
        member inline t.SetAdjoint(x: float32, str: CudaStream) = 
            let v = BitConverter.ToUInt32(BitConverter.GetBytes(x),0)
            t.diff.A.MemsetAsync(v,str.Stream)

        /// Set the matrix to a value.
        member inline t.SetPrimal (x: float32, str: CudaStream) = 
            let v = BitConverter.ToUInt32(BitConverter.GetBytes(x),0)
            t.diff.P.MemsetAsync(v,str.Stream)

        /// For temporary immediate use only.
        member t.ConvertdMLikeFromWorkspace(w: Workspace) =
            let v = w.ResizeIfF32 t.Size
            {rc = t.rc; diff = PrimalOnly v; is_dead = Undefined}

        interface IDisposable with
            member t.Dispose() = t.Dispose()

    and d4M =
        {
        mutable nchw : int * int * int * int
        mutable diff : AutoDiffType
        mutable is_dead : DeadFlagType // flag to skip backprop
        }

        /// Adds the image,channel, width and height dimensions of the 4d tensor.
        member t.AddDims = t.nchw |> fun (n,c,h,w) -> n+c+h+w

        static member private size_nchw (n:int,c,h,w) = n*c*h*w

        static member create' (size : (int * int * int * int), is_constant) =
            let diff = 
                let s = d4M.size_nchw size
                match is_constant with
                | true -> AutoDiffType.CreatePrimalOnly s
                | false -> AutoDiffType.CreatePA s
            {nchw = size; diff=diff ; is_dead=Undefined}

        static member create' (size : int * int * int * int, host_data : float32[], is_constant) =
            let t = d4M.create' (size, is_constant)
            t.diff.CopyToPrimal(host_data)
            t

        // Constructors for the singular d4M records.
        static member inline create (ar: int * int * int * int) = d4M.create'(ar, false)
        static member inline create (ar: (int * int * int * int), data: float32[]) = d4M.create'(ar, data, false)
        static member inline createConstant (ar : int * int * int * int) = d4M.create'(ar, true)
        static member inline createConstant (ar: (int * int * int * int), data: float32[]) = d4M.create'(ar, data, true)

        /// Number of rows (concatenates along the c,h,w dimensions)
        member t.Rows = t.nchw |> fun (_,c,h,w) -> c*h*w
        /// Number of columns (return the outer n dimension)
        member t.Columns = t.nchw |> fun (n,_,_,_) -> n
        /// Returns whether the function has an adjoint
        member t.HasAdjoint = t.diff.HasAdjoint

        /// Returns the size of matrix
        member t.Size = d4M.size_nchw t.nchw

        /// Get Adjoint Pointer
        member t.GAP = t.diff.A.DevicePointer

        /// Get Primal Pointer
        member t.GPP = t.diff.P.DevicePointer

        /// Get Adjoint Variable
        member t.GAV = t.diff.A

        /// Get Primal Variable
        member t.GPV = t.diff.P

        /// Returns the tensor's dimensions projected 
        /// to a 2D space according to the following formula:
        /// row = c*h*w
        /// column = n
        member t.RC = t.Rows, t.Columns
    
        /// Get the deadness flag
        member t.IsDead = t.is_dead

        /// Update the deadness flag
        member t.DeadIs v = t.is_dead <- v

        /// Returns the nchw
        member t.NCHW = t.nchw

        /// Returns the nchw (for the buggy tensor_add function)
        member t.NCHWBiasAdd = t.nchw

        /// Throws an exception if the sizes are not all equal
        static member inline GuardSizes(x:d4M, o: d4M) =
            if x.nchw <> o.nchw then failwithf "x.rc(%A) <> o.rc(%A)" x.nchw o.nchw

        /// Throws an exception if the sizes are not all equal
        static member inline GuardSizes(x:d4M, y:d4M, o: d4M) =
            if x.nchw <> y.nchw then failwithf "x.rc(%A) <> y.rc(%A)" x.nchw y.nchw
            if y.nchw <> o.nchw then failwithf "y.rc(%A) <> o.rc(%A)" y.nchw o.nchw

        /// Throws an exception if the sizes are not all equal
        static member inline GuardSizes(x:d4M, y:d4M, z: d4M, o: d4M) =
            if x.nchw <> y.nchw then failwithf "x.rc(%A) <> y.rc(%A)" x.nchw y.nchw
            if y.nchw <> z.nchw then failwithf "y.rc(%A) <> z.rc(%A)" y.nchw z.nchw
            if z.nchw <> o.nchw then failwithf "z.rc(%A) <> o.rc(%A)" z.nchw o.nchw

        /// Resizes the object. Does not free memory when resizing downwards.
        member t.ReplaceIf (ar : (int * int * int * int), is_constant) =
            t.nchw <- ar
            let new_size = d4M.size_nchw ar
            match is_constant with
            | true -> t.diff <- t.diff.ResizePrimalOnly new_size
            | false -> t.diff <- t.diff.ResizePrimalAndAdjoint new_size

        /// Gets an object the same size as it from the object pool
        member inline t.GetFromObjectPool(obj_pool: ObjectPool, is_constant, is_inference_only, str: CudaStream) =
            obj_pool.Getd4M(is_constant,t.nchw,is_inference_only,str)

        /// Copies the object by using the memory from the object pool.
        member inline t.CopyUsingObjectPool(obj_pool: ObjectPool, is_constant, is_inference_only, str: CudaStream) =
            let x = obj_pool.Getd4M(is_constant,t.nchw,is_inference_only,str)
            x.GPV.AsyncCopyToDevice(t.GPV,str.Stream)
            x

        member t.Dispose() = t.diff.Dispose()

        /// Sets the adjoint to a value.
        member inline t.SetAdjoint(x: float32, str: CudaStream) = 
            let v = BitConverter.ToUInt32(BitConverter.GetBytes(x),0)
            t.diff.A.MemsetAsync(v,str.Stream)

        /// Set the matrix to a value.
        member inline t.SetPrimal (x: float32, str: CudaStream) = 
            let v = BitConverter.ToUInt32(BitConverter.GetBytes(x),0)
            t.diff.P.MemsetAsync(v,str.Stream)

        /// For temporary immediate use only.
        member t.ConvertdMLikeFromWorkspace(w: Workspace) =
            let v = w.ResizeIfF32 t.Size
            {nchw = t.nchw; diff = PrimalOnly v; is_dead = Undefined}

        interface IDisposable with
            member t.Dispose() = t.Dispose()

    /// The new object pool. Zeroes out the adjoints concurrently on the forward phase.
    and ObjectPool() =
        let d2MPool = ResizeArray()
        let d2Mp = ref 0
        let d4MPool = ResizeArray()
        let d4Mp = ref 0

        let tensorDescriptorPool = Dictionary(HashIdentity.Structural)
        let filterDescriptorPool = Dictionary(HashIdentity.Structural)
        let convolutionDescriptorPool = Dictionary(HashIdentity.Structural)
        let poolingDescriptorPool = Dictionary(HashIdentity.Structural)
        let activationDescriptorPool = Dictionary(HashIdentity.Structural)
        let BNDescriptorPool = Dictionary(HashIdentity.Structural)

        static member inline private GetFromPool (pool : ResizeArray<_>) (pointer_to_pool : int ref) (creation_function) =
            if pool.Count > !pointer_to_pool then
                let t = pool.[!pointer_to_pool]
                pointer_to_pool := !pointer_to_pool+1
                t
            else
                let t = creation_function()
                pool.Add(t)
                pointer_to_pool := !pointer_to_pool+1
                t

        static member inline private GetFromDict (pool : Dictionary<_,_>) k creation_function set_function =
            match pool.TryGetValue k with
            | true, v -> v
            | false, _ ->
                let t = creation_function()
                set_function t k
                pool.Add(k, t)
                t

        member t.Getd2M (is_constant, (rc : (int*int)), is_inference_only, str: CudaStream): d2M =
            let t' = ObjectPool.GetFromPool d2MPool d2Mp (fun _ -> d2M.create'(rc,is_constant))
            t'.ReplaceIf(rc,is_constant)
            t'.is_dead <- Undefined
            if is_inference_only = false && t'.HasAdjoint then t'.diff.A.MemsetAsync(0u,str.Stream)
            t'

        member t.Getd4M (is_constant, (nchw : (int*int*int*int)), is_inference_only, str: CudaStream): d4M =
            let t' = ObjectPool.GetFromPool d4MPool d4Mp (fun _ -> d4M.create'(nchw,is_constant))

            t'.ReplaceIf(nchw,is_constant)
            t'.is_dead <- Undefined
            if is_inference_only = false && t'.HasAdjoint then t'.diff.A.MemsetAsync(0u,str.Stream)
            t'

        member t.GetTensorDescriptor (nchw : int*int*int*int) = 
            ObjectPool.GetFromDict tensorDescriptorPool nchw (fun _ -> new TensorDescriptor()) (fun (t: TensorDescriptor) x -> x |> t.SetTensor4dDescriptor)
        member t.GetFilterDescriptor (nchw : int*int*int*int) = 
            ObjectPool.GetFromDict filterDescriptorPool nchw (fun _ -> new FilterDescriptor()) (fun (t: FilterDescriptor) x -> x |> t.SetFilter4dDescriptor)
        member t.GetConvolutionDescriptor (convPars : ConvolutionParameters) = 
            ObjectPool.GetFromDict convolutionDescriptorPool convPars (fun _ -> new ConvolutionDescriptor()) (fun (t: ConvolutionDescriptor) x -> x |> t.SetConvolution2dDescriptor)
        member t.GetPoolingDescriptor (p : PoolingParameters) = 
            ObjectPool.GetFromDict poolingDescriptorPool p (fun _ -> new PoolingDescriptor()) (fun (t: PoolingDescriptor) x -> x |> t.SetPooling2dDescriptor)
        member t.GetActivationDescriptor (mode : cudnnActivationMode, nanopt : cudnnNanPropagation, reluCeiling as p) = 
            ObjectPool.GetFromDict activationDescriptorPool p (fun _ -> new ActivationDescriptor()) (fun (t: ActivationDescriptor) x -> x |> t.SetActivationDescriptor)
        member t.GetBNDescriptor ((nchw : int*int*int*int, mode : cudnnBatchNormMode, srcDesc : TensorDescriptor) as p) = 
            ObjectPool.GetFromDict BNDescriptorPool p 
                (fun _ -> new TensorDescriptor()) 
                (fun (t: TensorDescriptor) (nchw, mode, srcDesc) -> cudnn.DeriveBNTensorDescriptor(t,srcDesc,mode))

        // Resets the pointer in the object pool
        member t.Reset() =
            d2Mp := 0
            d4Mp := 0

        member __.Dispose() =
            let inline dis' ex pool = 
                // The disposer helper. Uses an extractor for the dictionary values.
                // This succintly demonstrates how to get around the lack of higher kinded types in F#.
                let pool = ex pool
                for x in pool do dispose x
            let inline dis x = dis' id x
            dis d2MPool
            dis d4MPool

            let inline dis x = 
                dis' (fun v -> (^a: (member Values: ^b) v)) x
            dis tensorDescriptorPool // ...It would have been faster to just copy paste .Value everywhere.
            dis filterDescriptorPool
            dis convolutionDescriptorPool
            dis poolingDescriptorPool
            dis activationDescriptorPool
            dis BNDescriptorPool

        interface IDisposable with
            member t.Dispose() = t.Dispose()

    let T = Operation.Transpose
    let nT = Operation.NonTranspose

    let inline extract_primal x = (^a : (member GPP: CUdeviceptr) x)
    let inline extract_adjoint x = (^a : (member GAP: CUdeviceptr) x)
    let inline extract_primal' x = (^a: (member GPV: CudaDeviceVariable<float32>) x)
    let inline extract_adjoint' x = (^a: (member GAV: CudaDeviceVariable<float32>) x)
    let inline rc x = (^a: (member RC: int * int) x)

    let inline GuardSizes2(x,y) =
        (^a: (static member GuardSizes: ^a * ^a -> unit) (x,y))
    let inline GuardSizes3(x,y,z) =
        (^a: (static member GuardSizes: ^a * ^a * ^a -> unit) (x,y,z))
    let inline GuardSizes4(x,y,z,o) =
        (^a: (static member GuardSizes: ^a * ^a * ^a * ^a -> unit) (x,y,z,o))
    let inline Size x =
        (^a : (member Size: int) x)

    let inline P x = (extract_primal, x)
    let inline A x = (extract_adjoint, x)
    let inline P' x = (extract_primal', x)
    let inline A' x = (extract_adjoint', x)

    let inline rows x = (^a : (member Rows: int) x)
    let inline cols x = (^a : (member Columns: int) x)

    let inline setPrimal x (v,str) = (^a : (member SetPrimal: float32 * CudaStream -> unit) x, v, str)
    let inline setAdjoint x (v,str) = (^a : (member SetAdjoint: float32 * CudaStream -> unit) x, v, str)

    let inline isDead x = (^a : (member IsDead: DeadFlagType) x)
    let inline deadIs x v = (^a : (member DeadIs: DeadFlagType -> unit) x, v)

    let inline hasAdjoint x = (^a : (member HasAdjoint: bool) x)

    let inline nchw x =
        (^a: (member NCHW: int * int * int * int) x)

    /// Helper for the buggy add_tensor function.
    let inline nchwBiasAdd x =
        (^a: (member NCHWBiasAdd: int * int * int * int) x)

    let inline convertdMLikeFromWorkspace (x: ^a) (w: Workspace) =
        (^a: (member ConvertdMLikeFromWorkspace: Workspace -> ^a) x, w)

    let inline addDims x =
        (^a: (member AddDims: int) x)

    let inline divup a b = (a-1)/b+1 // Integer division with rounding up. (a+b-1)/b is another variant on this.

    let kernels_dir = IO.Path.Combine(__SOURCE_DIRECTORY__,"Cuda Kernels")
    IO.Directory.CreateDirectory(kernels_dir) |> ignore // Creates the Cuda Kernels directory if it does not exist. WriteAllBytes would otherwise throw an exception.

    let load_kernel kernel_code kernel_name = 
        let kernel_path = IO.Path.Combine(kernels_dir,kernel_name)
        
        if IO.File.Exists(kernel_path) 
        then
            ctx.LoadKernelPTX(kernel_path,kernel_name) // For all the modules, it takes roughly 0.35s to compile them. Loading them from drive takes less than a millisecond.
        else
            let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,kernel_name)
            try k.Compile([|"-arch=compute_30"|])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()
            let ptx = k.GetPTX()
            IO.File.WriteAllBytes(kernel_path,ptx)
            ctx.LoadKernelPTX(ptx,kernel_name)

    let inline map_launcher(str: CudaStream, kernel: CudaKernel, total_size: int, [<ParamArray>] args: obj[]) =
        let block_size = 256
        let gridSize = min (2*numSm*(1024/block_size)) (divup total_size block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, args)

    /// o <- f(x)
    type DeviceUnaryTransformModule(op: string, unique_name : string) = 
        let kernel_name = "Map1Kernel"+unique_name
        let kernel_code = 
            [|"
            //Kernel code:
            extern \"C\" {
                typedef float floatType;
                __device__ inline floatType op(floatType x)
                {
                    return ";op;"
                }
        
                // Device code
                __global__ void ";kernel_name;"(const floatType* A, floatType* O, const int N)
                {
                    int i = blockDim.x * blockIdx.x + threadIdx.x;
                    const int stride = blockDim.x * gridDim.x;
                    while (i < N)
                    {
                        O[i] = op(A[i]);
                        i += stride;
                    }
                }
            }

            " |] |> String.concat ""

        let kernel = load_kernel kernel_code kernel_name

        member t.Kernel = kernel
        member inline t.A
                (str: CudaStream,
                 (ext_x: ^a -> CUdeviceptr, x: ^a),
                 (ext_o: ^a -> CUdeviceptr, o: ^a)) =
            GuardSizes2(x,o)
            let n = Size x
            map_launcher(str, t.Kernel, n, [|ext_x x; ext_o o; n|])

    /// o <- f(x,y)
    type DeviceBinaryTransformModule(op: string, unique_name) = 
        let kernel_name = "Map2Kernel" + unique_name
        let kernel_code = 
            [|"
            //Kernel code:
            extern \"C\" {
                typedef float floatType;
                __device__ inline floatType op(floatType x, floatType y)
                {
                    return ";op;"
                }
        
                // Device code
                __global__ void ";kernel_name;"(const floatType* A, const floatType* B, floatType* O, const int N)
                {
                    int i = blockDim.x * blockIdx.x + threadIdx.x;
                    const int stride = blockDim.x * gridDim.x;
                    while (i < N)
                    {
                        O[i] = op(A[i],B[i]);
                        i += stride;
                    }
                }
            }

            " |] |> String.concat ""
    
        let kernel = load_kernel kernel_code kernel_name

        member t.Kernel = kernel
        member inline t.A
                (str: CudaStream,
                 (ext_x: ^a -> CUdeviceptr, x: ^a),
                 (ext_y: ^a -> CUdeviceptr, y: ^a),
                 (ext_o: ^a -> CUdeviceptr, o: ^a)) =
            GuardSizes3(x,y,o)
            let n = Size x
            map_launcher(str, t.Kernel, n, [|ext_x x; ext_y y; ext_o o; n|])

    /// o <- f(x,y,z)
    type DeviceTrinaryTransformModule(op: string, unique_name) = 
        let kernel_name = "Map3Kernel" + unique_name
        let kernel_code = 
            [|"
            //Kernel code:
            extern \"C\" {
                typedef float floatType;
                __device__ inline floatType op(floatType x, floatType y, floatType z)
                {
                    return ";op;"
                }
        
                // Device code
                __global__ void ";kernel_name;"(const floatType* A, const floatType* B, const floatType* C, floatType* O, const int N)
                {
                    int i = blockDim.x * blockIdx.x + threadIdx.x;
                    const int stride = blockDim.x * gridDim.x;
                    while (i < N)
                    {
                        O[i] = op(A[i],B[i],C[i]);
                        i += stride;
                    }
                }
            }

            " |] |> String.concat ""

        let kernel = load_kernel kernel_code kernel_name

        member t.Kernel = kernel
        member inline t.A
                (str: CudaStream,
                 (ext_x: ^a -> CUdeviceptr, x: ^a),
                 (ext_y: ^a -> CUdeviceptr, y: ^a),
                 (ext_z: ^a -> CUdeviceptr, z: ^a),
                 (ext_o: ^a -> CUdeviceptr, o: ^a)) =
            GuardSizes4(x,y,z,o)
            let n = Size x
            map_launcher(str, t.Kernel, n, [|ext_x x; ext_y y; ext_z z; ext_o o; n|])

    let inline map_sum_launcher(str: CudaStream, kernel: CudaKernel, total_size: int, o: CudaDeviceVariable<float32>, [<ParamArray>] args: obj[]) =
        let block_size = 256
        let gridSize = min (2*numSm*(1024/block_size)) (divup total_size block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)

        kernel.RunAsync(str.Stream, args)
        lazy o.[SizeT 0]

    /// o <- sum(f(x))
    type DeviceUnaryMapSumModule(op: string, unique_name) = 
        let kernel_name = "Map1SumKernel" + unique_name
        let kernel_code = 
            [|"
            //Kernel code:
            extern \"C\" {
                typedef float floatType;
                __device__ inline floatType op(floatType x)
                {
                    return ";op;"
                }
        
                __device__ inline floatType warpDownReduce(floatType value){
                    #pragma unroll
	                for (int i = 16; i>0; i = i / 2) value += __shfl_down(value, i);
	                return value;
                }

                // Device code
                __global__ void ";kernel_name;"(const floatType* A, floatType* O, const int N)
                {
	                int i = blockDim.x * blockIdx.x + threadIdx.x;
	                const int stride = blockDim.x * gridDim.x;
	                __shared__ floatType temp[32];
                    if (threadIdx.x < 32) {
                        temp[threadIdx.x] = 0.0f; 
                        if (blockIdx.x == 0) O[0] = 0.0f;
                        }
                
                    floatType acc = 0.0f;
	                while (i < N)
	                {
		                acc += op(A[i]);
		                i += stride;
	                }
	                __syncthreads(); 
                    floatType out_partial = warpDownReduce(acc);
	                if (threadIdx.x % 32 == 0) temp[threadIdx.x / 32] = out_partial;
	                __syncthreads();
	                if (threadIdx.x < 32) out_partial = warpDownReduce(temp[threadIdx.x]);
	                if (threadIdx.x == 0) atomicAdd(O, out_partial);
                }
            }

            " |] |> String.concat ""

        let kernel = load_kernel kernel_code kernel_name

        let o = new CudaDeviceVariable<float32>(SizeT 1)

        member t.Kernel = kernel
        member t.O = o
        member inline t.A
                (str: CudaStream,
                 (ext_x: ^a -> CUdeviceptr, x: ^a)) =
            let n = Size x
            map_sum_launcher(str, t.Kernel, n, t.O, [|ext_x x; t.O.DevicePointer; n|])

    /// o <- sum(f(x,y))
    type DeviceBinaryMapSumModule(op: string, unique_name) = 
        let kernel_name = "Map2SumKernel" + unique_name
        let kernel_code = 
            [|"
            //Kernel code:
            extern \"C\" {
                typedef float floatType;
                __device__ inline floatType op(floatType x, floatType y)
                {
                    return ";op;"
                }
        
                __device__ inline floatType warpDownReduce(floatType value){
                    #pragma unroll
	                for (int i = 16; i>0; i = i / 2) value += __shfl_down(value, i);
	                return value;
                }

                // Device code
                __global__ void ";kernel_name;"(const floatType* A, const floatType* B, floatType* O, const int N)
                {
	                int i = blockDim.x * blockIdx.x + threadIdx.x;
	                const int stride = blockDim.x * gridDim.x;
	                __shared__ floatType temp[32]; 
                    if (threadIdx.x < 32) {
                        temp[threadIdx.x] = 0.0f; 
                        if (blockIdx.x == 0) O[0] = 0.0f;
                        }    
                    floatType acc = 0.0f;
	                while (i < N)
	                {
		                acc += op(A[i],B[i]);
		                i += stride;
	                }
	                __syncthreads(); 
                    floatType out_partial = warpDownReduce(acc);
	                if (threadIdx.x % 32 == 0) temp[threadIdx.x / 32] = out_partial;
	                __syncthreads();
	                if (threadIdx.x < 32) out_partial = warpDownReduce(temp[threadIdx.x]);
	                if (threadIdx.x == 0) atomicAdd(O, out_partial);
                }
            }

            " |] |> String.concat ""

        let kernel = load_kernel kernel_code kernel_name

        let o = new CudaDeviceVariable<float32>(SizeT 1)

        member t.Kernel = kernel
        member t.O = o
        member inline t.A
                (str: CudaStream,
                 (ext_x: ^a -> CUdeviceptr, x: ^a),
                 (ext_y: ^a -> CUdeviceptr, y: ^a)) =
            GuardSizes2(x,y)
            let n = Size x
            map_sum_launcher(str, t.Kernel, n, t.O, [|ext_x x; ext_y y; t.O.DevicePointer; n|])

    /// o <- f(coef_x,x)
    type DeviceUnaryCoefTransformModule(op: string, unique_name) = 
        let block_size = 256

        let kernel_name = "Map1CoefKernel" + unique_name
        let kernel_code = 
            [|"
            //Kernel code:
            extern \"C\" {
                typedef float floatType;
                __device__ inline floatType op(floatType coef_x, floatType x)
                {
                    return ";op;"
                }
        
                // Device code
                __global__ void ";kernel_name;"(const floatType coef_A, const floatType* A, floatType* O, const int N)
                {
                    int i = blockDim.x * blockIdx.x + threadIdx.x;
                    const int stride = blockDim.x * gridDim.x;
                    while (i < N)
                    {
                        O[i] = op(coef_A,A[i]);
                        i += stride;
                    }
                }
            }

            " |] |> String.concat ""

        let kernel = load_kernel kernel_code kernel_name

        member t.Kernel = kernel
        member inline t.A
                (str: CudaStream,
                 coef_x: float32, (ext_x: ^a -> CUdeviceptr, x: ^a), 
                                  (ext_o: ^a -> CUdeviceptr, o: ^a)) =
            GuardSizes2(x,o)
            let n = Size x
            map_launcher(str, t.Kernel, n, [|coef_x; ext_x x; ext_o o; n|])


    /// o <- f(coef_x,x,coef_y,y)
    type DeviceBinaryCoefTransformModule(op: string, unique_name) = 
        let block_size = 256

        let kernel_name = "Map2CoefKernel" + unique_name
        let kernel_code = 
            [|"
            //Kernel code:
            extern \"C\" {
                typedef float floatType;

                __device__ inline floatType op(floatType coef_x, floatType x, floatType coef_y, floatType y)
                {
                    return ";op;"
                }
        
                // Device code
                __global__ void ";kernel_name;"(const floatType coef_A, const floatType* A, const floatType coef_B, const floatType* B, floatType* O, const int N)
                {
                    int i = blockDim.x * blockIdx.x + threadIdx.x;
                    const int stride = blockDim.x * gridDim.x;
                    while (i < N)
                    {
                        O[i] = op(coef_A,A[i],coef_B,B[i]);
                        i += stride;
                    }
                }
            }

            " |] |> String.concat ""

        let kernel = load_kernel kernel_code kernel_name

        member t.Kernel = kernel
        member inline t.A
                (str: CudaStream,
                 coef_x: float32, (ext_x: ^a -> CUdeviceptr, x: ^a), 
                 coef_y: float32, (ext_y: ^a -> CUdeviceptr, y: ^a), 
                                  (ext_o: ^a -> CUdeviceptr, o: ^a)) =
            GuardSizes3(x,y,o)
            let n = Size x
            map_launcher(str, t.Kernel, n, [|coef_x; ext_x x; coef_y; ext_y y; ext_o o; n|])


    /// o <- f(coef_x,x,coef_y,y,coef_z,z)
    type DeviceTrinaryCoefTransformModule(op: string, unique_name) = 
        let block_size = 256

        let kernel_name = "Map3CoefKernel" + unique_name
        let kernel_code = 
            [|"
            //Kernel code:
            extern \"C\" {
                typedef float floatType;
                __device__ inline floatType op(floatType coef_x, floatType x, floatType coef_y, floatType y, floatType coef_z, floatType z)
                {
                    return ";op;"
                }
        
                // Device code
                __global__ void ";kernel_name;"(const floatType coef_A, const floatType* A, const floatType coef_B, const floatType* B, const floatType coef_C, const floatType* C, floatType* O, const int N)
                {
                    int i = blockDim.x * blockIdx.x + threadIdx.x;
                    const int stride = blockDim.x * gridDim.x;
                    while (i < N)
                    {
                        O[i] = op(coef_A,A[i],coef_B,B[i],coef_C,C[i]);
                        i += stride;
                    }
                }
            }

            " |] |> String.concat ""

        let kernel = load_kernel kernel_code kernel_name

        member t.Kernel = kernel
        member inline t.A
                (str: CudaStream,
                 coef_x: float32, (ext_x: ^a -> CUdeviceptr, x: ^a), 
                 coef_y: float32, (ext_y: ^a -> CUdeviceptr, y: ^a), 
                 coef_z: float32, (ext_z: ^a -> CUdeviceptr, z: ^a), 
                                  (ext_o: ^a -> CUdeviceptr, o: ^a)) =
            GuardSizes4(x,y,z,o)
            let n = Size x
            map_launcher(str, t.Kernel, n, [|coef_x; ext_x x; coef_y; ext_y y; coef_z; ext_z z; ext_o o; n|])

    let max_column_launcher(str: CudaStream, kernel: CudaKernel, num_columns: int, args: obj[]) =
        let block_size = 128

        kernel.GridDimensions <- dim3(num_columns)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(str.Stream, args)

    /// o <- max_col(x)
    /// Sets all except one of the max of a column to zero.
    type DeviceMaxColumnActivationModule() = 
        let kernel_name = "MaxColumnActivationKernel"
        let kernel_code = 
            [|"
            //Kernel code:
            extern \"C\" {
                typedef float floatType;
                #define INIT __int_as_float(0xff800000) // The constant init for the reduce operations. This is float negative infinity.
                // The max reduce version.
                __device__ inline floatType warpReduce(floatType value){
                    #pragma unroll
	                for (int i=1; i<32; i*=2) {
                        floatType tmp = __shfl_xor(value, i);
                        value = (tmp > value) ? tmp : value;
                        }
	                return value;
                }
              
                __device__ inline floatType blockReduce(floatType value){
	                __shared__ floatType temp[32];
                    if (threadIdx.x < 32) temp[threadIdx.x] = INIT; 
                    floatType out_partial = warpReduce(value);
                    __syncthreads();
	                if (threadIdx.x % 32 == 0) temp[threadIdx.x / 32] = out_partial;
                    __syncthreads();
	                if (threadIdx.x < 32) out_partial = warpReduce(temp[threadIdx.x]);
                    return out_partial;
                }

                // Device code
                __global__ void ";kernel_name;"(const floatType* A, floatType* O, const int num_rows, const int num_cols)
                {
                    int row = threadIdx.x;
                    //const int col = blockIdx.x;
                    int col_idx = blockIdx.x*num_rows; 
                    floatType max = INIT; // This is the negative infinity for floats.
                    int index = -1;
                    while (row < num_rows)
                    {
                       if (A[row+col_idx] > max) {
                            max = A[row+col_idx];
                            index = row;
                            }
                        row += blockDim.x;
                    }
                
                    __shared__ floatType max_index;
                    if (max == blockReduce(max)) max_index = index;
                    __syncthreads();
                    index = max_index; // These last four lines are to make absolutely sure that only one max is selected in case there is more than one.
                    row = threadIdx.x;
                    while (row < num_rows)
                    {
                        O[row+col_idx] = (row == index) ? max : 0.0f;
                        row += blockDim.x;
                    }
                }
            }

            "|] |> String.concat ""

        let kernel = load_kernel kernel_code kernel_name

        member t.Kernel = kernel
        member inline t.A
                (str: CudaStream,
                 (ext_x: ^a -> CUdeviceptr, x: ^a),
                 (ext_o: ^a -> CUdeviceptr, o: ^a)) = 
            GuardSizes2(x,o)
            let r,c = rc x
            max_column_launcher(str, t.Kernel, c, [|ext_x x; ext_o o; r; c|])

    // The gradient clipping module.
    let gradclipModule = lazy DeviceUnaryCoefTransformModule("(x < -coef_x) ? -coef_x : (x > coef_x ? coef_x : x);", "GradClip") // Unique names like GradClip are necessary for load and saving to drive. Be very careful of collisions.

    // coef_x = scale
    // coef_y = location
    // y does not get used.
    let randMapModule = lazy DeviceBinaryCoefTransformModule("coef_x*(x-0.5f)+coef_y;","RandMapper")

    /// Fills primal of a matrix by sampling from a random uniform distribution in <-1.0f,1.0f]
    let inline fillRandomUniformMatrix(str: CudaStream, x: ^a, scaling_factor: float32, location: float32) =
        cudaRandom.SetStream str.Stream
        cudaRandom.GenerateUniform(extract_primal' x)

        let x = extract_primal, x
        // 2.0f*scaling_factor ensures that it is rescaled around zero if the scaling_factor is 1.0f.
        randMapModule.Value.A(str, 2.0f*scaling_factor,x,location,x,x)

    // y <- alpha * x + y
    let inline saxpy 
            (str: CudaStream) 
            (alpha:float32) (ext_x, x: ^a) (ext_y, y: ^a) =
        GuardSizes2(x,y)
        cublas.Stream <- str.Stream
        cublas.Axpy(alpha,ext_x x,1,ext_y y,1)

    /// General matrix-matrix addition. Inplace version.
    let inline geam 
            (str: CudaStream) transa transb 
            (alpha: float32) (ext_a: ^a -> CudaDeviceVariable<float32>, A: ^a) 
            (beta: float32)  (ext_b: ^a -> CudaDeviceVariable<float32>, B: ^a) 
                             (ext_c: ^a -> CudaDeviceVariable<float32>, C: ^a) =
        let a_row = if transa = nT then rows A else cols A
        let a_col = if transa = nT then cols A else rows A
        let b_row = if transb = nT then rows B else cols B
        let b_col = if transb = nT then cols B else rows B
        
        if a_row <> b_row then failwithf "a_row <> b_row in geam! %i <> %i" a_row b_row
        if a_col <> b_col then failwithf "a_col <> b_col in geam! %i <> %i" a_col b_col

        if a_row <> rows C then failwithf "a_row <> C_num_rows in geam! %i <> %i" a_col (rows C)
        if a_col <> cols C then failwithf "a_col <> C_num_cols in geam! %i <> %i" a_col (cols C)

        let lda = if transa = nT then a_row else a_col
        let ldb = if transa = nT then b_row else b_col
        let ldc = a_row

        cublas.Stream <- str.Stream
        cublas.Geam(transa, transb, a_row, a_col, alpha, ext_a A, lda, ext_b B, ldb, beta, ext_c C, ldc)

    // Uncomment gemm to manifest the bug. 
    // The test in Program.fs will not be detected with this function uncommented.

    /// General matrix-matrix multiply from cuBLAS. Inplace version
    let inline gemm 
            (str: CudaStream) transa transb 
            (alpha: float32) (ext_a: _ -> CudaDeviceVariable<float32>, A)
                             (ext_b: _ -> CudaDeviceVariable<float32>, B)
            (beta: float32)  (ext_c: _ -> CudaDeviceVariable<float32>, C) =

        // -------

        // These two are meant to be called from inside gemm as they lack boundary checks.
        // I've added them to enhance gemm's vector handling capabilities for online learning
        // tasks.

        /// o <- alpha * op(A) * x + beta * o
        /// Matrix-vector multiplication. Inplace version.
        let inline gemv
                (str: CudaStream) transa transb
                (alpha:float32) (ext_a, A) 
                                (ext_x, x) 
                (beta:float32)  (ext_o, o) =
            let m = rows A
            let n = cols A
            let lda = m
            cublas.Stream <- str.Stream
            cublas.Gemv(transa, m, n, alpha, ext_a A, lda, ext_x x, 1, beta, ext_o o, 1)

        // A <- alpha * x * yT + beta * A (outer product)
        let inline ger 
                (str: CudaStream)
                (alpha: float32) (ext_x, x)
                                 (ext_y, y)
                (beta: float32)  (ext_a, a) =
            let dom_x = max (rows x) (cols x)
            if beta <> 1.0f then geam str nT nT beta (ext_a, a) 0.0f (ext_a, a) (ext_a, a) 
            cublas.Stream <- str.Stream
            cublas.Ger(alpha, ext_x x, 1, ext_y y, 1, ext_a a, dom_x)

        // -------

        let inline is_vector (x: ^a) = rows x = 1 || cols x = 1

        let a_col = if transa = nT then cols A else rows A
        let b_row = if transb = nT then rows B else cols B
        if a_col <> b_row then failwithf "a_col(%i) <> b_row(%i) in gemm!" a_col b_row
        let m = if transa = nT then rows A else cols A
        let n = if transb = nT then cols B else rows B
        let k = a_col
        let lda = if transa = nT then m else k
        let ldb = if transb = nT then k else n
        let ldc = m

        if m <> rows C || n <> cols C then failwithf "m(%i) <> rows C(%i) || n(%i) <> cols C(%i)" m (rows C) n (cols C)

        // If is outer product call ger
        if a_col = 1 && b_row = 1 then 
            ger str alpha (ext_a, A) (ext_b, B) beta (ext_c, C)
        // If the vector is on the right side or both are vectors call gemv normally.
        elif is_vector B then 
            gemv str transa transb alpha (ext_a,A) (ext_b,B) beta (ext_c,C)
        // If the vector is on the left side call gemv with the arguments switched and transposed
        // It does not actually transpose them, just their views. The function should work regardless.
        elif is_vector A then
            let opta = if transa = nT then T else nT
            let optb = if transb = nT then T else nT
            gemv str optb opta alpha (ext_b,B) (ext_a,A) beta (ext_c,C)
        // Just do the standard matrix multiply
        else
            cublas.Stream <- str.Stream
            cublas.Gemm(transa, transb, m, n, k, alpha, ext_a A, lda, ext_b B, ldb, beta, ext_c C, ldc)

            
    /// Does not only check, but also sets the undefined nodes to Dead or Alive.
//    let inline deadness_check c a (f : unit -> unit) =
//        match isDead c with
//        | Undefined -> failwith "The upper node should not be undefined."
//        | Dead -> // If the upper node is dead no backpropagation is done.
//            match isDead a with
//            | Undefined -> deadIs a Dead
//            | Dead | Alive -> () // If the bottom node is Alive do not modify it to be Dead as there exists a path from elsewhere through it.
//        | Alive -> 
//            deadIs a Alive
//            f()
//
//    type StanState =
//        {
//        IsInferenceOnly: bool
//        Workspace: Workspace
//        Str: CudaStream
//        Mem: ObjectPool
//        Tape: Stack<unit -> unit>
//        }
//
//        static member create(?is_inference_only) =
//            {IsInferenceOnly=defaultArg is_inference_only false; 
//             Workspace=new Workspace(); Str=new CudaStream(); 
//             Mem=new ObjectPool(); Tape=Stack()}
//
//        member t.WithIsInferenceOnly x =
//            {t with IsInferenceOnly=x}
//
//        member t.Dispose() =
//            t.Workspace.Dispose()
//            t.Str.Dispose()
//            t.Mem.Dispose()
//            t.Tape.Clear()
//
//        interface IDisposable with
//            member t.Dispose() = t.Dispose()
//
//    let inline with_is_inference_only (x: ^a) v =
//        (^a: (member WithIsInferenceOnly: bool -> ^a) x, v)
//
//    let inline tape (x: ^a) =
//        (^a: (member Tape: Stack<unit -> unit>) x)
//
//    let inline mem (x: ^a) =
//        (^a: (member Mem: ObjectPool) x)
//
//    let inline str (x: ^a) =
//        (^a: (member Str: CudaStream) x)
//
//    let inline workspace (x: ^a) =
//        (^a: (member Workspace: Workspace) x)
//
//    let inline is_inference_only (x: ^a) =
//        (^a: (member IsInferenceOnly: bool) x)
//
//    /// Gets a dM the same size as the first one from the object pool.
//    let inline getdMLike (x: ^a) (p: ObjectPool) is_constant is_inference_only str = // TODO: Like copy, refactor this one too.
//        (^a: (member GetFromObjectPool: ObjectPool * bool * bool * CudaStream -> ^a) x, p, is_constant, is_inference_only, str)
//
//    /// Copies the dM type using the object pool.
//    let inline copy (x: ^a) is_constant state =
//        (^a: (member CopyUsingObjectPool: ObjectPool * bool * bool * CudaStream -> ^a) x, mem state, is_constant,  is_inference_only state , str state)
//
//    ///// Matrix-matrix multiply.
//    let inline private matmult' (prev_output : d2M option) (a, b) (state: ^state) =
//        let c = 
//            match prev_output with
//            | None ->
//                let num_rows = rows a
//                let num_cols = cols b
//                (mem state).Getd2M(false,(num_rows,num_cols),(is_inference_only state),(str state))
//                |> fun c ->
//                    gemm (str state) nT nT 1.0f (P' a) (P' b) 0.0f (P' c)
//                    c
//            | Some c ->
//                gemm (str state) nT nT 1.0f (P' a) (P' b) 1.0f (P' c)
//                c
//    
//        if (is_inference_only state) = false then
//            if hasAdjoint a then 
//                let matmult_backward_left () = 
//                    deadness_check c a <| fun _ -> 
//                        gemm (str state) nT T 1.0f (A' c) (P' b) 1.0f (A' a)
//                (tape state).Push(matmult_backward_left)
//
//            if hasAdjoint b then 
//                let matmult_backward_right () = 
//                    deadness_check c b <| fun _ -> 
//                        gemm (str state) T nT 1.0f (P' a) (A' c) 1.0f (A' b)
//                (tape state).Push(matmult_backward_right)
//        c, state
//
//    let inline matmult a b = matmult' None (a, b)
//
//    let inline tensor_add' add_to_left alpha (left : ^a) beta (right : ^a) (state: ^state) =
//        let left_nchw = nchw left
//        let right_nchw = nchw right
//        let leftDesc = (mem state).GetTensorDescriptor left_nchw
//        let rightDesc = (mem state).GetTensorDescriptor right_nchw
//
//        let output = 
//            if add_to_left = false then
//                getdMLike left (mem state) false (is_inference_only state) (str state)
//                |> fun output ->
//                    geam (str state) nT nT 1.0f (P' left) 0.0f (P' output) (P' output)
//                    output
//            else 
//                left
//
//        if left_nchw <> right_nchw then
//            cudnn.SetStream (str state)
//            cudnn.AddTensor(beta,rightDesc,extract_primal' right, alpha,leftDesc,extract_primal' output) // Add right to output.
//        else 
//            geam (str state) nT nT beta (P' right) alpha (P' output) (P' output)
//
//        if (is_inference_only state) = false then
//            if hasAdjoint right then 
//                let tensor_add_right_backwards () =
//                    deadness_check output right
//                    <| fun _ ->
//                        if left_nchw = right_nchw then
//                            saxpy (str state) beta (A' output) (A' right)
//                        else
//                            cudnn.SetStream (str state)
//                            let left_nchw = nchwBiasAdd left // Ugly hack to get cuDNN to work with 2D matrices.
//                            let right_nchw = nchwBiasAdd right
//                            let leftDesc = (mem state).GetTensorDescriptor left_nchw
//                            let rightDesc = (mem state).GetTensorDescriptor right_nchw
//                            cudnn.ConvolutionBackwardBias(beta,leftDesc,extract_adjoint' output,1.0f,rightDesc,extract_adjoint' right)
//
//                (tape state).Push(tensor_add_right_backwards)
//
//            if add_to_left = false && hasAdjoint left then // No point in adding the adjoint to itself.
//                let tensor_add_left_backwards () = 
//                    deadness_check output left 
//                    <| fun _ -> 
//                        saxpy (str state) alpha (A' output) (A' left)
//                (tape state).Push(tensor_add_left_backwards)
//        output, state
//
//    let inline linear_layer_matmult (mm: (d2M*d2M) []) (bias: d2M option) (state: ^state) =
//        mm
//        |> Array.fold (fun (prev_output,state) input -> 
//            matmult' prev_output input state
//            |> fun (input',state') -> (Some input',state')
//            ) (None,state)
//        |>  function
//            | None, _ -> failwith "There must be one input in mm"
//            | Some left, state ->
//                match bias with
//                | None -> left, state
//                | Some right -> tensor_add' true 1.0f left 1.0f right state
//
//    /// The activation function. Zeroes out the target primal during the call.
//    let inline activation_forward mode (input : ^a) (state: ^state) =
//        let input_sizes = nchw input
//        let srcTensorDesc = (mem state).GetTensorDescriptor input_sizes
//
//        let output = getdMLike input (mem state) false (is_inference_only state) (str state)
//
//        cudnn.SetStream (str state)
//        cudnn.ActivationForward(mode,1.0f,srcTensorDesc,extract_primal' input,0.0f,srcTensorDesc,extract_primal' output)
//        if (is_inference_only state) = false then
//            if hasAdjoint input then 
//                let activation_backward () =
//                    deadness_check output input 
//                    <| fun _ -> 
//                        cudnn.SetStream (str state)
//                        cudnn.ActivationBackward(mode,1.0f,srcTensorDesc,extract_primal' output,srcTensorDesc,extract_adjoint' output,srcTensorDesc,extract_primal' input,1.0f,srcTensorDesc,extract_adjoint' input)
//                (tape state).Push(activation_backward)
//        output, state
//
//    /// The pooling function. Zeroes out the target primal during the call.
//    let inline pooling_forward p (input : d4M) (state: ^state) =
//        let poolingDescriptor = (mem state).GetPoolingDescriptor p
//        let input_sizes = input.nchw
//        let srcTensorDesc = (mem state).GetTensorDescriptor input_sizes
//        let dest_sizes = poolingDescriptor.GetPooling2dForwardOutputDim srcTensorDesc
//
//        let output = (mem state).Getd4M(false,input.nchw,(is_inference_only state),(str state))
//
//        let dstTensorDesc = (mem state).GetTensorDescriptor dest_sizes
//
//        cudnn.SetStream (str state)
//        cudnn.PoolingForward(poolingDescriptor,1.0f,srcTensorDesc,extract_primal' input,0.0f,dstTensorDesc,extract_primal' output)
//
//        if (is_inference_only state) = false then
//            if hasAdjoint input then 
//                let pooling_backward () =
//                    deadness_check output input 
//                    <| fun _ ->
//                        cudnn.SetStream (str state)
//                        cudnn.PoolingBackward(poolingDescriptor,1.0f,srcTensorDesc,extract_primal' output, srcTensorDesc,
//                                              extract_adjoint' output,dstTensorDesc,extract_primal' input,1.0f,dstTensorDesc,extract_adjoint' input)
//                (tape state).Push(pooling_backward)
//        output, state
//
//
//    let inline private convolutional_forward' (prev_output: d4M option) (convPar, data : d4M, filter : d4M) (state: ^state) =
//        let data_sizes = data.nchw
//        let filter_sizes = filter.nchw
//
//        let srcTensorDesc = (mem state).GetTensorDescriptor data_sizes
//    
//        let filterDesc = (mem state).GetFilterDescriptor filter_sizes
//        let convDesc = (mem state).GetConvolutionDescriptor convPar
//
//        let dims, output = 
//            let dims = convDesc.GetConvolution2dForwardOutputDim(srcTensorDesc,filterDesc)
//            match prev_output with
//            | Some prev_output ->
//                let prev_dims = prev_output.nchw
//                if dims <> prev_dims then failwith "dims <> prev_dims in linear_layer_conv"
//                prev_dims, prev_output
//            | None ->
//                dims, (mem state).Getd4M(false,dims,(is_inference_only state),(str state))
//
//        let dstTensorDesc = (mem state).GetTensorDescriptor dims
//
//        let algo = 
//            cudnn.GetConvolutionForwardAlgorithm(
//                srcTensorDesc,filterDesc,convDesc,dstTensorDesc,
//                cudnnConvolutionFwdPreference.PreferFastest,SizeT 0)
//        let _ = 
//            let workspace = 
//                cudnn.GetConvolutionForwardWorkspaceSize(srcTensorDesc, filterDesc, convDesc, dstTensorDesc, algo) 
//                |> int |> (workspace state).ResizeIf
//
//            let beta = 
//                match prev_output with
//                | None -> 0.0f
//                | Some _ -> 1.0f
//
//            cudnn.SetStream (str state)
//            cudnn.ConvolutionForward(1.0f,srcTensorDesc,extract_primal' data,filterDesc,extract_primal' filter,convDesc,algo,workspace,beta,dstTensorDesc,extract_primal' output) // Don't zero out the previous output.
//
//        if (is_inference_only state) = false then
//            if hasAdjoint filter then 
//                let convolution_backwards_filter () =
//                    deadness_check output filter 
//                    <| fun _ ->
//                        let algo = 
//                            cudnn.GetConvolutionBackwardFilterAlgorithm(
//                                srcTensorDesc,dstTensorDesc,convDesc,filterDesc,
//                                cudnnConvolutionBwdFilterPreference.PreferFastest,SizeT 0)
//
//                        let workspace =
//                            cudnn.GetConvolutionBackwardFilterWorkspaceSize(srcTensorDesc,dstTensorDesc,convDesc,filterDesc,algo) 
//                            |> int |> (workspace state).ResizeIf
//
//                        cudnn.SetStream (str state)
//                        cudnn.ConvolutionBackwardFilter(
//                            1.0f,srcTensorDesc,extract_primal' data,dstTensorDesc,extract_adjoint' output,
//                            convDesc,algo,workspace,1.0f,filterDesc,extract_adjoint' filter)
//
//                (tape state).Push(convolution_backwards_filter)
//
//            if hasAdjoint data then 
//                let convolution_backwards_data () =
//                    deadness_check output data 
//                    <| fun _ ->
//                        let algo = 
//                            cudnn.GetConvolutionBackwardDataAlgorithm(
//                                filterDesc,dstTensorDesc,convDesc,srcTensorDesc,
//                                cudnnConvolutionBwdDataPreference.PreferFastest,SizeT 0)
//
//                        let workspace = 
//                            cudnn.GetConvolutionBackwardDataWorkspaceSize(filterDesc,dstTensorDesc,convDesc,srcTensorDesc,algo) 
//                            |> int |> (workspace state).ResizeIf
//
//                        cudnn.SetStream (str state)
//                        cudnn.ConvolutionBackwardData(
//                            1.0f,filterDesc,extract_primal' filter,dstTensorDesc,
//                            extract_adjoint' output,convDesc,1.0f,algo,workspace,srcTensorDesc,extract_adjoint' data)
//
//                (tape state).Push(convolution_backwards_data)
//
//        output, state
//
//    /// The convolutional function. Zeroes out the target primal during the call.
//    let inline convolution_forward convPar (data : d4M) (filter : d4M) = 
//        convolutional_forward' None (convPar,data,filter)
//    
//    let inline batch_normalization_forward 
//            bnMode (bnScale : d4M) (bnBias : d4M) (bnRunningMean : d4M) 
//            (bnRunningVariance : d4M) exponentialAverageFactor (input : d4M) (state: ^state) =
//        let input_sizes = input.nchw
//        let bias_sizes = bnBias.nchw
//        let srcTensorDesc = (mem state).GetTensorDescriptor input_sizes
//
//        let bnDesc = 
//            (mem state).GetBNDescriptor (input_sizes, bnMode, srcTensorDesc)
//
//        let _ =
//            let mutable d,n,c,h,w,sn,sc,sh,sw = cudnnDataType.Double,0,0,0,0,0,0,0,0
//            bnDesc.GetTensor4dDescriptor(&d,&n,&c,&h,&w,&sn,&sc,&sh,&sw)
//            let bn_nchw = n,c,h,w
//            if bn_nchw <> bnScale.nchw then failwith "Tensor dimensions for bnScale are incorrect."
//            if bn_nchw <> bnBias.nchw then failwith "Tensor dimensions for bnBias are incorrect."
//            if bn_nchw <> bnRunningMean.nchw then failwith "Tensor dimensions for bnRunningMean are incorrect."
//            if bn_nchw <> bnRunningVariance.nchw then failwith "Tensor dimensions for bnRunningVariance are incorrect."
//
//        let alpha, beta = 1.0f, 0.0f
//        let epsilon = 1e-5
//        let bnSavedMean = (mem state).Getd4M(true,bias_sizes,(is_inference_only state),(str state))
//        let bnSavedVariance = (mem state).Getd4M(true,bias_sizes,(is_inference_only state),(str state))
//        let output = (mem state).Getd4M(false,input_sizes,(is_inference_only state),(str state))
//
//        if (is_inference_only state) then
//            cudnn.SetStream (str state)
//            cudnn.BatchNormalizationForwardTraining(
//                bnMode, alpha, beta, srcTensorDesc, extract_primal' input, srcTensorDesc,
//                extract_primal' output, bnDesc, extract_primal' bnScale, extract_primal' bnBias,
//                exponentialAverageFactor, extract_primal' bnRunningMean, extract_primal' bnRunningVariance,
//                epsilon, extract_primal' bnSavedMean, extract_primal' bnSavedVariance)
//
//            if hasAdjoint input then 
//                let batch_normalization_backward () =
//                    let dx_alpha, dx_beta = 1.0f, 1.0f
//                    let param_alpha, param_beta = 1.0f, 1.0f
//
//                    deadness_check output input 
//                    <| fun _ ->
//                        cudnn.SetStream (str state)
//                        cudnn.BatchNormalizationBackward(
//                            bnMode, dx_alpha, dx_beta, param_alpha, param_beta, srcTensorDesc,
//                            extract_primal' input, srcTensorDesc, extract_adjoint' output, srcTensorDesc,
//                            extract_adjoint' input, bnDesc, extract_primal' bnScale, extract_adjoint' bnScale,
//                            extract_adjoint' bnBias, epsilon, extract_primal' bnSavedMean, extract_primal' bnSavedVariance)
//
//                (tape state).Push batch_normalization_backward
//        else
//                cudnn.SetStream (str state)
//                cudnn.BatchNormalizationForwardInference(
//                    bnMode, alpha, beta, srcTensorDesc,extract_primal' input, srcTensorDesc,
//                    extract_primal' output, bnDesc, extract_primal' bnScale, extract_primal' bnBias,
//                    extract_primal' bnRunningMean, extract_primal' bnRunningVariance, epsilon)
//        
//        output, state
//    
//    let inline linear_layer_conv (convs: (ConvolutionParameters*d4M*d4M) []) (bias: d4M option) (state: ^state) =
//        let folder prev_output input = 
//            match prev_output with
//            | Some (output, state) ->
//                convolutional_forward' (Some output) input state |> Some
//            | None ->
//                convolutional_forward' None input state |> Some
//    
//        Array.fold folder None convs
//        |>  function
//            | Some(left, state) ->
//                match bias with
//                | None -> left, state
//                | Some right -> tensor_add' true 1.0f left 1.0f right state
//            | None -> failwith "linear_layer_conv has to have at least one input"
//
//
//    let hadamaradMultiplicationModule = lazy new DeviceBinaryTransformModule("x*y;", "HadMult")
//    let hadamaradMultiplicationErrorModule = lazy new DeviceTrinaryTransformModule("x*y+z;", "HadMultError")
//    /// Hadamarad (elementwise) multiplication function.
//    let inline private hadmult' (prev_output : ^a option) (a: ^a,b: ^a) (state: ^state) =
//        let c = 
//            match prev_output with
//            | Some c -> 
//                hadamaradMultiplicationErrorModule.Value.A((str state), P a, P b, P c, P c)
//                c
//            | None -> 
//                getdMLike a (mem state) false (is_inference_only state) (str state)
//                |> fun c -> 
//                    hadamaradMultiplicationModule.Value.A((str state), P a, P b, P c)
//                    c
//
//        if (is_inference_only state) = false then
//            if hasAdjoint a then 
//                let hadmult_backward_left () = 
//                    deadness_check c a 
//                    <| fun _ ->
//                        hadamaradMultiplicationErrorModule.Value.A((str state), P b, A c, A a, A a)
//                (tape state).Push hadmult_backward_left
//            if hasAdjoint b then 
//                let hadmult_backward_right () = 
//                    deadness_check c b 
//                    <| fun _ ->
//                        hadamaradMultiplicationErrorModule.Value.A((str state), P a, A c, A b, A b)
//                (tape state).Push hadmult_backward_right
//        c, state
//
//    let inline hadmult (a: ^a) (b: ^a) = hadmult' None (a, b)
//    let inline linear_layer_hadmult (hads: (^a * ^a)[]) (state: ^state) = 
//        hads 
//        |> Array.fold (fun (prev_output,state) input ->
//            hadmult' prev_output input state
//            |> fun (input',state') -> (Some input',state')
//            ) (None, state) 
//        |>  function
//            | None, _ -> failwith "linear_layer_hadmult requires at least one input"
//            | Some v, state -> v, state
//
//    let squareModule = lazy new DeviceUnaryTransformModule("x*x;","Square")
//    //y = error
//    //z = previous adjoint value
//    let squareErrorModule = lazy new DeviceTrinaryTransformModule("2.0f*x*y + z;","SquareError")
//    let inline square (a:^a) (state: ^state) =
//        let c = getdMLike a (mem state) false (is_inference_only state) (str state)
//
//        squareModule.Value.A((str state), P a, P c)
//
//        if (is_inference_only state) = false then
//            if hasAdjoint a then 
//                let square_backward () = 
//                    deadness_check c a 
//                    <| fun _ -> 
//                        squareErrorModule.Value.A((str state), P a, A c, A a, A a)
//                (tape state).Push square_backward
//        c, state
//
//    /// This one is for debugging currently
//    let squareSumModule = lazy new DeviceUnaryMapSumModule("x*x;", "SquareSum")
//
//    let sumModule = lazy new DeviceUnaryMapSumModule("x;", "Sum")
//    let sumErrorModule = lazy new DeviceUnaryCoefTransformModule("coef_x + x;", "SumError")
//    let inline sum (a: ^a) (state: ^state) =
//        let c = Df.create 0.0f
//
//        c.P := sumModule.Value.A((str state), P a)
//
//        if (is_inference_only state) = false then
//            if hasAdjoint a then 
//                let sum_backward () = 
//                    if !c.A <> 0.0f then 
//                        deadIs a Alive
//                        sumErrorModule.Value.A((str state), !c.A, A a, A a)
//                    else deadIs a Dead
//                (tape state).Push sum_backward
//        c, state
//
//    let inline scale (alpha: float32) (a:Df) (state: ^state) =
//        let c = Df.create 0.0f
//        c.P := lazy (alpha * a.P.Value.Value)
//    
//        if (is_inference_only state) = false then
//            let scale_backward () = a.A := alpha * !c.A + !a.A
//            (tape state).Push scale_backward
//
//        c, state
//
//    let inline sum_scalars (a:Df[]) (state: ^state) =
//        let c = Df.create 0.0f
//        c.P :=
//            lazy 
//                let mutable t = 0.0f
//                for l in a do 
//                    t <- t + l.P.Value.Value
//                t
//
//        if (is_inference_only state) = false then
//            let sum_scalars_backwards () = for l in a do l.A := !c.A + !l.A
//            (tape state).Push sum_scalars_backwards
//
//        c, state
//
//    let logModule = lazy new DeviceUnaryTransformModule("logf(x);","Log")
//    //y=error
//    //z=previous adjoint
//    let logErrorModule = lazy new DeviceTrinaryTransformModule("y / x + z;","LogError")
//    let inline log_ (a:^a) (state: ^state) =
//        let c = getdMLike a (mem state) false (is_inference_only state) (str state)
//
//        logModule.Value.A((str state), P a, P c)
//
//        if (is_inference_only state) = false then
//            if hasAdjoint a then
//                let log_backward () = 
//                    deadness_check c a 
//                    <| fun _ -> 
//                        logErrorModule.Value.A((str state), P a, A c, A a, A a)
//                (tape state).Push log_backward
//
//        c, state
//
//    //coef_x = scalar
//    //coef_y = coef
//    let scalarMatrixAddModule = lazy new DeviceBinaryCoefTransformModule("coef_x + coef_y*x;","ScalarMatrixAdd")
//    /// o <- scalar + coef*a
//    let inline scalar_matrix_add scalar coef (a:^a) (state: ^state) =
//        let c = getdMLike a (mem state) false (is_inference_only state) (str state)
//
//        scalarMatrixAddModule.Value.A((str state), scalar, P a, coef, P a, P c)
//
//        if (is_inference_only state) = false then
//            if hasAdjoint a then
//                let scalar_matrix_add_backward () = 
//                    deadness_check c a
//                    <| fun _ -> 
//                        saxpy (str state) coef (A' c) (A' a)
//                (tape state).Push scalar_matrix_add_backward
//        c, state
//
//    let inline add alpha (a: ^a) beta (b: ^a) (state: ^state) =
//        let c = getdMLike a (mem state) false (is_inference_only state) (str state)
//
//        geam (str state) nT nT alpha (P' a) beta (P' b) (P' c)
//
//        if (is_inference_only state) = false then
//            if hasAdjoint a then
//                let add_backward_left () = 
//                    deadness_check c a
//                    <| fun _ ->
//                        saxpy (str state) alpha (A' c) (A' a)
//                (tape state).Push add_backward_left
//            if hasAdjoint b then
//                let add_backward_right () = 
//                    deadness_check c b
//                    <| fun _ ->
//                        saxpy (str state) beta (A' c) (A' b)
//                (tape state).Push add_backward_right
//        c, state
//
//    let inline softmax_instance_forward (data : ^a) (state: ^state) =
//        let data_sizes = nchw data
//
//        let srcTensorDesc = (mem state).GetTensorDescriptor data_sizes
//        let output = getdMLike data (mem state) false (is_inference_only state) (str state)
//
//        let algo = cudnnSoftmaxAlgorithm.Accurate // Log mode forgets to re-exponentiate at the end.
//        let mode = cudnnSoftmaxMode.Instance
//
//        cudnn.SetStream (str state)
//        cudnn.SoftmaxForward(algo, mode, 1.0f, srcTensorDesc, extract_primal' data, 0.0f, srcTensorDesc, extract_primal' output)
//
//        if (is_inference_only state) = false then
//            if hasAdjoint data then
//                let softmax_channel_backward () =
//                    deadness_check output data
//                    <| fun _ ->
//                        cudnn.SetStream (str state)
//                        cudnn.SoftmaxBackward(algo,mode,1.0f,srcTensorDesc,extract_primal' output,srcTensorDesc,extract_adjoint' output,1.0f,srcTensorDesc,extract_adjoint' data)
//                (tape state).Push softmax_channel_backward
//        output, state
//
//    let inline softmax x = softmax_instance_forward x
//
//    let clipModule = lazy new DeviceTrinaryCoefTransformModule("((x < coef_x) ? coef_x : (x > coef_y ? coef_y : x))+coef_z;","Clip")
//    let clipErrorModule = lazy new DeviceTrinaryCoefTransformModule("y*((x < coef_x) ? 0.0f : (x > coef_y ? 0.0f : 1.0f))+z;","ClipError")
//    /// o <- clip(min,max,a)+scalar
//    /// The clip function. Can be used as Relu by setting max to positive infinity. 
//    /// Can be used to make linear clipped sigmoid by setting min,max,scalar to -0.5f,0.5f,0.5f.
//    let inline clip min max (a : ^a) scalar (state: ^state) =
//        let c = getdMLike a (mem state) false (is_inference_only state) (str state)
//
//        clipModule.Value.A((str state), min, P a, max, P a,scalar, P a, P c)
//
//        if (is_inference_only state) = false then
//            if hasAdjoint a then
//                let clip_backward () = 
//                    deadness_check c a
//                    <| fun _ -> 
//                        clipErrorModule.Value.A((str state), min, P a, max, A c, max, A a, A a)
//                (tape state).Push clip_backward
//        c, state
//
//    let inline relu x (state: ^state) = 
//        let t = (mem state).GetActivationDescriptor (cudnnActivationMode.Relu, defaultReluNanOption, 0.0)
//        activation_forward t x state
//    let inline sigmoid x (state: ^state) = 
//        let t = (mem state).GetActivationDescriptor (cudnnActivationMode.Sigmoid, defaultReluNanOption, 0.0)
//        activation_forward t x state
//    let inline tanh_ x (state: ^state) = 
//        let t = (mem state).GetActivationDescriptor (cudnnActivationMode.Tanh, defaultReluNanOption, 0.0)
//        activation_forward t x state
//    let inline clipped_sigmoid x (state: ^state) = 
//        let x,state = sigmoid x state
//        clip 0.0001f 0.9999f x 0.0f state
//    let inline clipped_softmax x (state: ^state) = 
//        let x,state = softmax x state
//        clip 0.0001f 0.9999f x 0.0f state
//
//    /// Bind for the state monad.
//    let inline (>>=) (a: ^s -> ^a * ^s) (f: ^a -> ^s -> ^b * ^s): ^s -> ^b * ^s =
//        fun state ->
//            let v',s' = a state
//            f v' s'
//
//
//    // TODO: In a future version of Spiral, make the two operations actually run in parallel by
//    // passing in different streams and waiting on them using events. This would be a more controlable
//    // that what I tried in V3 of the library where I did not observe any speedups from concurrency worth noting.
//    /// Runs the operations in parallel and collects the results.
//    let inline para2 f1 f2 (a0: ^a) (s: ^s) =
//        let a1, s = f1 a0 s
//        let a2, s = f2 a0 s
//        (a1,a2),s
//
//    let inline squared_error_cost (target: ^a) (activations: ^a) =
//        add 1.0f target -1.0f activations 
//        >>= square
//        >>= sum
//        >>= scale (0.5f / float32 (cols target))
//
//    type StateBuilder() =
//        member inline t.Return a = fun s -> (a,s)
//        member inline t.Bind(a,f) = a >>= f
//        member inline t.ReturnFrom x = x
//
//    let state = StateBuilder()
//
//    let inline cross_entropy_cost (target: ^a) (activations: ^a) = state {
//        let lt = target
//        let! ll = log_ activations
//        let! rt = scalar_matrix_add 1.0f -1.0f target
//        let! rl = scalar_matrix_add 1.0f -1.0f activations >>= log_
//        return! linear_layer_hadmult [|lt, ll; rt, rl|] 
//                >>= sum 
//                >>= scale (-1.0f / float32 (cols target))
//        }
//
//
//    let maxColumnActivationModule = lazy new DeviceMaxColumnActivationModule()
//    let accuracyModule = lazy new DeviceBinaryMapSumModule("(x*y == 0.0f) ? 0.0f : 1.0f;","Accuracy")
//    /// Gets the accuracy using the workspace memory.
//    // TODO: Using the workspace might be unsafe for this.
//    let inline get_accuracy (targets: ^a) (activations : ^a) (state: ^state) =
//        let a =
//            lazy
//                let o = convertdMLikeFromWorkspace targets (workspace state)
//                maxColumnActivationModule.Value.A((str state), P activations, P o)
//                accuracyModule.Value.A((str state), P targets, P o).Value
//                |> round |> int
//        a, state
//
//    /// Squared error cost that also returns the accuracy.
//    let inline squared_error_cost' (targets: ^a) (activations : ^a) (state: ^state) =
//        para2 (get_accuracy targets) (squared_error_cost targets) activations state
//
//    /// Cross entropy cost that also returns the accuracy.
//    let inline cross_entropy_cost' (targets: ^a) (activations : ^a) (state: ^state) =
//        para2 (get_accuracy targets) (cross_entropy_cost targets) activations state
//
//    let find_max_index (action_values : float32[]) =
//        let mutable max = Single.NegativeInfinity
//        let mutable index = -1
//        for i=0 to action_values.Length-1 do
//            let x = action_values.[i]
//            if max < x then max <- x; index <- i
//        index
//
//    /// As it says on the tin. TODO: Make initializers for other activation functions.
//    let inline reluInitializer (state: ^state) (a: ^a)  =
//        let scale = (1.0f / sqrt(addDims a |> float32))
//        fillRandomUniformMatrix((str state),a,scale,0.0f)
//        a
//
//    let inline sgd learning_rate (node : ^a) (state: ^state) = 
//        saxpy (str state) -learning_rate (A' node) (P' node)
//        extract_adjoint' node |> fun x -> x.MemsetAsync(0u,(str state).Stream)
//
//    let inline clipped_sgd clipping_threshold learning_rate (node : ^a) (state: ^state) = 
//        gradclipModule.Value.A((str state), clipping_threshold, A node,A node)
//        saxpy (str state) -learning_rate (A' node) (P' node)
//        extract_adjoint' node |> fun x -> x.MemsetAsync(0u,(str state).Stream)
//
//    let inline disposeLayer x =
//        (^a: (member ToArray: ^b[]) x)
//        |> Array.iter dispose
//
//
//    type FFRec<'state when 'state: (member Tape: Stack<unit -> unit>)
//                       and 'state: (member Mem: ObjectPool)
//                       and 'state: (member Str: CudaStream)
//                       and 'state: (member Workspace: Workspace)
//                       and 'state: (member IsInferenceOnly: bool)> =
//        {
//        W: d2M
//        b: d2M
//        a: d2M -> 'state -> d2M * 'state
//        }
//
//        member inline t.Run(x: d2M) = linear_layer_matmult [|t.W,x|] (Some t.b) >>= t.a
//        member inline l.ToArray = [|l.W;l.b|]
//        member inline t.Update (update_function : d2M -> ^state -> unit, state: ^state) =
//            t.ToArray |> Array.iter (fun x -> update_function x state)
//
//        static member inline createReluRandomLayer desired_hidden_size (input: d2M) (state: ^state) =
//            {
//             W = d2M.create(desired_hidden_size,input.Rows) |> reluInitializer state
//             b = d2M.create(desired_hidden_size,1) |> reluInitializer state
//             a = relu
//            } 
//
//        static member inline createSigmoidRandomLayer desired_hidden_size (input: d2M) (state: ^state) =
//            FFRec.createReluRandomLayer desired_hidden_size input state // TODO: Make sigmoid initializer later.
//            |> fun x -> {x with a = sigmoid}
//
//        static member inline createClippedSigmoidRandomLayer desired_hidden_size (input: d2M) (state: ^state) =
//            FFRec.createReluRandomLayer desired_hidden_size input state // TODO: Make sigmoid initializer later.
//            |> fun x -> {x with a = clipped_sigmoid}
//
//    type LayerType<'a, 'state, 'layer_type> =
//        | Uninitialized of desired_hidden_size: int * create_layer: (int -> 'a -> ^state -> 'layer_type)
//        | Initialized of node: 'layer_type
//
//        member t.Value =
//            match t with
//            | Initialized v -> v
//            | _ -> failwith "Node is uninitialized."
//
//    // The actual layers are higher order classes. Now this is genuinely new to V4 of Spiral.
//    // I could not have done this without statically resolved type parameters.
//    // I never expected I would get so much mileage out of them.
//    type Layer<'a, 'state, 'b,
//                'layer_type when 'layer_type: (member Run: 'a -> ('state -> 'b * 'state)) 
//                             and 'layer_type: (member ToArray: 'a[]) 
//                             and ^layer_type: (member Update: ('a -> 'state -> unit) * 'state -> unit)> =
//        {
//        mutable node: LayerType<'a, 'state, 'layer_type>
//        }
//
//        member inline t.RunLayer (input: ^a) = fun (state: ^state) ->
//            let inline run v x =
//                (^layer_type: (member Run: ^a -> (^state -> ^b * ^state)) v, x)
//            match t.node with
//            | Initialized v ->
//                run v input state
//            | Uninitialized(desired_hidden_size,create_layer) ->
//                let v = create_layer desired_hidden_size input state
//                t.node <- Initialized v
//                run v input state
//
//        member inline t.ToArray =
//            let inline toArray v =
//                (^layer_type: (member ToArray: ^a[]) v)
//            toArray t.node.Value
//
//        member inline t.UpdateLayer (update_function : ^a -> ^state -> unit, state: ^state) =
//            (^layer_type: (member Update: (^a -> ^state -> unit) * ^state -> unit) t.node.Value, update_function, state)
//
//        static member inline create(desired_hidden_size: int, create_layer: int -> 'a -> ^state -> 'layer_type) =
//            {node = Uninitialized(desired_hidden_size,create_layer)}
//
//    let inline FFReluLayer hidden_size = Layer.create(hidden_size,FFRec.createReluRandomLayer)
//    let inline FFSigmoidLayer hidden_size = Layer.create(hidden_size,FFRec.createSigmoidRandomLayer)
//    let inline FFClippedSigmoidLayer hidden_size = Layer.create(hidden_size,FFRec.createClippedSigmoidRandomLayer)
//
//    let inline runLayer layer input =
//        (^layer_type: (member RunLayer: ^b -> (^state -> ^b * ^state)) layer, input)
//
//    let inline updateLayer update_function state layer =
//        (^layer_type: (member UpdateLayer: (^a -> ^state -> unit) * ^state -> unit) layer, update_function, state)
//
//    let inline (>=>) a b = fun x -> a x >>= b
//
//    // For wavefront iteration.
//    // From https://docs.microsoft.com/en-us/dotnet/articles/fsharp/language-reference/computation-expressions
//    /// Computations that can be run step by step.
//    type Eventually<'T> =
//        | Done of 'T
//        | NotYetDone of (unit -> Eventually<'T>)
//
//        static member doBind (expr: Eventually<'a>) (func: 'a -> Eventually<'b>) =
//            match expr with
//            | Done value -> NotYetDone (fun () -> func value)
//            | NotYetDone work -> NotYetDone (fun () -> Eventually<_>.doBind (work()) func)
//
//        static member doReturn expr = Done expr
//
//        // The stepping action for the computations.
//        static member step expr =
//            match expr with
//            | Done _ -> expr
//            | NotYetDone func -> func ()
//
//    /// Generic function for training and inference.
//    let inline private run 
//            (cost: ^a -> ^a -> ^state -> ^m) 
//            (extractor: ^m -> (Lazy<int> * Df) * ^state)
//            (update: ^state -> unit)
//            (set : (^a * ^a)[]) (state: ^state) =
//        let mutable accumulated_cost = 0.0f
//        let mutable accuracy = 0
//        for target,input in set do
//            (mem state).Reset()
//            let (hits,r),_ = cost target input state |> extractor
//            accumulated_cost <- accumulated_cost + r.P.Value.Value
//
//            if is_inference_only state = true then
//                if (tape state).Count > 0 then
//                    failwith "Forgot to use the is_inference_only flag in a library function somewhere"
//                accuracy <- accuracy + hits.Value
//            else
//                r.A := 1.0f
//                let tape = tape state
//                while tape.Count > 0 do
//                    tape.Pop()()
//                update state
//            
//        accuracy, accumulated_cost / float32 set.Length
//    
//    /// Trains the network in the depth direction.
//    let inline train cost update set state = run cost id update set (with_is_inference_only state false) |> snd
//    /// Runs the network without doing backprop.
//    let inline infer cost set state = run cost id (fun _ -> ()) set (with_is_inference_only state true)
//    /// Generic extractor used by train' and infer' functions.
//    let rec extract_eventually =
//        function
//        | Done x -> x
//        | NotYetDone x -> x() |> extract_eventually
//    /// Trains the delayed network in the depth direction.
//    let inline train' cost update set state = run cost extract_eventually update set (with_is_inference_only state false) |> snd
//    /// Runs the delayed network without doing backprop.
//    let inline infer' cost set state = run cost extract_eventually (fun _ -> ()) set (with_is_inference_only state true)
