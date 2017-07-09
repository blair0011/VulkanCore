using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using VulkanCore;
using VulkanCore.Ext;

namespace VulkanMinimalCompute
{
    public class ComputeApplication
    {
        const int WIDTH = 3200; // Size of rendered mandelbrot set.
        const int HEIGHT = 2400; // Size of renderered mandelbrot set.
        const int WORKGROUP_SIZE = 32; // Workgroup size in compute shader.
#if !DEBUG
        const bool enableValidationLayers = false;
#else
        const bool enableValidationLayers = true;
#endif

        // The pixels of the rendered mandelbrot set are in this format:
        private struct Pixel
        {
            public float r, g, b, a;
        };

        /*
        In order to use Vulkan, you must create an instance. 
        */
        Instance instance;

        DebugReportCallbackExt debugReportCallback;
        /*
        The physical device is some device on the system that supports usage of Vulkan.
        Often, it is simply a graphics card that supports Vulkan. 
        */
        PhysicalDevice physicalDevice;
        /*
        Then we have the logical device VkDevice, which basically allows 
        us to interact with the physical device. 
        */
        Device device;

        int computeQueueFamilyIndex;

        /*
        The pipeline specifies the pipeline that all graphics and compute commands pass though in Vulkan.
        We will be creating a simple compute pipeline in this application. 
        */
        Pipeline[] pipelines;
        PipelineLayout pipelineLayout;
        ShaderModule computeShaderModule;

        /*
        The command buffer is used to record commands, that will be submitted to a queue.
        To allocate such command buffers, we use a command pool.
        */
        CommandPool commandPool;
        CommandBuffer[] commandBuffers;

        /*
        Descriptors represent resources in shaders. They allow us to use things like
        uniform buffers, storage buffers and images in GLSL. 
        A single descriptor represents a single resource, and several descriptors are organized
        into descriptor sets, which are basically just collections of descriptors.
        */
        DescriptorPool descriptorPool;
        DescriptorSet descriptorSet;
        DescriptorSetLayout descriptorSetLayout;

        /*
        The mandelbrot set will be rendered to this buffer.
        The memory that backs the buffer is bufferMemory. 
        */
        VulkanCore.Buffer buffer;
        DeviceMemory bufferMemory;

        int bufferSize; // size of `buffer` in bytes.

        List<string> enabledLayers = new List<string>();

        /*
        In order to execute commands on a device(GPU), the commands must be submitted
        to a queue. The commands are stored in a command buffer, and this command buffer
        is given to the queue. 
        There will be different kinds of queues on the device. Not all queues support
        graphics operations, for instance. For this application, we at least want a queue
        that supports compute operations. 
        */
        Queue queue; // a queue supporting compute operations.

        /*
        Groups of queues that have the same capabilities(for instance, they all supports graphics and computer operations),
        are grouped into queue families. 

        When submitting a command buffer, you must specify to which queue in the family you are submitting to. 
        This variable keeps track of the index of that queue in its family. 
        */
        int queueFamilyIndex;

        public void run()
        {
            unsafe
            {
                // Buffer size of the storage buffer that will contain the rendered mandelbrot set.
                bufferSize = sizeof(Pixel) * WIDTH * HEIGHT;
            }

            // Initialize vulkan:
            createInstance();
            findPhysicalDevice();
            createDevice();
            createBuffer();
            createDescriptorSetLayout();
            createDescriptorSet();
            createComputePipeline();
            createCommandBuffer();

            // Finally, run the recorded command buffer.
            runCommandBuffer();

            // The former command rendered a mandelbrot set to a buffer.
            // Save that buffer as a png on disk.
            saveRenderedImage();

            // Clean up all vulkan resources.
            cleanup();
        }

        public unsafe void saveRenderedImage()
        {
            List<byte> image = new List<byte>();
            // Map the buffer memory, so that we can read from it on the CPU.
            IntPtr mappedMemory = bufferMemory.Map(0, bufferSize);
            Pixel* pmappedMemory = (Pixel*)mappedMemory.ToPointer();

            // Get the color data from the buffer, and cast it to bytes.
            // We save the data to a vector.
            image.Capacity = (WIDTH * HEIGHT * 4);
            for (int i = 0; i < WIDTH * HEIGHT; i += 1)
            {
                Pixel p = pmappedMemory[i];
                image.Add((byte)(255.0f * (p.r)));
                image.Add((byte)(255.0f * (p.g)));
                image.Add((byte)(255.0f * (p.b)));
                image.Add((byte)(255.0f * (p.a)));
            }
            // Done reading, so unmap.
            bufferMemory.Unmap();

            // Now we save the acquired color data to a .png.
            using(var stream = new MemoryStream(image.ToArray()))
            {
                using (var bmp = new Bitmap(WIDTH, HEIGHT, PixelFormat.Format32bppArgb))
                {
                    BitmapData bmData = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height),
                        ImageLockMode.WriteOnly, bmp.PixelFormat);
                    IntPtr pNative = bmData.Scan0;
                    Marshal.Copy(image.ToArray(), 0, pNative, image.Count);
                    bmp.UnlockBits(bmData);
                    bmp.Save("mandelbrot.png", ImageFormat.Png);
                }
            }
        }

        public void createInstance()
        {
            List<string> enabledExtensions = new List<string>();

            /*
            By enabling validation layers, Vulkan will emit warnings if the API
            is used incorrectly. We shall enable the layer VK_LAYER_LUNARG_standard_validation,
            which is basically a collection of several useful validation layers.
            */
            if (enableValidationLayers)
            {
                /*
                We get all supported layers with vkEnumerateInstanceLayerProperties.
                */
                List<LayerProperties> layerProperties = new List<LayerProperties>(Instance.EnumerateLayerProperties());
                List<ExtensionProperties> extensionProperties = new List<ExtensionProperties>(Instance.EnumerateExtensionProperties());

                /*
                And then we simply check if VK_LAYER_LUNARG_standard_validation is among the supported layers.
                */
                bool foundLayer = false;
                foreach (LayerProperties prop in layerProperties)
                {

                    if ("VK_LAYER_LUNARG_standard_validation" == prop.LayerName)
                    {
                        foundLayer = true;
                        break;
                    }

                }

                if (!foundLayer)
                {
                    throw new Exception("Layer VK_LAYER_LUNARG_standard_validation not supported\n");
                }
                enabledLayers.Add("VK_LAYER_LUNARG_standard_validation"); // Alright, we can use this layer.

                /*
                We need to enable an extension named VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
                in order to be able to print the warnings emitted by the validation layer.
                So again, we just check if the extension is among the supported extensions.
                */

                bool foundExtension = false;
                foreach (ExtensionProperties prop in extensionProperties)
                {
                    if (Constant.InstanceExtension.ExtDebugReport == prop.ExtensionName)
                    {
                        foundExtension = true;
                        break;
                    }

                }

                if (!foundExtension)
                {
                    throw new Exception("Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");
                }
                enabledExtensions.Add(Constant.InstanceExtension.ExtDebugReport);
            }

            /*
            Next, we actually create the instance.

            */

            /*
            Contains application info. This is actually not that important.
            The only real important field is apiVersion.
            */
            ApplicationInfo applicationInfo = new ApplicationInfo();
            applicationInfo.ApplicationName = "Hello world app";
            applicationInfo.ApplicationVersion = 0;
            applicationInfo.EngineName = "awesomeengine";
            applicationInfo.EngineVersion = 0;
            applicationInfo.ApiVersion = default(VulkanCore.Version);

            InstanceCreateInfo createInfo = new InstanceCreateInfo();
            createInfo.ApplicationInfo = applicationInfo;

            // Give our desired layers and extensions to vulkan.
            createInfo.EnabledLayerNames = enabledLayers.ToArray();


            /*
            Actually create the instance.
            Having created the instance, we can actually start using vulkan.
            */
            instance = new Instance(createInfo);

            /*
            Register a callback function for the extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME, so that warnings emitted from the validation
            layer are actually printed.
            */
            if (enableValidationLayers)
            {
                // Create and register callback.
#if !DEBUG 
                CreateDebugReportCallback(true);
#else
                CreateDebugReportCallback(false);
#endif
            }

        }

        private DebugReportCallbackExt CreateDebugReportCallback(bool debug)
        {
            if (!debug) return null;

            // Attach debug callback.
            var debugReportCreateInfo = new DebugReportCallbackCreateInfoExt(
                DebugReportFlagsExt.All,
                args =>
                {
                    Debug.WriteLine($"[{args.Flags}][{args.LayerPrefix}] {args.Message}");
                    return args.Flags.HasFlag(DebugReportFlagsExt.Error);
                }
            );
            return instance.CreateDebugReportCallbackExt(debugReportCreateInfo);
        }

        public void findPhysicalDevice()
        {
            /*
            In this function, we find a physical device that can be used with Vulkan.
            */

            /*
            So, first we will list all physical devices on the system with vkEnumeratePhysicalDevices .
            */
            List<PhysicalDevice> physicalDevices = new List<PhysicalDevice>(instance.EnumeratePhysicalDevices());

            /*
            Next, we choose a device that can be used for our purposes. 
            With VkPhysicalDeviceFeatures(), we can retrieve a fine-grained list of physical features supported by the device.
            However, in this demo, we are simply launching a simple compute shader, and there are no 
            special physical features demanded for this task.
            With VkPhysicalDeviceProperties(), we can obtain a list of physical device properties. Most importantly,
            we obtain a list of physical device limitations. For this application, we launch a compute shader,
            and the maximum size of the workgroups and total number of compute shader invocations is limited by the physical device,
            and we should ensure that the limitations named maxComputeWorkGroupCount, maxComputeWorkGroupInvocations and 
            maxComputeWorkGroupSize are not exceeded by our application.  Moreover, we are using a storage buffer in the compute shader,
            and we should ensure that it is not larger than the device can handle, by checking the limitation maxStorageBufferRange. 
            However, in our application, the workgroup size and total number of shader invocations is relatively small, and the storage buffer is
            not that large, and thus a vast majority of devices will be able to handle it. This can be verified by looking at some devices at_
            http://vulkan.gpuinfo.org/
            Therefore, to keep things simple and clean, we will not perform any such checks here, and just pick the first physical
            device in the list. But in a real and serious application, those limitations should certainly be taken into account.
            */
            foreach (PhysicalDevice device in physicalDevices)
            {
                QueueFamilyProperties[] queueFamilyProperties = device.GetQueueFamilyProperties();
                for (int j = 0; j < queueFamilyProperties.Length; j++)
                {
                    if (queueFamilyProperties[j].QueueFlags.HasFlag(Queues.Compute))
                    { // As above stated, we do no feature checks, so just accept.
                        computeQueueFamilyIndex = j;
                        physicalDevice = device;
                        break;
                    }
                }
            }
        }

        public void createDevice()
        {
            /*
            We create the logical device in this function.
            */

            /*
            When creating the device, we also specify what queues it has.
            */
            // Store memory properties of the physical device.
            PhysicalDeviceMemoryProperties MemoryProperties = physicalDevice.GetMemoryProperties();
            PhysicalDeviceFeatures Features = physicalDevice.GetFeatures();
            PhysicalDeviceProperties Properties = physicalDevice.GetProperties();

            // Create a logical device.
            var queueCreateInfos = new DeviceQueueCreateInfo[1];
            queueCreateInfos[0] = new DeviceQueueCreateInfo(computeQueueFamilyIndex, 1, 1.0f);

            var deviceCreateInfo = new DeviceCreateInfo(
                queueCreateInfos,
                new[] { Constant.DeviceExtension.NVExternalMemory },
                Features);
            device = physicalDevice.CreateDevice(deviceCreateInfo);

            // Get queue(s).
            queue = device.GetQueue(computeQueueFamilyIndex);

            // Create command pool(s).
            //commandPool = device.CreateCommandPool(new CommandPoolCreateInfo(computeQueueFamilyIndex));
        }

        // find memory type with desired properties.
        int findMemoryType(int memoryTypeBits, MemoryProperties properties)
        {
            PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.GetMemoryProperties();
            return memoryProperties.MemoryTypes.IndexOf(memoryTypeBits, properties);
        }

        void createBuffer()
        {
            /*
            We will now create a buffer. We will render the mandelbrot set into this buffer
            in a computer shade later. 
            */

            BufferCreateInfo bufferCreateInfo = new BufferCreateInfo();
            bufferCreateInfo.Size = bufferSize; // buffer size in bytes. 
            bufferCreateInfo.Usage = BufferUsages.StorageBuffer;// VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
            bufferCreateInfo.SharingMode = SharingMode.Exclusive;// VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

            buffer = device.CreateBuffer(bufferCreateInfo);

            /*
            But the buffer doesn't allocate memory for itself, so we must do that manually.
            */

            /*
            First, we find the memory requirements for the buffer.
            */
            MemoryRequirements memoryRequirements = buffer.GetMemoryRequirements();

            /*
            Now use obtained memory requirements info to allocate the memory for the buffer.
            */
            MemoryAllocateInfo allocateInfo = new MemoryAllocateInfo();
            allocateInfo.AllocationSize = memoryRequirements.Size; // specify required memory.
            /*
            There are several types of memory that can be allocated, and we must choose a memory type that:
            1) Satisfies the memory requirements(memoryRequirements.memoryTypeBits). 
            2) Satifies our own usage requirements. We want to be able to read the buffer memory from the GPU to the CPU
                with vkMapMemory, so we set VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT. 
            Also, by setting VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory written by the device(GPU) will be easily 
            visible to the host(CPU), without having to call any extra flushing commands. So mainly for convenience, we set
            this flag.
            */
            allocateInfo.MemoryTypeIndex = findMemoryType(
                memoryRequirements.MemoryTypeBits, MemoryProperties.HostCoherent | MemoryProperties.HostVisible);// VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

            bufferMemory = device.AllocateMemory(allocateInfo); // allocate memory on device.

            // Now associate that allocated memory with the buffer. With that, the buffer is backed by actual memory. 
            buffer.BindMemory(bufferMemory);
        }

        void createDescriptorSetLayout()
        {
            /*
            Here we specify a descriptor set layout. This allows us to bind our descriptors to 
            resources in the shader. 
            */

            /*
            Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point
            0. This binds to 
              layout(std140, binding = 0) buffer buf
            in the compute shader.
            */
            DescriptorSetLayoutBinding descriptorSetLayoutBinding = new DescriptorSetLayoutBinding();
            descriptorSetLayoutBinding.Binding = 0; // binding = 0
            descriptorSetLayoutBinding.DescriptorType = DescriptorType.StorageBuffer;// VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorSetLayoutBinding.DescriptorCount = 1;
            descriptorSetLayoutBinding.StageFlags = ShaderStages.Compute;// VK_SHADER_STAGE_COMPUTE_BIT;

            DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = new DescriptorSetLayoutCreateInfo();
            // descriptorSetLayoutCreateInfo.bindingCount = 1; // only a single binding in this descriptor set layout. 
            DescriptorSetLayoutBinding[] temp = { descriptorSetLayoutBinding };
            descriptorSetLayoutCreateInfo.Bindings = temp;

            // Create the descriptor set layout. 
            descriptorSetLayout = device.CreateDescriptorSetLayout(descriptorSetLayoutCreateInfo);
        }

        void createDescriptorSet()
        {
            /*
            So we will allocate a descriptor set here.
            But we need to first create a descriptor pool to do that. 
            */

            /*
            Our descriptor pool can only allocate a single storage buffer.
            */
            DescriptorPoolSize descriptorPoolSize = new DescriptorPoolSize();
            descriptorPoolSize.Type = DescriptorType.StorageBuffer;// VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorPoolSize.DescriptorCount = 1;

            DescriptorPoolCreateInfo descriptorPoolCreateInfo = new DescriptorPoolCreateInfo();
            descriptorPoolCreateInfo.MaxSets = 1; // we only need to allocate one descriptor set from the pool.
            descriptorPoolCreateInfo.Flags = DescriptorPoolCreateFlags.FreeDescriptorSet;
            descriptorPoolCreateInfo.PoolSizes = new[] { descriptorPoolSize };

            // create descriptor pool.
            descriptorPool = device.CreateDescriptorPool(descriptorPoolCreateInfo);

            /*
            With the pool allocated, we can now allocate the descriptor set. 
            */
            DescriptorSetAllocateInfo descriptorSetAllocateInfo = new DescriptorSetAllocateInfo(1, descriptorSetLayout);
            // allocate a single descriptor set.

            // allocate descriptor set.
            descriptorSet = descriptorPool.AllocateSets(descriptorSetAllocateInfo)[0];

            /*
            Next, we need to connect our actual storage buffer with the descrptor. 
            We use vkUpdateDescriptorSets() to update the descriptor set.
            */

            // Specify the buffer to bind to the descriptor.
            DescriptorBufferInfo descriptorBufferInfo = new DescriptorBufferInfo();
            descriptorBufferInfo.Buffer = buffer;
            descriptorBufferInfo.Offset = 0;
            descriptorBufferInfo.Range = bufferSize;

            WriteDescriptorSet writeDescriptorSet = new WriteDescriptorSet();
            writeDescriptorSet.DstSet = descriptorSet; // write to this descriptor set.
            writeDescriptorSet.DstBinding = 0; // write to the first, and only binding.
            writeDescriptorSet.DstArrayElement = 0;
            writeDescriptorSet.DescriptorCount = 1; // update a single descriptor.
            writeDescriptorSet.DescriptorType = DescriptorType.StorageBuffer;// VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
            writeDescriptorSet.BufferInfo = new[] { descriptorBufferInfo };

            // perform the update of the descriptor set.
            WriteDescriptorSet[] k = { writeDescriptorSet };
            descriptorPool.UpdateSets(k);
        }

        // Read file into array of bytes, and cast to uint32_t*, then return.
        // The data has been padded, so that it fits into an array uint32_t.
        byte[] readFile(string filename)
        {
            byte[] bytes;
            const int defaultBufferSize = 4096;
            using (Stream stream = new FileStream(filename, FileMode.Open, FileAccess.Read))
            {
                using (var ms = new MemoryStream())
                {
                    stream.CopyTo(ms, defaultBufferSize);
                    bytes = ms.ToArray();
                }
            }
            return bytes;
        }

        void createComputePipeline()
        {
            /*
            We create a compute pipeline here. 
            */

            /*
            Create a shader module. A shader module basically just encapsulates some shader code.
            */
            // the code in comp.spv was created by running the command:
            // glslangValidator.exe -V shader.comp
            byte[] code = readFile("shaders/shader.comp.spv");
            ShaderModuleCreateInfo createInfo = new ShaderModuleCreateInfo();
            createInfo.Code = code;

            computeShaderModule = device.CreateShaderModule(createInfo);

            /*
            Now let us actually create the compute pipeline.
            A compute pipeline is very simple compared to a graphics pipeline.
            It only consists of a single stage with a compute shader. 
            So first we specify the compute shader stage, and it's entry point(main).
            */
            PipelineShaderStageCreateInfo shaderStageCreateInfo = new PipelineShaderStageCreateInfo();
            shaderStageCreateInfo.Stage = ShaderStages.Compute;// VK_SHADER_STAGE_COMPUTE_BIT;
            shaderStageCreateInfo.Module = computeShaderModule;
            shaderStageCreateInfo.Name = "main";

            /*
            The pipeline layout allows the pipeline to access descriptor sets. 
            So we just specify the descriptor set layout we created earlier.
            */
            PipelineLayoutCreateInfo pipelineLayoutCreateInfo = new PipelineLayoutCreateInfo(new[] { descriptorSetLayout });
            pipelineLayout = device.CreatePipelineLayout(pipelineLayoutCreateInfo);

            ComputePipelineCreateInfo pipelineCreateInfo = new ComputePipelineCreateInfo();
            pipelineCreateInfo.Stage = shaderStageCreateInfo;
            pipelineCreateInfo.Layout = pipelineLayout;

            /*
            Now, we finally create the compute pipeline. 
            */
            ComputePipelineCreateInfo[] ci = { pipelineCreateInfo };
            pipelines = device.CreateComputePipelines(ci);
        }

        void createCommandBuffer()
        {
            /*
            We are getting closer to the end. In order to send commands to the device(GPU),
            we must first record commands into a command buffer.
            To allocate a command buffer, we must first create a command pool. So let us do that.
            */
            CommandPoolCreateInfo commandPoolCreateInfo = new CommandPoolCreateInfo();
            commandPoolCreateInfo.Flags = 0;
            // the queue family of this command pool. All command buffers allocated from this command pool,
            // must be submitted to queues of this family ONLY. 
            commandPoolCreateInfo.QueueFamilyIndex = queueFamilyIndex;
            commandPool = device.CreateCommandPool(commandPoolCreateInfo);

            /*
            Now allocate a command buffer from the command pool. 
            */
            CommandBufferAllocateInfo commandBufferAllocateInfo = new CommandBufferAllocateInfo();
            //commandBufferAllocateInfo.commandPool = commandPool; // specify the command pool to allocate from. 
            // if the command buffer is primary, it can be directly submitted to queues. 
            // A secondary buffer has to be called from some primary command buffer, and cannot be directly 
            // submitted to a queue. To keep things simple, we use a primary command buffer. 
            commandBufferAllocateInfo.Level = CommandBufferLevel.Primary;// VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            commandBufferAllocateInfo.CommandBufferCount = 1; // allocate a single command buffer. 
            commandBuffers = commandPool.AllocateBuffers(commandBufferAllocateInfo); // allocate command buffer.

            /*
            Now we shall start recording commands into the newly allocated command buffer. 
            */
            CommandBufferBeginInfo beginInfo = new CommandBufferBeginInfo();
            beginInfo.Flags = CommandBufferUsages.OneTimeSubmit;// VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // the buffer is only submitted and used once in this application.
            commandBuffers[0].Begin(beginInfo); // start recording commands.

            /*
            We need to bind a pipeline, AND a descriptor set before we dispatch.
            The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
            */
            commandBuffers[0].CmdBindPipeline(PipelineBindPoint.Compute, pipelines[0]);// VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            commandBuffers[0].CmdBindDescriptorSets(PipelineBindPoint.Compute, pipelineLayout, 0, new[] { descriptorSet });// VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);

            /*
            Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
            The number of workgroups is specified in the arguments.
            If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
            */
            commandBuffers[0].CmdDispatch((int)Math.Ceiling((double)(WIDTH / WORKGROUP_SIZE)), (int)Math.Ceiling((double)(HEIGHT / WORKGROUP_SIZE)), 1);

            commandBuffers[0].End(); // end recording commands.
        }

        void runCommandBuffer()
        {
            /*
            Now we shall finally submit the recorded command buffer to a queue.
            */

            SubmitInfo submitInfo = new SubmitInfo();
            submitInfo.CommandBuffers = commandBuffers.ToHandleArray(); // the command buffer to submit.

            /*
              We create a fence.
            */
            FenceCreateInfo fenceCreateInfo = new FenceCreateInfo();
            fenceCreateInfo.Flags = 0;
            Fence fence = device.CreateFence(fenceCreateInfo);

            /*
            We submit the command buffer on the queue, at the same time giving a fence.
            */
            queue.Submit(submitInfo, fence);
            /*
            The command will not have finished executing until the fence is signalled.
            So we wait here.
            We will directly after this read our buffer from the GPU,
            and we will not be sure that the command has finished executing unless we wait for the fence.
            Hence, we use a fence here.
            */
            fence.Wait();

            fence.Dispose();
        }

        public void cleanup()
        {
            /*
            Clean up all Vulkan Resources. 
            */
            bufferMemory.Dispose();
            buffer.Dispose();
            computeShaderModule.Dispose();
            descriptorPool.Dispose();
            descriptorSetLayout.Dispose();           
            pipelineLayout.Dispose();
            foreach (Pipeline pipeline in pipelines)
                pipeline.Dispose();
            commandPool.Dispose();
            device.Dispose();
            instance.Dispose();
        }
    }
}
