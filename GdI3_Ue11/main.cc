#include <iostream>

#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cerrno>
#include <math.h>

#include "cl_1.2.hpp"
#include "timer.h"


void matrixMulCPU( float* a, float* b, float* c, unsigned int widthA, unsigned int heightA, unsigned int widthB)
{
	unsigned int i,j,k;
	for (i = 0; i < heightA; i++) {
		for (j = 0; j < widthB; j++) {
			c[i*widthB + j] = 0;
			for (k = 0; k < widthA; k++) {
				c[i*widthB + j] += a[i*widthA + k] * b[k*widthB + j];
			}
		}
	}
}


void
read_file (std::string const& filename, std::string* data)
{
    std::ifstream in(filename.c_str(), std::ios::binary);
    if (!in)
        throw std::runtime_error(::strerror(errno));
    
    in.seekg(0, std::ios::end);
    data->resize(in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(&data->at(0), data->size());
    in.close();
}


bool matrixMulGPU(cl::Context const & context, cl::CommandQueue * queue,
                  cl::Kernel * kernel, float* a, float* b, float* c,
                  unsigned int widthA, unsigned int heightA, unsigned int widthB)
{
    unsigned int sizeA = widthA * heightA;
    unsigned int sizeB = widthB * widthA;
    unsigned int sizeC = widthB * heightA;
    
    
    /* create buffers for OpenCL Device */
    cl::Buffer bufferA, bufferB, bufferC;
    try {
        // create input buffers and copy from host
        bufferA = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * sizeA, a);
        bufferB = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * sizeB, b);
        // create output buffer
        bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY,  sizeC * sizeof(float), NULL);
    } catch (...) {
        return false;
    }
    
    /* assign arguments to kernel */
    kernel->setArg(0, bufferC);
    kernel->setArg(1, bufferA);
    kernel->setArg(2, bufferB);
    kernel->setArg(3, widthA);
    kernel->setArg(4, heightA);
    kernel->setArg(5, widthB);

    /* set kernel launch configuration */
    cl::NDRange localWorkSize(8, 8);
    cl::NDRange globalWorkSize(widthB, heightA);
       
    /* launch kernel */
    cl::Event event;
    queue->enqueueNDRangeKernel(*kernel, cl::NDRange(), globalWorkSize, localWorkSize, NULL, &event);
    event.wait();

    /* read result from buffer to host memory */
    queue->enqueueReadBuffer(bufferC, true, 0, sizeC * sizeof(float), c);

    /* finish queue */
    queue->finish();

    return true;
}


cl::Kernel compileKernel(cl::Context const& context,
                         std::vector<cl::Device> const & devices,
                         std::string const& filename,
                         std::string const& kernelname)
{
    std::ifstream kernelFile(filename.c_str());
    if (! kernelFile.is_open())
        throw ("Cannot open source file for compiling!");
    
    std::string source;
    read_file(filename, &source);
    cl::Program::Sources sources(1, std::make_pair(source.c_str(), source.length()));
    cl::Program matrixMulProgramm(context, sources);
    if (matrixMulProgramm.build(devices) != 0)
        std::cout << "Error: " << matrixMulProgramm.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
    cl::Kernel matrixMulKernel(matrixMulProgramm, kernelname.c_str());
    return matrixMulKernel;
}

int main(int argc, const char * argv[])
{
    /* get context for all OpenCL Devices */
    cl::Context context = cl::Context(CL_DEVICE_TYPE_ALL);
    /* check number of devices */
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    if (devices.size() < 1) {
        std::cerr << "No OpenCL devices detected!" << std::endl;
        return 0;
    }
    /* print device info */
    std::cout << "Found " << devices.size() << " OpenCL device(s):" << std::endl;
    for (std::size_t i = 0; i < devices.size(); ++i) {
        std::cout << devices[i].getInfo<CL_DEVICE_VENDOR>() << " ";
        std::cout << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
        std::cout << "   Max work group sizes: ";
        std::cout << devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0] << "x";
        std::cout << devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[1] << "x";
        std::cout << devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[2] << std::endl;
        std::cout << "   Local Memory size: ";
        std::cout << devices[i].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()/1024 << " kb" << std::endl;
        std::cout << "   Global Memory size: ";
        std::cout << devices[i].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()/(1024*1024) << " mb" << std::endl;
    }

    /* compile kernel for all devices */
    std::cout << "Compile OpenCL Kernel ..." << std::endl;
    cl::Kernel matrixMulKernel;
    try {
        matrixMulKernel = compileKernel(context, devices, "matrixMulKernel.cl", "matrixMul");
    } catch (std::exception & e) {
        std::cout << e.what() << std::endl;
        return 0;
    }
    
    /* define matrix dimensions */
    unsigned int widthA = 768;
    unsigned int heightA = 1024;
    unsigned int widthB = 1024;

    unsigned int heightB = widthA;
    unsigned int widthC = widthB;
    unsigned int heightC = heightA;

    std::cout << "Matrices dimensions: A(" << widthA <<" x " << heightA
        << "), B(" << widthB <<" x " << heightB
        << "), C(" << widthC <<" x " << heightC << ")" << std::endl;
    
    /* allocate memory for matrices */
    unsigned int sizeA = widthA * heightA;
    float* hostDataA = new float[sizeA];
    
    unsigned int sizeB = widthB * heightB;
    float* hostDataB = new float[sizeB];
    
    unsigned int sizeC = widthC * heightC;
    float* hostCgpu = new float[sizeC];
    float* hostCcpu = new float[sizeC];
    
    /* initialize matrices with random numbers from the interval [0,1] */
    srand(2006);
    const float fScale = 1.0f / (float)RAND_MAX;
    
    for(size_t i = 0; i < sizeA; ++i)
        hostDataA[i] = fScale * rand();
    for(size_t i = 0; i < sizeB; ++i)
        hostDataB[i] = fScale * rand();
    for(size_t i = 0; i < sizeB; ++i)
        hostCgpu[i] = 0;

    /* create timer */
    WallTimer timer;

    /* get queue for first OpenCL device */
    /* (this will be the GPU if available - otherwise the CPU) */
    std::cout << "Multiplying matrices with OpenCL ... " << std::endl;
    /* generate queue for first device */
    cl::CommandQueue gpuQueue(context, devices[0]);
    
    timer.reset();

    /* start computation */
    if(!matrixMulGPU(context, &gpuQueue, &matrixMulKernel, hostDataA, hostDataB, hostCgpu, widthA, heightA, widthB)) {
        std::cout << "ERROR: executing OpenCL matrix multiplication failed..." << std::endl;
        return 1;
    }
    std::cout << "Finished multiplying matrices with OpenCL" << std::endl;
    float openCLTime = timer.get_elapsed_sec();
    std::cout << "Time: " << openCLTime << " s" << std::endl;
    
    std::cout << "Multiplying matrices on CPU ... " << std::endl;

    timer.reset();

    /* start computation */
    matrixMulCPU(hostDataA, hostDataB, hostCcpu, widthA, heightA, widthB);
    std::cout << "Finished multiplying matrices on CPU" << std::endl;
    float cpuTime = timer.get_elapsed_sec();
    std::cout << "Time: " << cpuTime << " s" << std::endl;
    
    std::cout << "Speedup: " << openCLTime/cpuTime << std::endl;
    
    std::cout << "Verifying correctness... ";
    
	 float max_delta = 0, delta;

	 for (size_t i = 0; i < sizeC; i++) {
		 delta = abs(hostCgpu[i] - hostCcpu[i]);
		 if (delta > max_delta)
			 max_delta = delta;
	 }
	 std::cout << "delta = " << max_delta << std::endl;
    
    delete[] hostDataA;
    delete[] hostDataB;
    delete[] hostCcpu;
    delete[] hostCgpu;
    return 0;
}
