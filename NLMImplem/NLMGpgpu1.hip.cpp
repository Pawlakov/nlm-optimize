#include "NLMGpgpu1.hip.h"

__device__ float fiL2FloatDist(float* input, int x0, int y0, int x1, int y1, int windowRadius, int channels, int width, int channelLength)
{
	float sum = 0.0f;
	for (int z = 0; z < channels; z++)
	{
		for (int y = -windowRadius; y <= windowRadius; y++)
		{
			for (int x = -windowRadius; x <= windowRadius; x++)
			{
				int l0 = (y0 + y) * width + x0 + x;
				int l1 = (y1 + y) * width + x1 + x;
				float difference = (input[z * channelLength + l0] - input[z * channelLength + l1]);
				sum += (difference * difference);
			}
		}
	}

	return sum;
}

__global__ void getWeightsKernel(float* weights, int windowRadius, int blockRadius, float sigma, float filteringParam, float* input, int channels, int width, int height)
{
	int xy = threadIdx.x + blockIdx.x * blockDim.x;
	int x = xy % width;
	int y = xy / width;

	if (x < width && y < height)
	{
		int channelLength = width * height;

		int windowLength1D = 2 * windowRadius + 1;
		int windowLength2D = windowLength1D * windowLength1D;
		int windowLength3D = channels * windowLength2D;

		int blockLength1D = 2 * blockRadius + 1;
		int blockLength2D = blockLength1D * blockLength1D;

		int windowRadiusReduced = MIN(windowRadius, MIN(width - 1 - x, MIN(height - 1 - y, MIN(x, y))));

		int blockXMin = MAX(x - blockRadius, windowRadiusReduced);
		int blockYMin = MAX(y - blockRadius, windowRadiusReduced);

		int blockXMax = MIN(x + blockRadius, width - 1 - windowRadiusReduced);
		int blockYMax = MIN(y + blockRadius, height - 1 - windowRadiusReduced);

		for (int blockXYRelative = 0; blockXYRelative < blockLength2D; blockXYRelative++)
		{
			weights[xy * blockLength2D + blockXYRelative] = 0.0f;
		}

		for (int blockY = blockYMin; blockY <= blockYMax; blockY++)
		{
			int blockYRelative = blockY - y + blockRadius;
			for (int blockX = blockXMin; blockX <= blockXMax; blockX++)
			{
				int blockXRelative = blockX - x + blockRadius;
				int blockXYRelative = blockYRelative * blockLength1D + blockXRelative;
				float difference = fiL2FloatDist(input, x, y, blockX, blockY, windowRadiusReduced, channels, width, channelLength);

				difference = MAX(difference - 2.0f * windowLength3D * sigma * sigma, 0.0f);
				difference = difference / (filteringParam * filteringParam * sigma * sigma * windowLength3D);

				float weight = expf(-difference);

				weights[xy * blockLength2D + blockXYRelative] = weight;
			}
		}
	}
}

__global__ void filterKernel(int windowRadius, int blockRadius, float sigma, float filteringParam, float* input, float* output, int channels, int width, int height, float* weights)
{
	int outputXY = threadIdx.x + blockIdx.x * blockDim.x;

	if (outputXY < width * height)
	{
		int count = 0;

		int channelLength = width * height;

		int blockLength1D = 2 * blockRadius + 1;
		int blockLength2D = blockLength1D * blockLength1D;

		for (int z = 0; z < channels; z++)
		{
			output[z * channelLength + outputXY] = 0.0f;
		}

		for (int windowYOffset = -windowRadius; windowYOffset <= windowRadius; windowYOffset++)
		{
			for (int windowXOffset = -windowRadius; windowXOffset <= windowRadius; windowXOffset++)
			{
				int inputXY = outputXY - windowXOffset - (windowYOffset * width);
				int inputX = inputXY % width;
				int inputY = inputXY / width;

				int windowRadiusReduced = MIN(windowRadius, MIN(width - 1 - inputX, MIN(height - 1 - inputY, MIN(inputX, inputY))));

				if (inputXY > 0 && inputXY < channelLength)
				{
					float totalWeight = 0.0f;
					for (int blockXYRelative = 0; blockXYRelative < blockLength2D; blockXYRelative++)
					{
						totalWeight += weights[inputXY * blockLength2D + blockXYRelative];
					}

					if (totalWeight > fTiny && abs(windowYOffset) <= windowRadiusReduced && abs(windowXOffset) <= windowRadiusReduced)
					{
						int blockXMin = MAX(inputX - blockRadius, windowRadiusReduced);
						int blockJMin = MAX(inputY - blockRadius, windowRadiusReduced);

						int blockXMax = MIN(inputX + blockRadius, width - 1 - windowRadiusReduced);
						int blockYMax = MIN(inputY + blockRadius, height - 1 - windowRadiusReduced);

						for (int blockY = blockJMin; blockY <= blockYMax; blockY++)
						{
							int blockYRelative = blockY - inputY + blockRadius;
							for (int blockX = blockXMin; blockX <= blockXMax; blockX++)
							{
								int blockXRelative = blockX - inputX + blockRadius;
								int blockXYRelative = blockYRelative * blockLength1D + blockXRelative;
								float weight = weights[inputXY * blockLength2D + blockXYRelative];

								int inputValueXY = (blockY + windowYOffset) * width + blockX + windowXOffset;

								if (blockX == inputX && blockY == inputY)
								{
									count++;
								}

								for (int z = 0; z < channels; z++)
								{
									output[z * channelLength + outputXY] += (weight * input[z * channelLength + inputValueXY] / totalWeight);
								}
							}
						}
					}
				}
			}
		}

		if (count > 0)
		{
			for (int z = 0; z < channels; z++)
			{
				output[z * channelLength + outputXY] /= count;
			}
		}
		else
		{
			for (int z = 0; z < channels; z++)
			{
				output[z * channelLength + outputXY] = input[z * channelLength + outputXY];
			}
		}
	}
}

extern void Denoise(int windowRadius, int blockRadius, float sigma, float fFiltPar, float** input, float** output, int channels, int width, int height)
{
	int blockLength1D = 2 * blockRadius + 1;
	int blockLength2D = blockLength1D * blockLength1D;

	int channelLength = width * height;

	hipStream_t stream;
	hipStreamCreate(&stream);

	dim3 gpuThreads(256, 1, 1);
    dim3 gpuBlocks((channelLength + 256 - 1) / 256, 1, 1);

	size_t channelBytes = channelLength * sizeof(float);

    float* inputDevice;
	float* outputDevice;
	float* weightsDevice;
    HIP_CHECK(hipMalloc(&inputDevice, channels * channelBytes));
	HIP_CHECK(hipMalloc(&outputDevice, channels * channelBytes));
	HIP_CHECK(hipMalloc(&weightsDevice, blockLength2D * channelBytes));
    for (int i = 0; i < channels; i++)
    {
        HIP_CHECK(hipMemcpyAsync(inputDevice + i * channelLength, input[i], channelBytes, hipMemcpyHostToDevice, stream));
    }

    hipLaunchKernelGGL(getWeightsKernel, gpuBlocks, gpuThreads, 0, stream, weightsDevice, windowRadius, blockRadius, sigma, fFiltPar, inputDevice, channels, width, height);
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(filterKernel, gpuBlocks, gpuThreads, 0, stream, windowRadius, blockRadius, sigma, fFiltPar, inputDevice, outputDevice, channels, width, height, weightsDevice);
    HIP_CHECK(hipGetLastError());

	for (int i = 0; i < channels; i++)
    {
        HIP_CHECK(hipMemcpyAsync(output[i], outputDevice + i * channelLength, channelBytes, hipMemcpyDeviceToHost, stream));
    }

    HIP_CHECK(hipFree(inputDevice));
	HIP_CHECK(hipFree(outputDevice));
	HIP_CHECK(hipFree(weightsDevice));

	hipStreamDestroy(stream);
}