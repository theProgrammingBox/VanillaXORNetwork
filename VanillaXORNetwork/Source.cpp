#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <fstream>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::microseconds;
using std::exp;
using std::sqrt;
using std::max;
using std::min;
using std::ofstream;
using std::ios;

/*
VANILLA IMPORTANT LESSONS
1. Leaky relu is the best compared to tahn and relu
2. Having a large batch sizes allows for more clearly defined patterns when graphing the scores of multiple runs
3. Normalizing then clamping the gradients allows for larger / smaller learning rates while preventing gradeint explosion / vanishing
*/

class Random
{
public:
	Random(uint32_t seed = 0)	// seed the random number generator
	{
		state[0] = Hash((uint8_t*)&seed, 4, seed);
		state[1] = Hash((uint8_t*)&seed, 4, state[0]);
	}

	static uint32_t MakeSeed(uint32_t seed = 0)	// make seed from time and seed
	{
		uint32_t result = seed;
		result = Hash((uint8_t*)&result, 4, nanosecond());
		result = Hash((uint8_t*)&result, 4, microsecond());
		return result;
	}

	void Seed(uint32_t seed = 0)	// seed the random number generator
	{
		state[0] = Hash((uint8_t*)&seed, 4, seed);
		state[1] = Hash((uint8_t*)&seed, 4, state[0]);
	}

	uint32_t Ruint32()	// XORSHIFT128+
	{
		uint64_t a = state[0];
		uint64_t b = state[1];
		state[0] = b;
		a ^= a << 23;
		state[1] = a ^ b ^ (a >> 18) ^ (b >> 5);
		return uint32_t((state[1] + b) >> 16);
	}

	float Rfloat(float min = 0, float max = 1) { return min + (max - min) * Ruint32() * 2.3283064371e-10; }

	static uint32_t Hash(const uint8_t* key, size_t len, uint32_t seed = 0)	// MurmurHash3
	{
		uint32_t h = seed;
		uint32_t k;
		for (size_t i = len >> 2; i; i--) {
			memcpy(&k, key, 4);
			key += 4;
			h ^= murmur_32_scramble(k);
			h = (h << 13) | (h >> 19);
			h = h * 5 + 0xe6546b64;
		}
		k = 0;
		for (size_t i = len & 3; i; i--) {
			k <<= 8;
			k |= key[i - 1];
		}
		h ^= murmur_32_scramble(k);
		h ^= len;
		h ^= h >> 16;
		h *= 0x85ebca6b;
		h ^= h >> 13;
		h *= 0xc2b2ae35;
		h ^= h >> 16;
		return h;
	}

private:
	uint64_t state[2];

	static uint32_t murmur_32_scramble(uint32_t k) {
		k *= 0xcc9e2d51;
		k = (k << 15) | (k >> 17);
		k *= 0x1b873593;
		return k;
	}

	static uint32_t nanosecond() { return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count(); }
	static uint32_t microsecond() { return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count(); }
};

namespace GLOBAL
{
	Random random(Random::MakeSeed(0));
	constexpr uint32_t INPUT = 2;
	constexpr uint32_t HIDDEN = 2;
	constexpr uint32_t OUTPUT = 2;
	constexpr uint32_t ITERATIONS = 1900;
	constexpr uint32_t BATCHES = 32;
	constexpr uint32_t ACTIVATIONS = 2;
	constexpr uint32_t RUNS = 100;
	constexpr uint32_t AVERAGES = 100;
	constexpr float ONEF = 1.0f;
	constexpr float ZEROF = 0.0f;
	constexpr float LEARNING_RATE = 1.0f;
	float GRADIENT_SCALAR = LEARNING_RATE / sqrt(BATCHES);
}

void cpuGenerateUniform(float* matrix, uint32_t size, float min = 0, float max = 1)
{
	for (uint32_t counter = size; counter--;)
		matrix[counter] = GLOBAL::random.Rfloat(min, max);
}

void cpuSgemmStridedBatched(
	bool transB, bool transA,
	int CCols, int CRows, int AColsBRows,
	const float* alpha,
	float* B, int ColsB, int SizeB,
	float* A, int ColsA, int SizeA,
	const float* beta,
	float* C, int ColsC, int SizeC,
	int batchCount)
{
	for (int b = batchCount; b--;)
	{
		for (int m = CCols; m--;)
			for (int n = CRows; n--;)
			{
				float sum = 0;
				for (int k = AColsBRows; k--;)
					sum += (transA ? A[k * ColsA + n] : A[n * ColsA + k]) * (transB ? B[m * ColsB + k] : B[k * ColsB + m]);
				C[n * ColsC + m] = *alpha * sum + *beta * C[n * ColsC + m];
			}
		A += SizeA;
		B += SizeB;
		C += SizeC;
	}
}

void cpuSaxpy(int N, const float* alpha, const float* X, int incX, float* Y, int incY)
{
	for (int i = N; i--;)
		Y[i * incY] += *alpha * X[i * incX];
}

void cpuLeakyRelu(float* input, float* output, uint32_t size)
{
	for (size_t counter = size; counter--;)
		output[counter] = ((~(*(int32_t*)(input + counter) & 0x80000000) >> 31) * 0.9f + 0.1f) * input[counter];
}

void cpuLeakyReluDerivative(float* input, float* gradient, float* output, uint32_t size)
{
	for (size_t counter = size; counter--;)
		output[counter] = ((~(*(int32_t*)(input + counter) & 0x80000000) >> 31) * 0.9f + 0.1f) * gradient[counter];
}

void cpuLeakyRelu2(float* input, float* output, uint32_t size)
{
	for (size_t counter = size; counter--;)
		output[counter] = (((*(int32_t*)(input + counter) & 0x80000000) >> 31) * 0.9f + 0.1f) * input[counter];
}

void cpuLeakyRelu2Derivative(float* input, float* gradient, float* output, uint32_t size)
{
	for (size_t counter = size; counter--;)
		output[counter] = (((*(int32_t*)(input + counter) & 0x80000000) >> 31) * 0.9f + 0.1f) * gradient[counter];
}

void cpuActivation(float* input, float* gradient, float* output, uint32_t size, uint32_t activation)
{
	switch (activation)
	{
	case 0:
		cpuLeakyRelu(input, output, size);
		break;
	case 1:
		cpuLeakyRelu2(input, output, size);
		break;
	}
}

void cpuActivationDerivative(float* input, float* gradient, float* output, uint32_t size, uint32_t activation)
{
	switch (activation)
	{
	case 0:
		cpuLeakyReluDerivative(input, gradient, output, size);
		break;
	case 1:
		cpuLeakyRelu2Derivative(input, gradient, output, size);
		break;
	}
}

void cpuSoftmax(float* input, float* output, uint32_t size)
{
	float sum = 0;
	for (uint32_t counter = size; counter--;)
	{
		output[counter] = exp(input[counter]);
		sum += output[counter];
	}
	sum = 1.0f / sum;
	for (uint32_t counter = size; counter--;)
		output[counter] *= sum;
}

void cpuSoftmaxDerivative(float* input, float* output, bool endState, uint32_t action, uint32_t size)
{
	float sampledProbability = input[action];
	float gradient = (endState - sampledProbability);
	for (uint32_t counter = size; counter--;)
		output[counter] = gradient * input[counter] * ((counter == action) - sampledProbability);
}

void PrintMatrix(float* arr, uint32_t rows, uint32_t cols, const char* label) {
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", arr[i * cols + j]);
		printf("\n");
	}
	printf("\n");
}

void DiagonalMatrixInit(float* arr, uint32_t rows, uint32_t cols) {
	cpuGenerateUniform(arr, rows * cols, -0.1f, 0.1f);
	//memset(arr, 0, rows * cols << 2);
	uint32_t maxSteps = max(rows, cols);
	float stepx = (float)cols / maxSteps;
	float stepy = (float)rows / maxSteps;
	float x = 0.0f;
	float y = 0.0f;
	for (uint32_t step = maxSteps; step--;)
	{
		arr[uint32_t(y) * cols + uint32_t(x)] += (GLOBAL::random.Ruint32() & 1 << 1) - 1.0f;
		x += stepx;
		y += stepy;
	}
}

int main()
{
	const bool debug = false;

	float scores[GLOBAL::AVERAGES];
	float avgScore;
	uint32_t idx;
	
	float inputMatrix[GLOBAL::INPUT];
	float hiddenMatrix[GLOBAL::HIDDEN];
	float hiddenActivation[GLOBAL::HIDDEN];
	float outputMatrix[GLOBAL::OUTPUT];
	float outputActivation[GLOBAL::OUTPUT];
	float softmaxMatrix[GLOBAL::OUTPUT];
	
	float hiddenWeights[GLOBAL::INPUT * GLOBAL::HIDDEN];
	float outputWeights[GLOBAL::HIDDEN * GLOBAL::OUTPUT];
	float hiddenBias[GLOBAL::HIDDEN];
	float outputBias[GLOBAL::OUTPUT];

	float hiddenGradient[GLOBAL::HIDDEN];
	float hiddenActivationGradient[GLOBAL::HIDDEN];
	float outputGradient[GLOBAL::OUTPUT];
	float outputActivationGradient[GLOBAL::OUTPUT];
	
	float hiddenWeightsGradient[GLOBAL::INPUT * GLOBAL::HIDDEN];
	float outputWeightsGradient[GLOBAL::HIDDEN * GLOBAL::OUTPUT];
	float hiddenBiasGradient[GLOBAL::HIDDEN];
	float outputBiasGradient[GLOBAL::OUTPUT];
	
	ofstream dataFile;
	dataFile.open("data.txt", ios::out | ios::trunc | ios::binary);
	dataFile.write((char*)&GLOBAL::ACTIVATIONS, 4);
	dataFile.write((char*)&GLOBAL::RUNS, 4);
	dataFile.write((char*)&GLOBAL::ITERATIONS, 4);
	
	uint32_t activation = GLOBAL::ACTIVATIONS;
	while (activation--)
	{
		uint32_t run = GLOBAL::RUNS;
		while (run--)
		{
			memset(scores, 0, GLOBAL::AVERAGES << 2);
			avgScore = 0.0f;
			idx = 0;
			
			DiagonalMatrixInit(hiddenWeights, GLOBAL::INPUT, GLOBAL::HIDDEN);
			DiagonalMatrixInit(outputWeights, GLOBAL::HIDDEN, GLOBAL::OUTPUT);
			memset(hiddenBias, 0, GLOBAL::HIDDEN << 2);
			memset(outputBias, 0, GLOBAL::OUTPUT << 2);

			uint32_t iteration = GLOBAL::ITERATIONS;
			while (iteration--)
			{
				memset(hiddenWeightsGradient, 0, GLOBAL::INPUT * GLOBAL::HIDDEN << 2);
				memset(outputWeightsGradient, 0, GLOBAL::HIDDEN * GLOBAL::OUTPUT << 2);
				memset(hiddenBiasGradient, 0, GLOBAL::HIDDEN << 2);
				memset(outputBiasGradient, 0, GLOBAL::OUTPUT << 2);

				float averageScore = 0;
				uint32_t batch = GLOBAL::BATCHES;
				while (batch--)
				{
					bool expected = 0;
					for (uint32_t counter = GLOBAL::INPUT; counter--;)
					{
						inputMatrix[counter] = GLOBAL::random.Ruint32() & 1;
						expected ^= (bool)inputMatrix[counter];
					}
					cpuSgemmStridedBatched(false, false,
						GLOBAL::HIDDEN, GLOBAL::ONEF, GLOBAL::INPUT,
						&GLOBAL::ONEF,
						hiddenWeights, GLOBAL::HIDDEN, GLOBAL::ZEROF,
						inputMatrix, GLOBAL::INPUT, GLOBAL::ZEROF,
						&GLOBAL::ZEROF,
						hiddenMatrix, GLOBAL::HIDDEN, GLOBAL::ZEROF,
						GLOBAL::ONEF);
					cpuSaxpy(GLOBAL::HIDDEN, &GLOBAL::ONEF, hiddenBias, GLOBAL::ONEF, hiddenMatrix, GLOBAL::ONEF);
					cpuActivation(hiddenMatrix, hiddenGradient, hiddenActivation, GLOBAL::HIDDEN, activation);
					cpuSgemmStridedBatched(false, false,
						GLOBAL::OUTPUT, GLOBAL::ONEF, GLOBAL::HIDDEN,
						&GLOBAL::ONEF,
						outputWeights, GLOBAL::OUTPUT, GLOBAL::ZEROF,
						hiddenActivation, GLOBAL::HIDDEN, GLOBAL::ZEROF,
						&GLOBAL::ZEROF,
						outputMatrix, GLOBAL::OUTPUT, GLOBAL::ZEROF,
						GLOBAL::ONEF);
					cpuSaxpy(GLOBAL::OUTPUT, &GLOBAL::ONEF, outputBias, GLOBAL::ONEF, outputMatrix, GLOBAL::ONEF);
					cpuActivation(outputMatrix, outputGradient, outputActivation, GLOBAL::OUTPUT, activation);
					cpuSoftmax(outputActivation, softmaxMatrix, GLOBAL::OUTPUT);

					float number = GLOBAL::random.Rfloat(0.0f, 1.0f);
					uint32_t action = 0;
					for (;;)
					{
						number -= softmaxMatrix[action];
						if (number < 0) break;
						action -= (++action == GLOBAL::OUTPUT) * GLOBAL::OUTPUT;
					}
					bool endState = bool(action) == expected;
					
					cpuSoftmaxDerivative(softmaxMatrix, outputActivationGradient, endState, action, GLOBAL::OUTPUT);
					cpuActivationDerivative(outputMatrix, outputActivationGradient, outputGradient, GLOBAL::OUTPUT, activation);
					cpuSgemmStridedBatched(true, false,
						GLOBAL::HIDDEN, GLOBAL::ONEF, GLOBAL::OUTPUT,
						&GLOBAL::ONEF,
						outputWeights, GLOBAL::OUTPUT, GLOBAL::ZEROF,
						outputGradient, GLOBAL::OUTPUT, GLOBAL::ZEROF,
						&GLOBAL::ZEROF,
						hiddenActivationGradient, GLOBAL::HIDDEN, GLOBAL::ZEROF,
						GLOBAL::ONEF);
					cpuActivationDerivative(hiddenMatrix, hiddenActivationGradient, hiddenGradient, GLOBAL::HIDDEN, activation);

					cpuSgemmStridedBatched(false, true,
						GLOBAL::OUTPUT, GLOBAL::HIDDEN, GLOBAL::ONEF,
						&GLOBAL::ONEF,
						outputGradient, GLOBAL::OUTPUT, GLOBAL::ZEROF,
						hiddenActivation, GLOBAL::HIDDEN, GLOBAL::ZEROF,
						&GLOBAL::ONEF,
						outputWeightsGradient, GLOBAL::OUTPUT, GLOBAL::ZEROF,
						GLOBAL::ONEF);
					cpuSgemmStridedBatched(false, true,
						GLOBAL::HIDDEN, GLOBAL::INPUT, GLOBAL::ONEF,
						&GLOBAL::ONEF,
						hiddenGradient, GLOBAL::HIDDEN, GLOBAL::ZEROF,
						inputMatrix, GLOBAL::INPUT, GLOBAL::ZEROF,
						&GLOBAL::ONEF,
						hiddenWeightsGradient, GLOBAL::HIDDEN, GLOBAL::ZEROF,
						GLOBAL::ONEF);
					cpuSaxpy(GLOBAL::HIDDEN, &GLOBAL::ONEF, hiddenGradient, GLOBAL::ONEF, hiddenBiasGradient, GLOBAL::ONEF);
					cpuSaxpy(GLOBAL::OUTPUT, &GLOBAL::ONEF, outputGradient, GLOBAL::ONEF, outputBiasGradient, GLOBAL::ONEF);
					
					averageScore += endState;
				}
				
				averageScore /= GLOBAL::BATCHES;
				avgScore -= scores[idx];
				avgScore += averageScore;
				scores[idx] = averageScore;
				idx -= (++idx == GLOBAL::AVERAGES) * GLOBAL::AVERAGES;
				averageScore = avgScore / GLOBAL::AVERAGES;
				dataFile.write((char*)&averageScore, 4);

				if (iteration == 0 && debug)
				{
					printf("Average score: %f\n\n", averageScore);
					PrintMatrix(inputMatrix, GLOBAL::ONEF, GLOBAL::INPUT, "inputMatrix");
					PrintMatrix(hiddenMatrix, GLOBAL::ONEF, GLOBAL::HIDDEN, "hiddenMatrix");
					PrintMatrix(hiddenActivation, GLOBAL::ONEF, GLOBAL::HIDDEN, "hiddenActivation");
					PrintMatrix(outputMatrix, GLOBAL::ONEF, GLOBAL::OUTPUT, "outputMatrix");
					PrintMatrix(softmaxMatrix, GLOBAL::ONEF, GLOBAL::OUTPUT, "softmaxMatrix");
					PrintMatrix(outputGradient, GLOBAL::ONEF, GLOBAL::OUTPUT, "outputGradient");
					PrintMatrix(hiddenActivationGradient, GLOBAL::ONEF, GLOBAL::HIDDEN, "hiddenActivationGradient");
					PrintMatrix(hiddenGradient, GLOBAL::ONEF, GLOBAL::HIDDEN, "hiddenGradient");
					PrintMatrix(outputWeightsGradient, GLOBAL::HIDDEN, GLOBAL::OUTPUT, "outputWeightsGradient");
					PrintMatrix(hiddenWeightsGradient, GLOBAL::INPUT, GLOBAL::HIDDEN, "hiddenWeightsGradient");
					PrintMatrix(outputBiasGradient, GLOBAL::ONEF, GLOBAL::OUTPUT, "outputBiasGradient");
					PrintMatrix(hiddenBiasGradient, GLOBAL::ONEF, GLOBAL::HIDDEN, "hiddenBiasGradient");
					PrintMatrix(hiddenWeights, GLOBAL::INPUT, GLOBAL::HIDDEN, "hiddenWeights");
					PrintMatrix(outputWeights, GLOBAL::HIDDEN, GLOBAL::OUTPUT, "outputWeights");
					PrintMatrix(hiddenBias, GLOBAL::ONEF, GLOBAL::HIDDEN, "hiddenBias");
					PrintMatrix(outputBias, GLOBAL::ONEF, GLOBAL::OUTPUT, "outputBias");
				}

				float magnitude = 0;
				for (int i = GLOBAL::INPUT * GLOBAL::HIDDEN; i--;)
					magnitude += hiddenWeightsGradient[i] * hiddenWeightsGradient[i];
				for (int i = GLOBAL::HIDDEN; i--;)
					magnitude += hiddenBiasGradient[i] * hiddenBiasGradient[i];
				for (int i = GLOBAL::HIDDEN * GLOBAL::OUTPUT; i--;)
					magnitude += outputWeightsGradient[i] * outputWeightsGradient[i];
				for (int i = GLOBAL::OUTPUT; i--;)
					magnitude += outputBiasGradient[i] * outputBiasGradient[i];
				magnitude = sqrt(magnitude);
				magnitude = GLOBAL::GRADIENT_SCALAR * min(1.0f, max(0.01f, magnitude)) / magnitude;
				
				cpuSaxpy(GLOBAL::INPUT * GLOBAL::HIDDEN, &magnitude, hiddenWeightsGradient, GLOBAL::ONEF, hiddenWeights, GLOBAL::ONEF);
				cpuSaxpy(GLOBAL::HIDDEN, &magnitude, hiddenBiasGradient, GLOBAL::ONEF, hiddenBias, GLOBAL::ONEF);
				cpuSaxpy(GLOBAL::HIDDEN * GLOBAL::OUTPUT, &magnitude, outputWeightsGradient, GLOBAL::ONEF, outputWeights, GLOBAL::ONEF);
				cpuSaxpy(GLOBAL::OUTPUT, &magnitude, outputBiasGradient, GLOBAL::ONEF, outputBias, GLOBAL::ONEF);
			}
		}
	}
	dataFile.close();

	return 0;
}