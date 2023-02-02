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
using std::cout;
using std::vector;
using std::sort;
using std::exp;
using std::min;
using std::max;
using std::ofstream;

class Random
{
public:
	Random(uint32_t seed = 0)	// seed the random number generator
	{
		state[0] = Hash((uint8_t*)&seed, sizeof(seed), seed);
		state[1] = Hash((uint8_t*)&seed, sizeof(seed), state[0]);
	}

	static uint32_t MakeSeed(uint32_t seed = 0)	// make seed from time and seed
	{
		uint32_t result = seed;
		result = Hash((uint8_t*)&result, sizeof(result), nanosecond());
		result = Hash((uint8_t*)&result, sizeof(result), microsecond());
		return result;
	}

	void Seed(uint32_t seed = 0)	// seed the random number generator
	{
		state[0] = Hash((uint8_t*)&seed, sizeof(seed), seed);
		state[1] = Hash((uint8_t*)&seed, sizeof(seed), state[0]);
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
			memcpy(&k, key, sizeof(uint32_t));
			key += sizeof(uint32_t);
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

namespace GlobalVars
{
	Random random(Random::MakeSeed(0));
	constexpr uint32_t INPUT = 2;
	constexpr uint32_t HIDDEN = 2;
	constexpr uint32_t OUTPUT = 1;
	constexpr uint32_t ITERATIONS = 100000;
	constexpr uint32_t BATCHES = 8;
	constexpr float ONE = 1.0f;
	constexpr float ZERO = 0.0f;
	constexpr float LEARNING_RATE = 0.1f;
	constexpr float GRADIENT_SCALAR = LEARNING_RATE / BATCHES;
}

void cpuGenerateUniform(float* matrix, uint32_t size, float min = 0, float max = 1)
{
	for (uint32_t counter = size; counter--;)
		matrix[counter] = GlobalVars::random.Rfloat(min, max);
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
	for (int i = 0; i < N; i++)
		Y[i * incY] += *alpha * X[i * incX];
}

void cpuRelu(float* input, float* output, uint32_t size)
{
	for (uint32_t counter = size; counter--;)
		output[counter] = ((input[counter] > 0)) * input[counter];
}

void cpuReluDerivative(float* input, float* gradient, float* output, uint32_t size)
{
	for (uint32_t counter = size; counter--;)
		output[counter] = ((input[counter] > 0)) * gradient[counter];
}

void PrintMatrix(float* matrix, uint32_t rows, uint32_t cols)
{
	for (uint32_t row = 0; row < rows; row++)
	{
		for (uint32_t col = 0; col < cols; col++)
			cout << matrix[row * cols + col] << " ";
		cout << '\n';
	}
	cout << '\n';
}

int main()
{
	const bool debug = false;
	
	float inputMatrix[GlobalVars::INPUT];
	float hiddenMatrix[GlobalVars::HIDDEN];
	float hiddenActivation[GlobalVars::HIDDEN];
	float outputMatrix[GlobalVars::OUTPUT];
	float outputActivation[GlobalVars::OUTPUT];
	float hiddenWeights[GlobalVars::INPUT * GlobalVars::HIDDEN];
	float outputWeights[GlobalVars::HIDDEN * GlobalVars::OUTPUT];
	float hiddenBias[GlobalVars::HIDDEN];
	float outputBias[GlobalVars::OUTPUT];

	float hiddenGradient[GlobalVars::HIDDEN];
	float hiddenActivationGradient[GlobalVars::HIDDEN];
	float outputGradient[GlobalVars::OUTPUT];
	float outputActivationGradient[GlobalVars::OUTPUT];
	float hiddenWeightsGradient[GlobalVars::INPUT * GlobalVars::HIDDEN];
	float outputWeightsGradient[GlobalVars::HIDDEN * GlobalVars::OUTPUT];
	float hiddenBiasGradient[GlobalVars::HIDDEN];
	float outputBiasGradient[GlobalVars::OUTPUT];
	
	cpuGenerateUniform(hiddenWeights, GlobalVars::INPUT * GlobalVars::HIDDEN);
	cpuGenerateUniform(outputWeights, GlobalVars::HIDDEN * GlobalVars::OUTPUT);
	memset(hiddenBias, 0, GlobalVars::HIDDEN * sizeof(float));
	memset(outputBias, 0, GlobalVars::OUTPUT * sizeof(float));

	uint32_t iteration = GlobalVars::ITERATIONS;
	while (iteration--)
	{
		memset(hiddenWeightsGradient, 0, GlobalVars::INPUT * GlobalVars::HIDDEN * sizeof(float));
		memset(outputWeightsGradient, 0, GlobalVars::HIDDEN * GlobalVars::OUTPUT * sizeof(float));
		memset(hiddenBiasGradient, 0, GlobalVars::HIDDEN * sizeof(float));
		memset(outputBiasGradient, 0, GlobalVars::OUTPUT * sizeof(float));

		uint32_t batch = GlobalVars::BATCHES;
		while (batch--)
		{
			bool expected = 0;
			for (uint32_t counter = GlobalVars::INPUT; counter--;)
			{
				inputMatrix[counter] = GlobalVars::random.Ruint32() & 1;
				expected ^= (bool)inputMatrix[counter];
			}
			cpuSgemmStridedBatched(false, false,
				GlobalVars::HIDDEN, GlobalVars::ONE, GlobalVars::INPUT,
				&GlobalVars::ONE,
				hiddenWeights, GlobalVars::HIDDEN, GlobalVars::ZERO,
				inputMatrix, GlobalVars::INPUT, GlobalVars::ZERO,
				&GlobalVars::ZERO,
				hiddenMatrix, GlobalVars::HIDDEN, GlobalVars::ZERO,
				GlobalVars::ONE);
			cpuRelu(hiddenMatrix, hiddenActivation, GlobalVars::HIDDEN);
			cpuSgemmStridedBatched(false, false,
				GlobalVars::OUTPUT, GlobalVars::ONE, GlobalVars::HIDDEN,
				&GlobalVars::ONE,
				outputWeights, GlobalVars::OUTPUT, GlobalVars::ZERO,
				hiddenActivation, GlobalVars::HIDDEN, GlobalVars::ZERO,
				&GlobalVars::ZERO,
				outputMatrix, GlobalVars::OUTPUT, GlobalVars::ZERO,
				GlobalVars::ONE);
			cpuRelu(outputMatrix, outputActivation, GlobalVars::OUTPUT);

			outputActivationGradient[0] = expected - outputActivation[0];
			cpuReluDerivative(outputMatrix, outputActivationGradient, outputGradient, GlobalVars::OUTPUT);
			cpuSgemmStridedBatched(true, false,
				GlobalVars::HIDDEN, GlobalVars::ONE, GlobalVars::OUTPUT,
				&GlobalVars::ONE,
				outputWeights, GlobalVars::OUTPUT, GlobalVars::ZERO,
				outputGradient, GlobalVars::OUTPUT, GlobalVars::ZERO,
				&GlobalVars::ZERO,
				hiddenActivationGradient, GlobalVars::HIDDEN, GlobalVars::ZERO,
				GlobalVars::ONE);
			cpuReluDerivative(hiddenMatrix, hiddenActivationGradient, hiddenGradient, GlobalVars::HIDDEN);

			cpuSgemmStridedBatched(false, true,
				GlobalVars::OUTPUT, GlobalVars::HIDDEN, GlobalVars::ONE,
				&GlobalVars::ONE,
				outputGradient, GlobalVars::OUTPUT, GlobalVars::ZERO,
				hiddenActivation, GlobalVars::HIDDEN, GlobalVars::ZERO,
				&GlobalVars::ONE,
				outputWeightsGradient, GlobalVars::OUTPUT, GlobalVars::ZERO,
				GlobalVars::ONE);
			cpuSgemmStridedBatched(false, true,
				GlobalVars::HIDDEN, GlobalVars::INPUT, GlobalVars::ONE,
				&GlobalVars::ONE,
				hiddenGradient, GlobalVars::HIDDEN, GlobalVars::ZERO,
				inputMatrix, GlobalVars::INPUT, GlobalVars::ZERO,
				&GlobalVars::ONE,
				hiddenWeightsGradient, GlobalVars::HIDDEN, GlobalVars::ZERO,
				GlobalVars::ONE);
			cpuSaxpy(GlobalVars::HIDDEN, &GlobalVars::ONE, hiddenGradient, GlobalVars::ONE, hiddenBiasGradient, GlobalVars::ONE);
			cpuSaxpy(GlobalVars::OUTPUT, &GlobalVars::ONE, outputGradient, GlobalVars::ONE, outputBiasGradient, GlobalVars::ONE);

			if (debug)
			{
				cout << "inputMatrix:\n";
				PrintMatrix(inputMatrix, GlobalVars::ONE, GlobalVars::INPUT);
				cout << "hiddenMatrix:\n";
				PrintMatrix(hiddenMatrix, GlobalVars::ONE, GlobalVars::HIDDEN);
				cout << "hiddenActivation:\n";
				PrintMatrix(hiddenActivation, GlobalVars::ONE, GlobalVars::HIDDEN);
				cout << "outputMatrix:\n";
				PrintMatrix(outputMatrix, GlobalVars::ONE, GlobalVars::OUTPUT);
				cout << "outputActivation:\n";
				PrintMatrix(outputActivation, GlobalVars::ONE, GlobalVars::OUTPUT);
				cout << "outputActivationGradient:\n";
				PrintMatrix(outputActivationGradient, GlobalVars::ONE, GlobalVars::OUTPUT);
				cout << "outputGradient:\n";
				PrintMatrix(outputGradient, GlobalVars::ONE, GlobalVars::OUTPUT);
				cout << "outputWeightsGradient:\n";
				PrintMatrix(outputWeightsGradient, GlobalVars::HIDDEN, GlobalVars::OUTPUT);
				cout << "hiddenActivationGradient:\n";
				PrintMatrix(hiddenActivationGradient, GlobalVars::ONE, GlobalVars::HIDDEN);
				cout << "hiddenGradient:\n";
				PrintMatrix(hiddenGradient, GlobalVars::ONE, GlobalVars::HIDDEN);
				cout << "hiddenWeightsGradient:\n";
				PrintMatrix(hiddenWeightsGradient, GlobalVars::INPUT, GlobalVars::HIDDEN);
			}

			if (iteration < 10)
			{
				cout << "Expected: " << expected << '\n';
				cout << "Output: " << outputActivation[0] << '\n';
				cout << '\n';
			}
		}

		cpuSaxpy(GlobalVars::HIDDEN * GlobalVars::INPUT, &GlobalVars::GRADIENT_SCALAR, hiddenWeightsGradient, GlobalVars::ONE, hiddenWeights, GlobalVars::ONE);
		cpuSaxpy(GlobalVars::HIDDEN, &GlobalVars::GRADIENT_SCALAR, hiddenBiasGradient, GlobalVars::ONE, hiddenBias, GlobalVars::ONE);
		cpuSaxpy(GlobalVars::OUTPUT * GlobalVars::HIDDEN, &GlobalVars::GRADIENT_SCALAR, outputWeightsGradient, GlobalVars::ONE, outputWeights, GlobalVars::ONE);
		cpuSaxpy(GlobalVars::OUTPUT, &GlobalVars::GRADIENT_SCALAR, outputBiasGradient, GlobalVars::ONE, outputBias, GlobalVars::ONE);
	}
	
	/*cout << "Hidden Weights:\n";
	PrintMatrix(hiddenWeights, GlobalVars::INPUT, GlobalVars::HIDDEN);
	
	cout << "Hidden Bias:\n";
	PrintMatrix(hiddenBias, GlobalVars::ONE, GlobalVars::HIDDEN);

	cout << "Output Weights:\n";
	PrintMatrix(outputWeights, GlobalVars::HIDDEN, GlobalVars::OUTPUT);
	
	cout << "Output Bias:\n";
	PrintMatrix(outputBias, GlobalVars::ONE, GlobalVars::OUTPUT);*/

	return 0;
}