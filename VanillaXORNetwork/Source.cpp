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
using std::exp;
using std::fabs;
using std::sqrt;
using std::min;
using std::max;
using std::ofstream;

/*
IMPORTANT LESSONS
1. For simple networks/tasks, a batch size may be detrimental to performance.
2. More complexity is needed to allow a path to the solution since batches may average out a path.
3. a network of 2x2x1 can be solved with no batches most of the times.
4. a network of 2x8x1 is needed if using batches to get the same performance as the simple network with no batches.
5. relu may or may not be causing certain paths to be unlearnable.
6. the initial weights and biases are important especially with relu.
7. a functional activation function apears to need nonconditional gradients, like if x > 1, apply gradient * 0.1. You can't say x can only go downwards since the activation limits x to 1.
(I attempted cluGradient so the gradient is 0 if either (x > 1 and gradient > 0) or (x < -1 and gradient < 0) because you cant go above 1 and below -1 due to the activation function, but it leads to weird results)
(may be because they only affect certain nodes and the rest are stun locked? everyone is fighting for the not stunned nodes. with the gradient * 0.1 if outside, at least nothing is stunned)
8. clu suprizingly performs better then relu when using 2x2x1 with big batch size.
(clu was able to solve the 2x2x1 network with a batchsize of 16 while relu couldn't)
(unofficial ranking: CLU, LeakyRELU, RELU)
9. The pros of CLU: no diminishing or exploding gradient, allows both negative and poitive numbers, and very fast
10. appareantly CLU can handle larger learning rates very well
(I just applied learning rate insead of learning rate / batch size)
11. CLU handles high and low batch sizes a lot better then leaky and relu
12. Applying learning rate / sqrt(batches) works alot better for CLU. leaky and reul have about the same performance.
*/

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
	constexpr uint32_t HIDDEN = 8;
	constexpr uint32_t OUTPUT = 2;
	constexpr uint32_t ITERATIONS = 1900;
	constexpr uint32_t BATCHES = 100;
	constexpr uint32_t ACTIVATIONS = 3;
	constexpr uint32_t RUNS = 20;
	constexpr uint32_t AVERAGES = 100;
	constexpr float ONEF = 1.0f;
	constexpr float ZEROF = 0.0f;
	constexpr float LEARNING_RATE = 100.0f;
	float GRADIENT_SCALAR = LEARNING_RATE / sqrt(BATCHES);
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
	for (int i = N; i--;)
		Y[i * incY] += *alpha * X[i * incX];
}

void cpuRelu(float* input, float* output, uint32_t size)
{
	for (uint32_t counter = size; counter--;)
		output[counter] = (input[counter] > 0) * input[counter];
}

void cpuReluDerivative(float* input, float* gradient, float* output, uint32_t size)
{
	for (uint32_t counter = size; counter--;)
		output[counter] = (input[counter] > 0) * gradient[counter];
}

void cpuLeakyRelu(float* input, float* output, uint32_t size)
{
	for (uint32_t counter = size; counter--;)
		output[counter] = ((input[counter] > 0) * 0.9f + 0.1f) * input[counter];
}

void cpuLeakyReluDerivative(float* input, float* gradient, float* output, uint32_t size)
{
	for (uint32_t counter = size; counter--;)
		output[counter] = ((input[counter] > 0) * 0.9f + 0.1f) * gradient[counter];
}

void cpuClu(float* input, float* output, uint32_t size)
{
	for (size_t counter = size; counter--;)
		output[counter] = min(0.1f, max(-0.1f, input[counter]));
}

void cpuCluDerivative(float* input, float* gradient, float* output, uint32_t size)
{
	for (size_t counter = size; counter--;)
		output[counter] = gradient[counter] * ((input[counter] >= -0.1f && input[counter] <= 0.1f) * 0.9f + 0.1f);
}

void cpuActivation(float* input, float* gradient, float* output, uint32_t size, uint32_t activation)
{
	switch (activation)
	{
	case 0:
		cpuRelu(input, output, size);
		break;
	case 1:
		cpuLeakyRelu(input, output, size);
		break;
	case 2:
		cpuClu(input, output, size);
		break;
	default:
		break;
	}
}

void cpuActivationDerivative(float* input, float* gradient, float* output, uint32_t size, uint32_t activation)
{
	switch (activation)
	{
	case 0:
		cpuReluDerivative(input, gradient, output, size);
		break;
	case 1:
		cpuLeakyReluDerivative(input, gradient, output, size);
		break;
	case 2:
		cpuCluDerivative(input, gradient, output, size);
		break;
	default:
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
	int gradient = endState - sampledProbability;
	for (uint32_t counter = size; counter--;)
		output[counter] = gradient * input[counter] * ((counter == action) - sampledProbability);
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

	float scores[GlobalVars::AVERAGES];
	float avgScore;
	uint32_t idx;
	
	float inputMatrix[GlobalVars::INPUT];
	float hiddenMatrix[GlobalVars::HIDDEN];
	float hiddenActivation[GlobalVars::HIDDEN];
	float outputMatrix[GlobalVars::OUTPUT];
	float softmaxMatrix[GlobalVars::OUTPUT];
	
	float hiddenWeights[GlobalVars::INPUT * GlobalVars::HIDDEN];
	float outputWeights[GlobalVars::HIDDEN * GlobalVars::OUTPUT];
	float hiddenBias[GlobalVars::HIDDEN];
	float outputBias[GlobalVars::OUTPUT];

	float hiddenGradient[GlobalVars::HIDDEN];
	float hiddenActivationGradient[GlobalVars::HIDDEN];
	float outputGradient[GlobalVars::OUTPUT];
	
	float hiddenWeightsGradient[GlobalVars::INPUT * GlobalVars::HIDDEN];
	float outputWeightsGradient[GlobalVars::HIDDEN * GlobalVars::OUTPUT];
	float hiddenBiasGradient[GlobalVars::HIDDEN];
	float outputBiasGradient[GlobalVars::OUTPUT];
	
	ofstream dataFile;
	dataFile.open("data.txt");
	dataFile << GlobalVars::ACTIVATIONS << '\n';
	dataFile << GlobalVars::RUNS << '\n';
	dataFile << GlobalVars::ITERATIONS << '\n';
	dataFile <<'\n';
		
	uint32_t activation = GlobalVars::ACTIVATIONS;
	while (activation--)
	{
		uint32_t run = GlobalVars::RUNS;
		while (run--)
		{
			memset(scores, 0, GlobalVars::AVERAGES * sizeof(float));
			avgScore = 0.0f;
			idx = 0;

			cpuGenerateUniform(hiddenWeights, GlobalVars::INPUT * GlobalVars::HIDDEN, -1.0f, 1.0f);
			cpuGenerateUniform(outputWeights, GlobalVars::HIDDEN * GlobalVars::OUTPUT, -1.0f, 1.0f);
			memset(hiddenBias, 0, GlobalVars::HIDDEN * sizeof(float));
			memset(outputBias, 0, GlobalVars::OUTPUT * sizeof(float));

			uint32_t iteration = GlobalVars::ITERATIONS;
			while (iteration--)
			{
				memset(hiddenWeightsGradient, 0, GlobalVars::INPUT * GlobalVars::HIDDEN * sizeof(float));
				memset(outputWeightsGradient, 0, GlobalVars::HIDDEN * GlobalVars::OUTPUT * sizeof(float));
				memset(hiddenBiasGradient, 0, GlobalVars::HIDDEN * sizeof(float));
				memset(outputBiasGradient, 0, GlobalVars::OUTPUT * sizeof(float));

				float averageScore = 0;
				uint32_t batch = GlobalVars::BATCHES;
				while (batch--)
				{
					bool expected = 0;
					for (uint32_t counter = GlobalVars::INPUT; counter--;)
					{
						inputMatrix[counter] = GlobalVars::random.Ruint32() & 1;
						expected |= (bool)inputMatrix[counter];
					}
					cpuSgemmStridedBatched(false, false,
						GlobalVars::HIDDEN, GlobalVars::ONEF, GlobalVars::INPUT,
						&GlobalVars::ONEF,
						hiddenWeights, GlobalVars::HIDDEN, GlobalVars::ZEROF,
						inputMatrix, GlobalVars::INPUT, GlobalVars::ZEROF,
						&GlobalVars::ZEROF,
						hiddenMatrix, GlobalVars::HIDDEN, GlobalVars::ZEROF,
						GlobalVars::ONEF);
					cpuActivation(hiddenMatrix, hiddenGradient, hiddenActivation, GlobalVars::HIDDEN, activation);
					cpuSgemmStridedBatched(false, false,
						GlobalVars::OUTPUT, GlobalVars::ONEF, GlobalVars::HIDDEN,
						&GlobalVars::ONEF,
						outputWeights, GlobalVars::OUTPUT, GlobalVars::ZEROF,
						hiddenActivation, GlobalVars::HIDDEN, GlobalVars::ZEROF,
						&GlobalVars::ZEROF,
						outputMatrix, GlobalVars::OUTPUT, GlobalVars::ZEROF,
						GlobalVars::ONEF);
					cpuSoftmax(outputMatrix, softmaxMatrix, GlobalVars::OUTPUT);

					float number = GlobalVars::random.Rfloat(0.0f, 1.0f);
					uint32_t action = 0;
					while (true)
					{
						number -= softmaxMatrix[action];
						if (number < 0) break;
						action++;
						action -= (action == GlobalVars::OUTPUT) * GlobalVars::OUTPUT;
					}
					bool endState = (bool)action == expected;
					
					cpuSoftmaxDerivative(softmaxMatrix, outputGradient, endState, action, GlobalVars::OUTPUT);
					cpuSgemmStridedBatched(true, false,
						GlobalVars::HIDDEN, GlobalVars::ONEF, GlobalVars::OUTPUT,
						&GlobalVars::ONEF,
						outputWeights, GlobalVars::OUTPUT, GlobalVars::ZEROF,
						outputGradient, GlobalVars::OUTPUT, GlobalVars::ZEROF,
						&GlobalVars::ZEROF,
						hiddenActivationGradient, GlobalVars::HIDDEN, GlobalVars::ZEROF,
						GlobalVars::ONEF);
					cpuActivationDerivative(hiddenMatrix, hiddenActivationGradient, hiddenGradient, GlobalVars::HIDDEN, activation);

					cpuSgemmStridedBatched(false, true,
						GlobalVars::OUTPUT, GlobalVars::HIDDEN, GlobalVars::ONEF,
						&GlobalVars::ONEF,
						outputGradient, GlobalVars::OUTPUT, GlobalVars::ZEROF,
						hiddenActivation, GlobalVars::HIDDEN, GlobalVars::ZEROF,
						&GlobalVars::ONEF,
						outputWeightsGradient, GlobalVars::OUTPUT, GlobalVars::ZEROF,
						GlobalVars::ONEF);
					cpuSgemmStridedBatched(false, true,
						GlobalVars::HIDDEN, GlobalVars::INPUT, GlobalVars::ONEF,
						&GlobalVars::ONEF,
						hiddenGradient, GlobalVars::HIDDEN, GlobalVars::ZEROF,
						inputMatrix, GlobalVars::INPUT, GlobalVars::ZEROF,
						&GlobalVars::ONEF,
						hiddenWeightsGradient, GlobalVars::HIDDEN, GlobalVars::ZEROF,
						GlobalVars::ONEF);
					cpuSaxpy(GlobalVars::HIDDEN, &GlobalVars::ONEF, hiddenGradient, GlobalVars::ONEF, hiddenBiasGradient, GlobalVars::ONEF);
					cpuSaxpy(GlobalVars::OUTPUT, &GlobalVars::ONEF, outputGradient, GlobalVars::ONEF, outputBiasGradient, GlobalVars::ONEF);

					/*if (iteration == 0)
					{
						cout << "Expected: " << expected << '\n';
						cout << "Output: " << outputActivation[0] << '\n';
						cout << '\n';
					}*/

					averageScore += endState;
				}
				
				avgScore -= scores[idx];
				avgScore += averageScore / GlobalVars::BATCHES;
				scores[idx++] = averageScore / GlobalVars::BATCHES;
				idx -= (idx == GlobalVars::AVERAGES) * GlobalVars::AVERAGES;
				dataFile << avgScore / GlobalVars::AVERAGES << ' ';

				/*if (debug && (iteration == 0))
				{
					cout << "inputMatrix:\n";
					PrintMatrix(inputMatrix, GlobalVars::ONEF, GlobalVars::INPUT);
					cout << "hiddenMatrix:\n";
					PrintMatrix(hiddenMatrix, GlobalVars::ONEF, GlobalVars::HIDDEN);
					cout << "hiddenActivation:\n";
					PrintMatrix(hiddenActivation, GlobalVars::ONEF, GlobalVars::HIDDEN);
					cout << "outputMatrix:\n";
					PrintMatrix(outputMatrix, GlobalVars::ONEF, GlobalVars::OUTPUT);
					cout << "outputActivation:\n";
					PrintMatrix(outputActivation, GlobalVars::ONEF, GlobalVars::OUTPUT);
					cout << "outputActivationGradient:\n";
					PrintMatrix(outputActivationGradient, GlobalVars::ONEF, GlobalVars::OUTPUT);
					cout << "outputGradient:\n";
					PrintMatrix(outputGradient, GlobalVars::ONEF, GlobalVars::OUTPUT);
					cout << "hiddenActivationGradient:\n";
					PrintMatrix(hiddenActivationGradient, GlobalVars::ONEF, GlobalVars::HIDDEN);
					cout << "hiddenGradient:\n";
					PrintMatrix(hiddenGradient, GlobalVars::ONEF, GlobalVars::HIDDEN);
					cout << "outputWeightsGradient:\n";
					PrintMatrix(outputWeightsGradient, GlobalVars::HIDDEN, GlobalVars::OUTPUT);
					cout << "hiddenWeightsGradient:\n";
					PrintMatrix(hiddenWeightsGradient, GlobalVars::INPUT, GlobalVars::HIDDEN);
					cout << "outputBiasGradient:\n";
					PrintMatrix(outputBiasGradient, GlobalVars::ONEF, GlobalVars::OUTPUT);
					cout << "hiddenBiasGradient:\n";
					PrintMatrix(hiddenBiasGradient, GlobalVars::ONEF, GlobalVars::HIDDEN);
					cout << "hiddenWeights:\n";
					PrintMatrix(hiddenWeights, GlobalVars::INPUT, GlobalVars::HIDDEN);
					cout << "outputWeights:\n";
					PrintMatrix(outputWeights, GlobalVars::HIDDEN, GlobalVars::OUTPUT);
					cout << "hiddenBias:\n";
					PrintMatrix(hiddenBias, GlobalVars::ONEF, GlobalVars::HIDDEN);
					cout << "outputBias:\n";
					PrintMatrix(outputBias, GlobalVars::ONEF, GlobalVars::OUTPUT);
				}*/
				
				cpuSaxpy(GlobalVars::INPUT* GlobalVars::HIDDEN, &GlobalVars::GRADIENT_SCALAR, hiddenWeightsGradient, GlobalVars::ONEF, hiddenWeights, GlobalVars::ONEF);
				cpuSaxpy(GlobalVars::HIDDEN, &GlobalVars::GRADIENT_SCALAR, hiddenBiasGradient, GlobalVars::ONEF, hiddenBias, GlobalVars::ONEF);
				cpuSaxpy(GlobalVars::HIDDEN* GlobalVars::OUTPUT, &GlobalVars::GRADIENT_SCALAR, outputWeightsGradient, GlobalVars::ONEF, outputWeights, GlobalVars::ONEF);
				cpuSaxpy(GlobalVars::OUTPUT, &GlobalVars::GRADIENT_SCALAR, outputBiasGradient, GlobalVars::ONEF, outputBias, GlobalVars::ONEF);
			}
			dataFile << '\n';
		}
		dataFile << '\n';
	}
	dataFile.close();

	return 0;
}