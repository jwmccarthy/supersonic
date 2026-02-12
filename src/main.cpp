#include <iostream>
#include <chrono>

#include "RLEnvironment.cuh"

int main() {
    // Minimal initialization smoke test.
    const int sims = 1024;
    const int numB = 4;
    const int numO = 4;
    const int seed = 1234;
    const int iter = 10000;

    RLEnvironment env(sims, numB, numO, seed);
    std::cout << "RLEnvironment initialized.\n";
    
    env.reset();
    std::cout << "RLEnvironment reset.\n";

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iter; ++i)
    {
        env.step();
    }
    
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << iter << " steps in " << ms << " ms (" << (iter / ms * 1000) << " steps/sec)\n";

    return 0;
}
