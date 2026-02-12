#include <iostream>

#include "RLEnvironment.cuh"

int main() {
    // Minimal initialization smoke test.
    const int sims = 1024;
    const int numB = 4;
    const int numO = 4;
    const int seed = 1234;

    RLEnvironment env(sims, numB, numO, seed);
    std::cout << "RLEnvironment initialized.\n";
    
    env.reset();
    std::cout << "RLEnvironment reset.\n";

    return 0;
}
