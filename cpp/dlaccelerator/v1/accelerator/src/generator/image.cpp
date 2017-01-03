#include <random>
#include <iterator>
#include <iostream>
#include <algorithm>
#include <functional>

#include "image.h"

using namespace std;



vector<int> image_generator()
{
    // Create an instance of an engine.
    random_device rnd_device;
    // Specify the engine and distribution.
    mt19937 mersenne_engine(rnd_device());
    uniform_int_distribution<int> dist(0, 255);

    auto gen = std::bind(dist, mersenne_engine);
    vector<int> image(784);
    generate(begin(image), end(image), gen);

    /*
    // Print the vector
    for (auto i: image)
    {
        cout << i << " ";
    }
    */

    return image;
}
