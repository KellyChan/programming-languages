/*

ELM: Extreme Learning Machine

    Neural Model:

        1. input neurons: X

        2. hidden neurons:
               calculation (linear): Y = W1 * X + b1
               activiation (tanh)  : Y1 = tanh(Y)

        3. output neurons:
               calculation (linear): Y_Hat = W2 * Y1 + b2

        * NOTE: b1, b2 = 0 


    Train:

*/

#include <math.h>                         // tanh, log
#include <stdio.h>                        // printf

#include <string>
#include <fstream>
#include <iostream>
#include "proto/ptout/elm.pb.h"

using namespace std;

/*

Tanh Layer: Hyperbolic tangent function

  Calculation:
      tanh(x) = sinh(x) / cosh(x)
              = (e^x - e^(-x)) / (e^x + e^(-x))
*/
float activation_tanh(float z)
{
    float output;

    output = tanh(z);
    printf("tanh(%f) = %f\n", z, output);
 
    return output;
}


float multiply(float w, float x)
{
    float output;
    output = w * x;
    printf("%f * %f = %f\n", w, x, output);

    return output;
}


int elm(float w1, float w2, float input)
{
    // hidden layer
    float hidden_interim = multiply(w1, input);   
    float hidden_output = activation_tanh(hidden_interim);

    // output layer
    float output = multiply(w2, hidden_output);

    return 0;
}


void GetELMCONF(const ELMCONF::ELM& elmconf)
{
    cout << "ELM Params (x): " << elmconf.x() << endl;
    cout << "ELM Params (w1): " << elmconf.w1() << endl;
    cout << "ELM Params (w2): " << elmconf.w2() << endl;
}


int main(int argc, char* argv[])
{

    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <ELMCONF_FILE>" << endl;
        return -1;
    }

    
    // ELMCONF
    ELMCONF::ELM elmconf;
    {
        // Read the existing ELMCONF file.
        fstream input(argv[1], ios::in | ios::binary);
        if (!elmconf.ParseFromIstream(&input))
        {
            cerr << "Failed to parse ELMCONF file." << endl;
            return -1;
        }
    }

    GetELMCONF(elmconf);

    float w1 = elmconf.w1();
    float w2 = elmconf.w2();
    float input = elmconf.x();

    // float w1 = 2;
    // float w2 = 3;
    // float input = 4;
    elm(w1, w2, input);

    return 0;
}
