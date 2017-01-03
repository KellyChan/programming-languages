/*

ELM: Extreme Learning Machine

    Neural Model:

        1. input neurons: X

        2. hidden neurons:
               calculation (linear): Y = W1 * X + B1
               activiation (tanh)  : Y1 = tanh(Y)

                                    where tanh(Y) = sinh(Y) / cosh(Y)
                                                  = (e^(Y) - e^(-Y)) / (e^(Y) + e^(-Y))
                                          

        3. output neurons:
               calculation (linear): Y_Hat = W2 * Y1 + B2

        * NOTE: B1, B2 = 0 


    Train:

*/

#include <math.h>                         // tanh, log
#include <stdio.h>                        // printf

#include <string>
#include <fstream>
#include <iostream>

#include "generator/image.h"
#include "generator/weights.h"
#include "layers/linear.h"
#include "layers/activator.h"
#include "proto/ptout/elm.pb.h"

using namespace std;


// Tanh Layer: Hyperbolic tangent function
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


void PrintMatrix(const Matrix& X, string name)
{

    cout << "\n --- " << name << " --- \n" << endl;

    int nrows = X.size();
    int ncols = X[0].size();
    
    for (int i = 0; i < nrows; ++i)
    {
        for (int j = 0; j < ncols; ++j)
        {
            cout << X[i][j] << " ";
        }
        cout << endl;
    }
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
    elm(w1, w2, input);

    vector<int> image = image_generator();
    int X_rows = 784;     // 784
    int X_cols = 1;      // 1
    Matrix InImg = weights_generator(X_rows, X_cols);

    int W1_rows = 7840;  // 7840
    int W1_cols = 784;   // 784
    Matrix W1 = weights_generator(W1_rows, W1_cols);
    int W2_rows = 10;    // 10
    int W2_cols = 7840;  // 7840
    Matrix W2 = zero_weights_generator(W2_rows, W2_cols);

    Matrix hidden_interim = linear_fn_without_bias(W1, InImg);
    Matrix hidden_output = activator_tanh(hidden_interim);
    Matrix final_output = linear_fn_without_bias(W2, hidden_output);

    
    // PrintMatrix(W1, "W1");
    // PrintMatrix(InImg, "InImg");
    // PrintMatrix(hidden_interim, "hidden_interim");
    // PrintMatrix(hidden_output, "hidden_output");
    // PrintMatrix(W2, "W2");
    PrintMatrix(final_output, "final_output");
    

    return 0;
}
