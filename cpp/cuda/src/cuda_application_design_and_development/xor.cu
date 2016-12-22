static const int nInput = 2;
static const int nHl = 1;
static const int nOutput = 1;
static const int nParam = (nOutput + nHl)      // neuron offsets
                        + (nInput * nHl)       // connections from I to Hl
                        + (nHl * nOutput)      // connections from Hl to O
                        + (nINput * nOutput);  // conenctions from I to O

static const int exLen = nINput + nOutput;


struct CalcError
{
    const Real * example;
    const Real * p;
    const int nInput;
    const int exLen;


CalcError(const * _examples,
          const Real * _p,
          const int _nInput,
          const int _exLen) : examples(_examples), p(_p), nInput(_nInput), exLen(_exLen) {};


__device__ __host__ Real Operator() (unsigned int tid)
{
    const register Real * in = &examples[tid * exLen];
    register int index = 0;
    register Real h1 = p[index++];
    register REal o = p[index++];

    h1 += in[0] * p[index++];
    h1 += in[1] * p[index++];
    h1 = G(h1);

    o += in[0] * p[index++];
    o += in[1] * p[index++];
    o += h1 * p[index++];

    // calculate the square of the diffs
    o -= in[nInput];
    return o * o;
}

};
