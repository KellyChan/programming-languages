#include <string>
#include <fstream>
#include <iostream>

#include "./ptout/elm.pb.h"

using namespace std;


// PromptForELMParams: Fills in the parameters of a ELM model based on user input.
void PromptELMCONF(ELMCONF::ELM* elm)
{
    cout << "Enter ELM Input (X): ";
    float x;
    cin >> x;
    elm->set_x(x);

    cout << "Enter ELM Params (w1): ";
    float w1;
    cin >> w1;
    elm->set_w1(w1);

    cout << "Enter ELM Params (w2): ";
    float w2;
    cin >> w2;
    elm->set_w2(w2);
}


// Main function:
// - reads the entire ELMCONF from a file;
// - adds the parameters based on user input;
// - writes it back out to the same file.
int main(int argc, char* argv[])
{
    // Verify that the version of the library that we linked against is
    // compatiable with the veresion of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <FilePath_ToBeSaved>" << endl;
        return -1;
    }

    ELMCONF::ELM elm;

    {
        // read the existing ELMCONF file
        fstream input(argv[1], ios::in | ios::binary);
        if (!input)
        {
            cout << argv[1] << ": File not found. Creating a new file." << endl;
        }
        else if (!elm.ParseFromIstream(&input))
        {
            cerr << "Failed to parse ELMCONF file." << endl;
            return -1;
        }
    }

    // Add a ELMCONF file
    PromptELMCONF(&elm);

    {
        // Write the new ELMCONF file back to disk.
        fstream output(argv[1], ios::out | ios::trunc | ios::binary);
        if (!elm.SerializeToOstream(&output))
        {
            cerr << "Failed to write ELMCONF file." << endl;
            return -1;
        }
    }

    // Optional: Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

    return 0;
}

