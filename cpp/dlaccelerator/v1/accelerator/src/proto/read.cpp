#include <string>
#include <fstream>
#include <iostream>

#include "./ptout/elm.pb.h"

using namespace std;


// List ELMCONF parameters
void ListELMCONF(const ELMCONF::ELM& elm)
{
    cout << "ELM Input (x): " << elm.x() << endl;
    cout << "ELM Params (w1): " << elm.w1() << endl;
    cout << "ELM Params (w2): " << elm.w2() << endl;
}


// Main function:
// - reads the entire ELMCONF from a file;
// - prints all the parameters inside.
int main(int argc, char* argv[])
{
    // Verify that the version of the library that we linked against is compatible
    // with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <ELMCONF_File>" << endl;
        return -1;
    }

    ELMCONF::ELM elm;

    {
        // Read the existing ELMCONF file.
        fstream input(argv[1], ios::in | ios::binary);
        if (!elm.ParseFromIstream(&input))
        {
            cerr << "Failed to parse ELMCONF file." << endl;
            return -1;
        }
    }

    ListELMCONF(elm);

    // Optional: Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

    return 0;
}
