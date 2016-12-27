#include <string>
#include <fstream>
#include <memory>
#include <cstdio>


class FileDeleter
{
    private:
        std::string filename;

    public:
        FileDeleter (const std::string& fn) : filename (fn) {}
        void operator () (std::ofstream* fp) {
            fp->close();
            std::remove(filename.c_str());
        }
};


int main()
{
    std::shared_ptr<std::ofstream> fp(new std::ofstream("tmpfile.txt"),
                                      FileDeleter("tmpfile.txt"));
}
