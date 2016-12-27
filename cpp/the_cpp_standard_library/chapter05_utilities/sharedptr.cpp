#include <iostream>
#include <string>
#include <vector>
#include <memory>

using namespace std;


int main()
{
    shared_ptr<string> pNico(new string("nico"));
    shared_ptr<string> pJutta(new string("jutta"));

    (*pNico)[0] = "N";
    pJutta->replace(0, 1, "J");

    vector<shared_ptr<string>> whoMadeCoffee;
    whoMadeCoffee.push_back(pJutta);
    whoMadeCoffee.push_back(pJutta);
    whoMadeCoffee.push_back(pNico);
    

    for (auto ptr: whoMadeCoffee) {
        cout << *ptr << " ";
    }
}
