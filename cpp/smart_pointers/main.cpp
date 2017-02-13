#include <memory>
#include <iostream>


class Test
{
    public:
        Test(int a = 0):m_a(a) {}
        ~Test() { std::cout << "Calling destructor" << std::endl; }

    public:
        int m_a;
};


void func(std::shared_ptr<Test> p)
{
    std::cout << p->m_a << std::endl;
}


int main()
{
    std::shared_ptr<Test> p(new Test(4));
    func(p);

    return 0;
}
