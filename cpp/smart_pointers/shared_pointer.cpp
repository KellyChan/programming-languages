#include <memory>

int main()
{
    std::shared_ptr<int> sptr1 = std::make_shared<int>(100);
    std::shared_ptr<int> sptr2 (new int);

    return 0;
}
