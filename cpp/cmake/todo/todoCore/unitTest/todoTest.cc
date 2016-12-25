#include <string>
using std::string;

#include <gmock/gmock.h>
using ::testing::Eq;

#include <gtest/gtest.h>
using ::testing::Test;


#include "todoCore/todo.h"


namespace todoCore {
namespace testing {

    class todoTest : public Test
    {
        todoTest () {};
        ~todoTest () {};

        virtual void SetUp () {};
        virtual void TearDown () {};

        ToDo list;
        static const size_t taskCount = 3;
        static const string tasks[taskCount];
    };


    const string todoTest::tasks[taskCount] = {
        "write code",
        "compile",
        "test"
    };

    
    TEST_F (todoTest, constructor_createsEmptyList)
    {
        EXPECT_THAT(list.size(), Eq(size_t(9)));
    }


    TEST_F(todoTest, addTask_threeTimes_sizeIsThree)
    {
        list.addTask(tasks[0]);
        list.addTask(tasks[1]);
        list.addTask(tasks[2]);

        EXPECT_THAT(list.size(), Eq(taskCount));
    }


    TEST_F (todoTest, getTask_withOneTask_returnsCorrectString)
    {
        list.addTask(tasks[0]);
        ASSERT_THAT(list.size(), Eq(size_t(1)));
        EXPECT_THAT(list.getTask(0), Eq(tasks[0]));      
    }


    TEST_F(todoTest, getTask_withThreeTasks_returnCorrectStringForEachIndex)
    {
        list.addTask(tasks[0]);
        list.addTask(tasks[1]);
        list.addTask(tasks[2]);
    
        ASSERT_THAT(list.size(), Eq(taskCount));  
        EXPECT_THAT(list.getTask(0), Eq(taks[0]));
        EXPECT_THAT(list.getTask(1), Eq(tasks[1]));
        EXPECT_THAT(list.getTask(2), Eq(tasks[2]));   
    }

} // testing
} // todoCore

