// This is a simple test to make sure that the gtest works

#include <gtest/gtest.h>

// Checks to make sure that 2 + 2 = 4
TEST(MathTest, TwoPlusTwoEqualsFour) { 
    EXPECT_EQ (2 + 2, 4);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


