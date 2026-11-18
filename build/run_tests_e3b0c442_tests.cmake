add_test([=[MargrabeBenchmark.MatchesAnalyticalPriceWithin3StdErrors]=]  /Users/zilongchang/Documents/Documents/multi-asset-pricer/build/run_tests [==[--gtest_filter=MargrabeBenchmark.MatchesAnalyticalPriceWithin3StdErrors]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[MargrabeBenchmark.MatchesAnalyticalPriceWithin3StdErrors]=]
  PROPERTIES
    
    _CMAKE_TEST_BUILD_DEPENDS [==[/Users/zilongchang/Documents/Documents/multi-asset-pricer/build/run_tests]==]
    DEF_SOURCE_LINE [==[/Users/zilongchang/Documents/Documents/multi-asset-pricer/tests/test_margrabe.cpp:18]==]
    WORKING_DIRECTORY [==[/Users/zilongchang/Documents/Documents/multi-asset-pricer/build]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    [[_CMAKE_DEFAULT_FIXTURE_REPEAT_MODE]] [[AROUND_EACH_REPEAT]]
)
set(run_tests_TESTS [==[MargrabeBenchmark.MatchesAnalyticalPriceWithin3StdErrors]==])
