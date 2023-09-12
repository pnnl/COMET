
import glob

import subprocess

files = glob.glob("test_*.py")


print("\nFound" , len(files), "test cases")

print("Running the tests......")

failed_tests = 0
list_failed_tests = []

for test_file in files:

    test_result = subprocess.call(['python3', test_file])

    if(test_result != 0):
        print("=======================================================")
        print(test_file)
        failed_tests = failed_tests + 1
        list_failed_tests.append(test_file)


print("**********************************************\n")
print("Passed = ", len(files) - failed_tests)

print("Failed = " , failed_tests)

if(list_failed_tests):
    print("Following tests failed:")

    for failed_test in list_failed_tests:
        print('  ' , failed_test)