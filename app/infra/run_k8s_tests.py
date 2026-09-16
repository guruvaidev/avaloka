import unittest
import argparse
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def run_tests(platform):
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'infra')

    if platform == 'gcp':
        print("Running GCP Kubernetes deployment tests...")
        suite.addTest(loader.discover(test_dir, pattern='test_k8s_deployment.py'))
    elif platform == 'aws':
        print("Running AWS EKS deployment tests...")
        suite.addTest(loader.discover(test_dir, pattern='test_eks_deployment.py'))
    else:
        print("Invalid platform specified. Use 'gcp' or 'aws'.")
        return

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Kubernetes deployment tests for a specific cloud platform.')
    parser.add_argument('--platform', required=True, choices=['gcp', 'aws'], help='Specify the cloud platform (gcp or aws).')
    args = parser.parse_args()
    run_tests(args.platform)