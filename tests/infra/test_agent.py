import unittest
import os
import subprocess

class TestInfraAgent(unittest.TestCase):

    def setUp(self):
        # Create a dummy CSV file
        with open("test.csv", "w") as f:
            f.write("col1,col2\n")
            f.write("1,2\n")

        # Create a dummy python script
        with open("process.py", "w") as f:
            f.write("import pandas as pd\n")
            f.write("df = pd.read_csv('test.csv')\n")
            f.write("print(df.head())\n")

        # Create a dummy Dockerfile
        with open("Dockerfile", "w") as f:
            f.write("FROM python:3.9-slim\n")
            f.write("COPY . .\n")
            f.write("RUN pip install pandas\n")
            f.write("CMD [\"python\", \"process.py\"]\n")

    def tearDown(self):
        os.remove("test.csv")
        os.remove("process.py")
        os.remove("Dockerfile")
        subprocess.run(["docker", "rmi", "test-image"], capture_output=True)


    def test_process_csv(self):
        # Build the docker image
        subprocess.run(["docker", "build", "-t", "test-image", "."], check=True)

        # Run the docker container
        result = subprocess.run(["docker", "run", "test-image"], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0)
        self.assertIn("col1", result.stdout)
        self.assertIn("col2", result.stdout)
        self.assertIn("1", result.stdout)
        self.assertIn("2", result.stdout)

if __name__ == '__main__':
    unittest.main()
