from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app', 'infra', '.env'))
import unittest
import time
import requests
import subprocess
import json
import pandas as pd
from langgraph.graph import StateGraph
from app.api.workflow import build_graph
from app.graph.etl_state import ETLState
from langchain_core.messages import HumanMessage

class TestE2EGroupbySumGKE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure environment variables are set for GCP
        cls.project_id = os.environ.get("GCP_PROJECT_ID")
        cls.cluster_name = os.environ.get("GCP_CLUSTER_NAME", "test-gke-cluster")
        cls.zone = os.environ.get("GCP_ZONE", "us-central1-c")
        cls.sample_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app', 'sample_data', 'salaries.csv')

        if not cls.project_id:
            raise unittest.SkipTest("GCP_PROJECT_ID environment variable not set. Skipping E2E GKE test.")

        # Build the Docker image for the Python app
        python_app_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app', 'infra', 'python_app')
        print(f"Building Docker image from {python_app_dir}...")
        build_cmd = ["docker", "build", "-t", "infra-agent-python-app:latest", python_app_dir]
        try:
            subprocess.run(build_cmd, check=True, capture_output=True, text=True)
            print("Docker image built successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error building Docker image: {e.stderr}")
            raise

        # Initialize the LangGraph workflow
        cls.graph = build_graph().compile()

    @classmethod
    def tearDownClass(cls):
        # Cleanup resources if the cluster was created by the test
        # This part is handled by k8s_invoker.cleanup_resources, but can be explicitly called here if needed
        pass

    def test_e2e_groupby_sum_on_gke(self):
        print("\n--- Starting E2E Groupby Sum on GKE Test ---")

        # 1. Initial state: Request GKE infrastructure and groupby sum operation
        initial_state = ETLState(
            messages=[HumanMessage(content="I want to group the salaries.csv file by 'job_title' and calculate the sum of 'salary_in_usd' on a GKE cluster using a Python Docker application.")],
            planner_definition={},
            ready_to_summarize=False,
            ready_to_code=False,
            coder_definition={},
            infrastructure_request={"type": "gcp", "app_type": "python-docker"},
            infrastructure_provisioned=None
        )

        # 2. Invoke the graph
        print("Invoking LangGraph workflow...")
        final_state = self.graph.invoke(initial_state)
        print("LangGraph workflow invocation complete.")

        # 3. Verify infrastructure provisioning and application deployment
        self.assertIn("infrastructure_provisioned", final_state)
        self.assertIsNotNone(final_state["infrastructure_provisioned"])
        self.assertEqual(final_state["infrastructure_provisioned"]["status"], "provisioned")
        self.assertEqual(final_state["infrastructure_provisioned"]["type"], "gcp")
        self.assertEqual(final_state["infrastructure_provisioned"]["app_type"], "python-docker")

        print("Infrastructure provisioned and application deployed successfully (according to agent).")

        # 4. Get service IP and Port
        service_ip = None
        service_port = None
        for outcome in final_state["infrastructure_provisioned"]["details"]:
            if outcome["step_name"] == "Get infra-agent-service IP/Port" and outcome["status"] == "SUCCESS":
                service_ip = outcome["details"]["ip"]
                service_port = outcome["details"]["port"]
                break
        
        self.assertIsNotNone(service_ip, "Could not retrieve service IP for infra-agent-service.")
        self.assertIsNotNone(service_port, "Could not retrieve service Port for infra-agent-service.")
        print(f"Infra-agent service accessible at {service_ip}:{service_port}")

        # Wait for 30 seconds for the service to be ready
        print("Waiting for 30 seconds for the service to be ready...")
        time.sleep(30)
    
        # 5. Send data to the deployed application and get result
        print("Sending sample data to the deployed Python application...")
        url = f"http://{service_ip}:{service_port}/groupby_sum"
        
        with open(self.sample_data_path, 'rb') as f:
            files = {'file': ('salaries.csv', f, 'text/csv')}
            data = {'group_by_column': 'job_title', 'sum_column': 'salary'}
            
            try:
                response = requests.post(url, files=files, data=data) # 5 minutes timeout
                response.raise_for_status() # Raise an exception for HTTP errors
                result = response.json()
                print("Received response from Python application.")
            except requests.exceptions.RequestException as e:
                self.fail(f"Failed to connect to deployed application or receive valid response: {e}")
            except json.JSONDecodeError:
                self.fail(f"Failed to decode JSON response: {response.text}")

        # 6. Assert the correctness of the result
        # Calculate expected results from the actual CSV data
        result_df = pd.DataFrame(result)
        
        # Expected results based on actual salaries.csv data:
        # Manager: 10000 + 90000 = 100000
        # Producer: 8000
        # Engineer: 100000 + 120000 = 220000
        # Director: 110000
        expected_result_subset = [
            {'job_title': 'Director', 'salary': 110000}, 
            {'job_title': 'Engineer', 'salary': 220000}, 
            {'job_title': 'Manager', 'salary': 100000},
            {'job_title': 'Producer', 'salary': 8000}
        ]
        expected_result_df = pd.DataFrame(expected_result_subset)
        
        # Sort both dataframes by job_title for consistent comparison
        result_df_sorted = result_df.sort_values('job_title').reset_index(drop=True)
        expected_result_df_sorted = expected_result_df.sort_values('job_title').reset_index(drop=True)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertTrue(result_df_sorted.equals(expected_result_df_sorted), 
                       f"Expected: {expected_result_df_sorted.to_dict('records')}, Got: {result_df_sorted.to_dict('records')}")
        
        # Example of a more robust check:
        # Load the CSV and calculate expected result
        # df_sample = pd.read_csv(self.sample_data_path)
        # expected_df = df_sample.groupby('job_title')['salary'].sum().reset_index()
        # expected_dict = expected_df.set_index('job_title')['salary'].to_dict()

        # for item in result:
        #     job_title = item.get('job_title')
        #     salary_sum = item.get('salary_in_usd')
        #     self.assertIn(job_title, expected_dict)
        #     self.assertEqual(salary_sum, expected_dict[job_title])

        print("Groupby sum result verified successfully.")
        print("--- E2E Groupby Sum on GKE Test Completed Successfully ---")

if __name__ == "__main__":
    unittest.main()
