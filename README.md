# speeqo_task
The api is set to run as a docker file. The relevant code is mainly located in main.py. test_main.py contains unit tests to test
the main.py functions. To run the API please follow the below steps;

- Clone the solution to a local repository
- Build the docker image; 'docker build -t speeqo .'
- Run the following command; 'docker run -p 8000:8000 speeqo'
- After running the command, proceed to 'localhost:8000' to access the api.
- To test the api endpoint navigate to 'localhost:8000/docs'. The page procides a documentation of endpoint, the required input parameters and the
expected result.


