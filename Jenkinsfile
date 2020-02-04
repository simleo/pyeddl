pipeline {
    agent none
    stages {
        stage('Parallel Stages') {
            parallel {
                stage('linux') {
                    agent {
                        docker {
                            label 'docker'
                            image 'simleo/pyeddl-base:c023a6e'
                        }
                    }
                    stages {
                        stage('Build') {
                            steps {
				echo 'Building'
				sh 'python3 setup.py install --user'
                            }
                        }
                        stage('Test') {
                            steps {
				echo 'Testing'
				sh 'pytest tests'
				sh 'python3 examples/Tensor/eddl_tensor.py'
				sh 'python3 examples/NN/other/eddl_ae.py --epochs 1'
                            }
                        }
                        stage('linux_end') {
                            steps {
                                echo 'Success!'
                            }
                        }
                    }
                    post {
                        cleanup {
                            deleteDir()
                        }
                    }
                }
                stage('linux_gpu') {
                    agent {
                        docker {
                            label 'docker && gpu'
                            image 'simleo/pyeddl-gpu-base:c023a6e'
                            args '--gpus 1'
                        }
                    }
                    stages {
                        stage('Build') {
			    environment {
				EDDL_WITH_CUDA = 'true'
			    }
                            steps {
				echo 'Building'
				sh 'python3 setup.py install --user'
			    }
                        }
                        stage('Test') {
                            steps {
				echo 'Testing'
				sh 'pytest tests'
				sh 'python3 examples/Tensor/eddl_tensor.py --gpu'
				sh 'bash examples/NN/other/run_all_fast.sh'
				sh 'bash examples/NN/1_MNIST/run_all_fast.sh'
			    }
                        }
                        stage('linux_gpu_end') {
                            steps {
                                echo 'Success!'
                            }
                        }
                    }
                    post {
                        cleanup {
                            deleteDir()
                        }
                    }
                }
            }
        }
    }
}