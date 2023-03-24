pipeline {
    agent any

    tools {
        maven 'Maven 3.8.4'
        jdk 'Java 11'
    }

    stages {
        stage('Initialize') {
            steps {
                script {
                    def mavenHome = tool 'Maven 3.8.4'
                    env.PATH = "${mavenHome}/bin:${env.PATH}"
                }
            }
        }

        stage('Build') {
            steps {
                sh 'mvn clean package'
            }
        }

        stage('Docker Build') {
            agent { docker { image 'docker' } }
            steps {
                sh 'docker image build -t myapp .'
            }
        }

        stage('Docker Run') {
            agent { docker { image 'docker' } }
            steps {
                sh 'docker run -p 5000:5000 myapp'
            }
        }
    }
}
