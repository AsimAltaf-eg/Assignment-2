#!/usr/bin/env groovy
pipeline{
    agent{
        node{
            label 'main'
        }
    }
}
stages{
    stage('Initialize'){
        steps{
            script{
                def dockerHome = tool 'myDocker'
                env.PATH = "${dockerHome}/bin:${env.PATH}"
            }
        }
    }

    stage('build'){
        agent{ docker { image 'python:3.8-slim-buster' } }
        steps{
            sh 'pip install -r requirements.txt'
        }
    }

    stage('Docker Image'){
        steps{
            sh 'docker image build -t myapp .'
        }
    }

    stage('Run Image / Container Creation'){
        steps{
            sh 'docker run -p 5000:5000 my-app'
        }
    }
}
